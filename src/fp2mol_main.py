"""
fp2mol_main.py — DDP-compatible decoder training / pretraining entry point.

Serves both dataset families through one script:

  * ``dataset.name=fp2mol``  -> fingerprint-molecule pretraining
    (``--config-name=config_decoder_pretrain``)
  * every other supported name -> NEIMS-style spectrum datasets
    (default ``config_decoder``)

DDP flow
--------
Mirrors spec2mol_main.py. All ranks enter main() simultaneously (srun, one task
per GCD, RANK/LOCAL_RANK/WORLD_SIZE exported by the launcher). Two things must
happen exactly once before any rank can build a model:

  1. The PyG ``processed/*.pt`` files must exist. They are written by
     ``FP2MolDataset.process`` / ``NeimsDataset.process``, which runs from the
     DataModule constructor -- i.e. on every rank, concurrently, into the same
     paths.
  2. The dataset stat files (n_counts / atom_types / edge_types / valencies)
     must exist. ``*_infos`` computes and ``np.savetxt``s them when they are
     missing -- again on every rank, into the same paths.

Both are guarded with the same file-based sentinel spec2mol uses, so no
torch.distributed call is needed before trainer.fit():

  Rank 0: build the DataModule + DatasetInfos (processing and stat computation
          happen here if needed), then touch the sentinel.
  Others: poll for the sentinel, then build the same objects -- by then a pure
          file read.

After that all ranks construct the model and call trainer.fit(); Lightning's
DDPStrategy injects the DistributedSampler.
"""

import os
import sys
import math
import time
import signal
import pathlib
import warnings
import logging
import datetime
import faulthandler
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from rdkit import RDLogger


from src import utils
from src.diffusion_model_fp2mol import FP2MolDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from analysis.visualization import MolecularVisualization

from datasets import fp2mol_dataset
from datasets import neims_dataset

dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

warnings.filterwarnings("ignore", category=PossibleUserWarning)
RDLogger.DisableLog("rdApp.*")

# See spec2mol_main.py for the rationale on both of the following: Lightning's
# _pytree.py still calls torch's deprecated `LeafSpec` API (one line per rank at
# dataloader setup), and DDP deliberately stashes the AccumulateGrad node across
# iterations, which trips a stream-mismatch check on every backward pass.
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
)
if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

# Same values spec2mol_dataset uses. Duplicated rather than imported so this
# entry point does not pull in the MIST/spectrum stack it never needs.
_SENTINEL_POLL_INTERVAL = 5.0
_SENTINEL_TIMEOUT = 7200


def _ddp_env():
    """Return (local_rank, global_rank, world_size) from the launcher's env vars."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, global_rank, world_size


def _sentinel_path(cfg) -> str:
    """Path of the 'dataset is built' marker for this exact dataset variant.

    Keyed on the dataset name and the Morgan parameters because those select the
    ``processed/morgan_r-*__morgan_nbits-*`` subdirectory: two runs differing only
    in fingerprint size need separate processing, so they must not share a marker.
    """
    root = os.path.join(getattr(cfg.general, "parent_dir", "."), cfg.dataset.datadir)
    name = f".dataset_ready__{cfg.dataset.name}__r{cfg.dataset.morgan_r}__n{cfg.dataset.morgan_nbits}"
    return os.path.join(root, name)


def get_datamodule(cfg):
    """Build the datamodule and dataset infos for the configured dataset.

    'fp2mol' is the fingerprint-molecule pretraining dataset; every other
    supported dataset goes through the NEIMS pipeline. Both constructors do real
    work on first call (graph processing, stat computation) -- see
    _ensure_dataset_ready for why that must not happen on all ranks at once.
    """
    name = cfg.dataset.name

    if name == "fp2mol":
        datamodule = fp2mol_dataset.FP2MolDataModule(cfg)
        dataset_infos = fp2mol_dataset.FP2Mol_infos(
            datamodule, cfg, recompute_statistics=False
        )
    elif name in (
        "neims",
        "neims_tms",
        "gecko_atmomaccs",
        "msg_neims",
        "mixed_augment_test",
        "gecko_new",
    ):
        datamodule = neims_dataset.NeimsDataModule(cfg)
        dataset_infos = neims_dataset.Neims_infos(
            datamodule, cfg, recompute_statistics=False
        )
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    logging.info(f"{name} config loaded")
    return datamodule, dataset_infos


def _ensure_dataset_ready(cfg, global_rank: int, is_ddp: bool):
    """Build the datamodule/infos, letting rank 0 go first.

    Rank 0 builds them (writing processed graphs and stat files if absent) and
    touches the sentinel; other ranks poll for the sentinel and then build from
    the finished files. Without this every rank races on the same
    ``torch.save``/``np.savetxt`` targets, which produces truncated .pt files and
    interleaved stat files rather than a clean error.
    """
    sentinel = _sentinel_path(cfg)

    if global_rank != 0 and is_ddp:
        if not os.path.exists(sentinel):
            logging.warning(
                f"Rank {global_rank}: waiting for rank 0 to finish dataset setup..."
            )
            start = time.time()
            while not os.path.exists(sentinel):
                if time.time() - start > _SENTINEL_TIMEOUT:
                    raise TimeoutError(
                        f"Rank {global_rank} timed out ({_SENTINEL_TIMEOUT}s) "
                        f"waiting for sentinel '{sentinel}'."
                    )
                time.sleep(_SENTINEL_POLL_INTERVAL)
        return get_datamodule(cfg)

    if not os.path.exists(sentinel):
        logging.info(
            "Rank 0: dataset sentinel not found — processing graphs and computing "
            "stats if needed (one-time setup, may take several minutes)."
        )
    datamodule, dataset_infos = get_datamodule(cfg)
    if not os.path.exists(sentinel):
        os.makedirs(os.path.dirname(sentinel), exist_ok=True)
        pathlib.Path(sentinel).touch()
        logging.info(f"Rank 0: sentinel written → {sentinel}")
    return datamodule, dataset_infos


# ---------------------------------------------------------------------------
# Resume / checkpoint helpers
# ---------------------------------------------------------------------------


def get_resume(cfg, model_kwargs):
    """Resumes a run. It loads previous config without allowing to update keys (used for testing)."""
    saved_cfg = cfg.copy()
    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only
    val_samples_to_generate = cfg.general.val_samples_to_generate
    test_samples_to_generate = cfg.general.test_samples_to_generate
    # Topology of THIS run, not of the run that produced the checkpoint. cfg is
    # replaced wholesale below and update_config_with_new_keys only fills in keys
    # that are MISSING, so gpus/num_nodes/strategy from the checkpoint would
    # silently win over what the launcher actually started. See spec2mol_main.py.
    gpus = cfg.general.gpus
    num_nodes = getattr(cfg.general, "num_nodes", 1)
    trainer_strategy = getattr(cfg.train, "trainer_strategy", "auto")
    ddp_timeout_hours = getattr(cfg.train, "ddp_timeout_hours", None)

    # map_location=cpu, never None: None restores each tensor to the device
    # recorded in the checkpoint (cuda:0, whatever rank 0 held when saving), so
    # under DDP every rank on a node materialises the full model on GPU 0 before
    # Lightning moves it to the right device.
    # weights_only=False: save_hyperparameters() embeds the Hydra config, so the
    # pickle contains an omegaconf DictConfig, which PyTorch 2.6's
    # weights_only=True default rejects. Self-produced files, so no exposure.
    model = FP2MolDenoisingDiffusion.load_from_checkpoint(
        resume, map_location=torch.device("cpu"), weights_only=False, **model_kwargs
    )
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.val_samples_to_generate = val_samples_to_generate
    cfg.general.test_samples_to_generate = test_samples_to_generate
    cfg.general.gpus = gpus
    utils.safe_setattr(cfg.general, "num_nodes", num_nodes)
    utils.safe_setattr(cfg.train, "trainer_strategy", trainer_strategy)
    if ddp_timeout_hours is not None:
        utils.force_setattr(cfg.train, "ddp_timeout_hours", ddp_timeout_hours)
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    # map_location / weights_only: see get_resume().
    model = FP2MolDenoisingDiffusion.load_from_checkpoint(
        resume_path,
        map_location=torch.device("cpu"),
        weights_only=False,
        **model_kwargs,
    )
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + "_resume"

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def load_decoder_from_lightning_ckpt(model, ckpt_path):
    """DEPRECATED, USE load_decoder_weights

    Load decoder weights from a PyTorch Lightning checkpoint."""

    # weights_only=False: see get_resume().
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    decoder_state_dict = {}

    for k, v in state_dict.items():

        if k.startswith("decoder."):
            decoder_state_dict[k[len("decoder.") :]] = v

        elif k.startswith("model.decoder."):  # support other Lightning formats
            decoder_state_dict[k[len("model.decoder.") :]] = v

    missing, unexpected = model.decoder.load_state_dict(
        decoder_state_dict, strict=False
    )

    logging.info(f"Loaded decoder from: '{ckpt_path}'")
    logging.info(f"Missing keys: {missing}")
    logging.info(f"Unexpected keys: {unexpected}")


def load_decoder_weights(model, ckpt_path):
    """
    Load decoder weights from either:
    - PyTorch Lightning checkpoint (.ckpt)
    - Standalone decoder weights (.pt)

    Includes robust warnings for common failure cases.
    """

    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logging.info(f"Loading decoder weights from: {ckpt_path}")

    # weights_only=False: see get_resume().
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    decoder_state_dict = {}

    # -------------------------------------------------
    # CASE 1: standalone .pt file (already extracted)
    # -------------------------------------------------
    if ckpt_path.suffix == ".pt" and "state_dict" not in ckpt:

        if not isinstance(ckpt, dict):
            raise TypeError(
                f"{ckpt_path} does not contain a valid state_dict (expected dict, got {type(ckpt)})"
            )

        decoder_state_dict = ckpt

        logging.info("Detected standalone decoder .pt file")

    # -------------------------------------------------
    # CASE 2: Lightning checkpoint
    # -------------------------------------------------
    else:

        if "state_dict" not in ckpt:
            logging.warning(
                "No 'state_dict' key found in checkpoint. Trying to interpret file as raw state_dict."
            )

        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Invalid checkpoint format: expected dict or state_dict, got {type(state_dict)}"
            )

        for k, v in state_dict.items():

            if k.startswith("decoder."):
                decoder_state_dict[k[len("decoder.") :]] = v

            elif k.startswith("model.decoder."):
                decoder_state_dict[k[len("model.decoder.") :]] = v

        if len(decoder_state_dict) == 0:
            logging.warning(
                "No decoder weights were found in the checkpoint. "
                "Check that the checkpoint actually contains a decoder."
            )

    # -------------------------------------------------
    # SANITY CHECKS BEFORE LOADING
    # -------------------------------------------------
    if len(decoder_state_dict) == 0:
        raise RuntimeError("Decoder state_dict is empty — aborting load.")

    # Check if the keys look like a decoder
    if not any("tf_layers" in k for k in decoder_state_dict.keys()):
        logging.warning(
            "Loaded weights do not appear to contain transformer layers (tf_layers). "
            "This may not be a valid decoder checkpoint."
        )

    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------
    missing, unexpected = model.decoder.load_state_dict(
        decoder_state_dict, strict=False
    )

    # -------------------------------------------------
    # POST-LOAD WARNINGS
    # -------------------------------------------------
    if len(missing) > 0:
        logging.warning(f"Missing decoder keys ({len(missing)}): {missing}")

    if len(unexpected) > 0:
        logging.warning(f"Unexpected decoder keys ({len(unexpected)}): {unexpected}")

    if len(missing) == 0 and len(unexpected) == 0:
        logging.info("Decoder weights loaded cleanly (no missing or unexpected keys).")

    logging.info("Decoder loading complete.")


def freeze_weights(model, cfg):
    if cfg.general.finetune_strategy == "freeze_transformer_layers":
        for param in model.decoder.tf_layers.parameters():
            param.requires_grad = False
    else:
        raise NotImplementedError("Unknown finetuning strategy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../configs", config_name="config_decoder")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    # 0. DDP rank detection
    # ------------------------------------------------------------------
    local_rank, global_rank, world_size = _ddp_env()
    is_ddp = world_size > 1

    # On-demand stack dump on every rank. A DDP hang otherwise gives you nothing:
    # ranks sit at 100% GPU while RCCL spins, and the watchdog only fires if the
    # hang is inside an enqueued collective. See spec2mol_main.py for the
    # srun/pkill invocation. Keep the file object alive for the process lifetime.
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        _fault_log = open(f"faulthandler_rank{global_rank:02d}.log", "w")
        faulthandler.register(
            signal.SIGUSR1, file=_fault_log, all_threads=True, chain=False
        )

    # Point each rank at its GPU before any CUDA call
    if is_ddp and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # ------------------------------------------------------------------
    # 1. Logging (rank 0 only — otherwise every line appears world_size times)
    # ------------------------------------------------------------------
    # The root level is not sufficient on its own: a logger's own level decides
    # whether a record is created, and pytorch_lightning/__init__.py sets its own
    # to INFO. logging.disable is a global floor applied before any per-logger
    # check, so it catches those too. WARNING and above still get through.
    logging.getLogger().setLevel(logging.INFO if global_rank == 0 else logging.WARNING)
    if global_rank != 0:
        logging.disable(logging.INFO)

    logger = logging.getLogger("msms_main")
    logger.setLevel(logging.INFO if global_rank == 0 else logging.WARNING)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s [rank%(process)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if global_rank == 0:
        fh = logging.FileHandler(os.path.join("msms_main.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logging.info(
        f"DDP: world_size={world_size}, global_rank={global_rank}, local_rank={local_rank}"
    )
    if global_rank == 0:
        logging.info(f"Output directory: {os.getcwd()}")
        logging.info(cfg)

    model_dtype = dtype_map[getattr(cfg.model, "model_dtype", "float32")]
    torch.set_default_dtype(model_dtype)
    logging.info(f"Model dtype set to {model_dtype}")

    # ------------------------------------------------------------------
    # 2. Dataset (rank 0 first — see _ensure_dataset_ready)
    # ------------------------------------------------------------------
    datamodule, dataset_infos = _ensure_dataset_ready(cfg, global_rank, is_ddp)
    logging.info("Dataset loaded")
    if global_rank == 0:
        logging.info(
            f"Train Size: {len(datamodule.train_dataloader())}, Val Size: {len(datamodule.val_dataloader())}, Test Size: {len(datamodule.test_dataloader())}"
        )

    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(
            cfg.model.extra_features, dataset_info=dataset_infos
        )
    else:
        extra_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    logging.info(f"Dataset infos: {dataset_infos.output_dims}")
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    visualization_tools = MolecularVisualization(
        cfg.dataset.remove_h, dataset_infos=dataset_infos
    )

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "dtype": model_dtype,
    }

    # ------------------------------------------------------------------
    # 3. LR scaling: sqrt rule (1 baseline GPU → N total GPUs)
    #
    # Linear scaling (Goyal et al.) is for SGD+momentum; for the adaptive
    # optimizers this repo uses it overshoots badly at large N. See spec2mol_main.
    # ------------------------------------------------------------------
    num_nodes = getattr(cfg.general, "num_nodes", 1)
    total_gpus = cfg.general.gpus * num_nodes
    if total_gpus > 1:
        lr_scale = math.sqrt(total_gpus)
        cfg.train.lr = cfg.train.lr * lr_scale
        logging.info(
            f"Scaled LR by sqrt({total_gpus}) = {lr_scale:.2f} → {cfg.train.lr:.6f}"
        )

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split("checkpoints")[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split("checkpoints")[0])

    utils.make_result_dirs(["preds/", "models/", "logs/", f"logs/{cfg.general.name}"])

    model = FP2MolDenoisingDiffusion(cfg=cfg, **model_kwargs)

    try:
        if cfg.general.pretrained is not None:
            logging.info(f"Trying to load model from: '{cfg.general.pretrained}'")
            if cfg.general.pretrained.endswith(".ckpt"):
                load_decoder_from_lightning_ckpt(model, cfg.general.pretrained)
            elif cfg.general.pretrained.endswith(".pt"):
                load_decoder_weights(model, cfg.general.pretrained)
            else:
                raise NotImplementedError(
                    "Only PyTorch Lightning checkpoints currently supported!"
                )
    except Exception as e:
        print("Could not load pretrained model:", e)

    try:
        if cfg.general.finetune_strategy is not None:
            freeze_weights(model, cfg)
    except Exception as e:
        print("Could not freeze weights:", e)

    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",  # best (top-5) checkpoints
            filename="{epoch}",
            monitor="val/NLL",
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
        )
        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}", filename="last", every_n_epochs=1
        )  # most recent checkpoint
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:  # TODO: Implement EMA for FP2Mol
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == "debug":
        logging.warning("Run is called 'debug' -- it will run with fast_dev_run. ")

    loggers = [
        CSVLogger(save_dir=f"logs/{name}", name=name),
    ]

    # ------------------------------------------------------------------
    # 4. DDP strategy
    #
    # Built as a DDPStrategy object rather than the registry string because only
    # the object form accepts `timeout`. That timeout is the NCCL watchdog
    # deadline for every collective and defaults to 30 minutes -- far too short
    # for the test epoch, where each rank runs the same NUMBER of sampling
    # batches but generation cost scales with molecule size, so a rank drawing
    # small molecules reaches the epoch-end barrier long before the slowest one.
    # See the barrier in FP2MolDenoisingDiffusion.on_test_epoch_end.
    # ------------------------------------------------------------------
    configured_strategy = getattr(cfg.train, "trainer_strategy", "auto")
    if configured_strategy != "auto":
        strategy_name = configured_strategy
    else:
        strategy_name = "ddp_find_unused_parameters_true" if is_ddp else "auto"

    if isinstance(strategy_name, str) and strategy_name.startswith("ddp"):
        ddp_timeout_hours = float(getattr(cfg.train, "ddp_timeout_hours", 8))
        trainer_strategy = DDPStrategy(
            find_unused_parameters="find_unused_parameters_true" in strategy_name,
            timeout=datetime.timedelta(hours=ddp_timeout_hours),
        )
        strategy_desc = f"{strategy_name} (collective timeout {ddp_timeout_hours}h)"
    else:
        trainer_strategy = strategy_name
        strategy_desc = str(strategy_name)

    if global_rank == 0:
        logging.info(f"Trainer strategy: {strategy_desc}")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    limit_val_batches = getattr(cfg.train, "limit_val_batches", None)
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=trainer_strategy,
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        num_nodes=num_nodes,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=name == "debug",
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        limit_val_batches=limit_val_batches,
        logger=loggers,
        enable_progress_bar=getattr(cfg.train, "progress_bar", False),
        precision=getattr(cfg.train, "precision", "32-true"),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.set_float32_matmul_precision("medium")
        except:
            logging.info("Could not enable float32 matmul precision - medium")
    logging.info(f"Current path: {Path.cwd()}")
    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ["debug", "test"] and not getattr(
            cfg.general, "skip_test", False
        ):
            trainer.test(model, datamodule=datamodule)
        else:
            logging.info("Skipped test epoch")
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            logging.info(f"Evaluating all checkpoints in: {directory}")
            # sorted(): os.listdir order is filesystem-dependent and not guaranteed
            # identical across ranks. Ranks iterating in different orders would test
            # different weights in the same trainer.test() call and desync every
            # collective in it.
            for file in sorted(os.listdir(directory)):
                if ".ckpt" in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    logging.info(f"Loading checkpoint: {ckpt_path}")
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
