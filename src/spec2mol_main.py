"""
spec2mol_main.py — DDP-compatible training entry point.

DDP flow
--------
All ranks enter main() simultaneously (launched by torchrun / srun+torchrun).
Before the model is constructed we need dataset metadata (stat files + input
dims).  We handle this with a lightweight file-based sentinel mechanism so
that no torch.distributed call is needed before trainer.fit():

  1. Rank 0: check whether the sentinel file (.stat_files_ready) exists.
             If not, load the full dataset and compute stat files (one-time
             setup), then touch the sentinel.
     Other ranks: poll for the sentinel file with a 5-second interval.

  2. All ranks: instantiate Spec2MolDatasetInfos from the now-guaranteed
     stat files (pure file reads, no data loading, no distributed call).

  3. All ranks: create the model with the known input/output dims.

  4. All ranks: trainer.fit(model, datamodule=datamodule)
     Lightning calls prepare_data() on rank 0 (no-op: sentinel already
     exists) then calls setup() on all ranks with LOCAL_RANK-based I/O
     staggering to load dataset pickles without filesystem contention.
     Lightning's DDPStrategy injects DistributedSampler automatically.
"""

import os
import sys
import math
import time
import pathlib
import warnings
import logging

import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from rdkit import RDLogger

from src import utils
from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.analysis.visualization import MolecularVisualization
from src.datasets import spec2mol_dataset
from src.datasets.spec2mol_dataset import (
    Spec2MolDataModule,
    Spec2MolDatasetInfos,
    compute_dataset_stats,
    _sentinel_path,
    _SENTINEL_POLL_INTERVAL,
    _SENTINEL_TIMEOUT,
)

warnings.filterwarnings("ignore", category=PossibleUserWarning)
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def _ddp_env():
    """Return (local_rank, global_rank, world_size) from torchrun env vars."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, global_rank, world_size


def _ensure_stats_ready(cfg, global_rank: int, is_ddp: bool):
    """
    Guarantee that dataset stat files exist before any rank proceeds.

    Rank 0: compute stats if the sentinel is absent, then touch the sentinel.
    Others: poll for the sentinel file (no torch.distributed needed).
    """
    sentinel = _sentinel_path(cfg)

    if global_rank == 0:
        if not os.path.exists(sentinel):
            logging.info(
                "Rank 0: stat files not found — computing (one-time setup). "
                "This may take several minutes."
            )
            compute_dataset_stats(cfg)
            pathlib.Path(sentinel).touch()
            logging.info(f"Rank 0: sentinel written → {sentinel}")
        else:
            logging.info("Rank 0: stat files present, skipping computation.")
    elif is_ddp:
        if not os.path.exists(sentinel):
            logging.info(
                f"Rank {global_rank}: waiting for rank 0 to finish stat computation..."
            )
            start = time.time()
            while not os.path.exists(sentinel):
                if time.time() - start > _SENTINEL_TIMEOUT:
                    raise TimeoutError(
                        f"Rank {global_rank} timed out ({_SENTINEL_TIMEOUT}s) "
                        f"waiting for sentinel '{sentinel}'."
                    )
                time.sleep(_SENTINEL_POLL_INTERVAL)
            logging.info(f"Rank {global_rank}: sentinel found, proceeding.")


# ---------------------------------------------------------------------------
# Resume / checkpoint helpers (unchanged from original)
# ---------------------------------------------------------------------------


def safe_setattr(cfg_section, key, value):
    if isinstance(cfg_section, DictConfig):
        if key in cfg_section:
            cfg_section[key] = value
    else:
        if hasattr(cfg_section, key):
            setattr(cfg_section, key, value)


def get_resume(cfg, model_kwargs):
    """Resume from a saved Lightning checkpoint (test-only mode)."""
    saved_cfg = cfg.copy()

    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only
    val_samples_to_generate = cfg.general.val_samples_to_generate
    test_samples_to_generate = cfg.general.test_samples_to_generate
    num_test_samples = cfg.general.num_test_samples
    eval_batch_size = cfg.train.eval_batch_size
    decoder = getattr(cfg.general, "decoder", None)
    encoder = getattr(cfg.general, "encoder", None)
    inference_only = getattr(cfg.dataset, "inference_only", None)
    override_prev_dataset_cfg = getattr(cfg.dataset, "override_prev_dataset_cfg", False)
    dataset_cfg = cfg.dataset

    map_loc = torch.device("cpu") if cfg.general.force_cpu else None
    cfg = Spec2MolDenoisingDiffusion.load_from_checkpoint(
        resume, map_location=map_loc, **model_kwargs
    ).cfg
    logging.info(f"Loaded cfg from {resume}")

    cfg.general.name = name
    cfg.general.test_only = resume
    cfg.general.val_samples_to_generate = val_samples_to_generate
    cfg.general.test_samples_to_generate = test_samples_to_generate
    cfg.general.num_test_samples = num_test_samples
    cfg.train.eval_batch_size = eval_batch_size
    safe_setattr(cfg.general, "encoder", encoder)
    safe_setattr(cfg.general, "decoder", decoder)
    safe_setattr(cfg.dataset, "inference_only", inference_only)
    if override_prev_dataset_cfg:
        cfg.dataset = dataset_cfg
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)

    model = Spec2MolDenoisingDiffusion.load_from_checkpoint(
        resume, map_location=map_loc, cfg=cfg, **model_kwargs
    )
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """Resume training with optional config overrides."""
    saved_cfg = cfg.copy()
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)

    map_loc = torch.device("cpu") if cfg.general.force_cpu else None
    model = Spec2MolDenoisingDiffusion.load_from_checkpoint(
        resume_path, map_location=map_loc, **model_kwargs
    )
    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]
    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + "_resume"
    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def apply_encoder_finetuning(model, strategy):
    if strategy is None:
        pass
    elif strategy == "freeze":
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif strategy == "ft-unfold":
        for param in model.encoder.named_parameters():
            if param[0].split(".")[1] != "2":
                param[1].requires_grad = False
    elif strategy == "freeze-unfold":
        for param in model.encoder.named_parameters():
            if param[0].split(".")[1] == "2":
                param[1].requires_grad = False
    elif strategy == "ft-transformer":
        for param in model.encoder.named_parameters():
            if param[0].split(".")[1] != "0":
                param[1].requires_grad = False
    elif strategy == "freeze-transformer":
        for param in model.encoder.named_parameters():
            if param[0].split(".")[1] == "0":
                param[1].requires_grad = False
    elif strategy == "ft-spectra-fragment":
        for name, param in model.encoder.named_parameters():
            if name.split(".")[1] not in ["0", "1"]:
                param.requires_grad = False
    else:
        raise NotImplementedError(f"Unknown Finetune Strategy: {strategy}")


def apply_decoder_finetuning(model, strategy):
    if strategy is None:
        pass
    elif strategy == "freeze":
        for param in model.decoder.parameters():
            param.requires_grad = False
    elif strategy == "ft-input":
        for p in model.decoder.named_parameters():
            if p[0].split(".")[0] not in ["mlp_in_X", "mlp_in_E", "mlp_in_y"]:
                p[1].requires_grad = False
    elif strategy == "freeze-input":
        for p in model.decoder.named_parameters():
            if p[0].split(".")[0] in ["mlp_in_X", "mlp_in_E", "mlp_in_y"]:
                p[1].requires_grad = False
    elif strategy == "ft-transformer":
        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.decoder.tf_layers.parameters():
            param.requires_grad = True
    elif strategy == "freeze-transformer":
        for param in model.decoder.tf_layers.parameters():
            param.requires_grad = False
    elif strategy == "ft-output":
        for p in model.decoder.named_parameters():
            if p[0].split(".")[0] not in ["mlp_out_X", "mlp_out_E", "mlp_out_y"]:
                p[1].requires_grad = False
    else:
        raise NotImplementedError(f"Unknown Finetune Strategy: {strategy}")


def load_weights(model, path):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state_dict}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logging.info(f"Loaded weights from {path}")
    logging.info(f"Missing keys: {missing}")
    logging.info(f"Unexpected keys: {unexpected}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    # 0. DDP rank detection
    # ------------------------------------------------------------------
    local_rank, global_rank, world_size = _ddp_env()
    is_ddp = world_size > 1

    def _ckpt(label: str):
        """Print a timestamped checkpoint visible from all ranks."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}][rank{global_rank:02d}] {label}", flush=True)

    _ckpt("started main()")

    # Point each DDP rank to its GPU before any CUDA call
    if is_ddp and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    _ckpt("cuda device set")

    # ------------------------------------------------------------------
    # 1. Logging (only rank 0 prints to avoid interleaved output)
    # ------------------------------------------------------------------
    name: str = cfg.general.name
    utils.make_result_dirs(["preds/", "logs/", "models/", f"logs/{name}"])

    # Set root logger level so ALL modules (spec2mol_dataset, Lightning, etc.)
    # are silenced on non-zero ranks — not just the named logger below.
    logging.getLogger().setLevel(logging.INFO if global_rank == 0 else logging.WARNING)

    logger = logging.getLogger("msms_main")
    logger.setLevel(logging.INFO if global_rank == 0 else logging.WARNING)
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s [rank%(process)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if global_rank == 0:
        fh = logging.FileHandler(os.path.join("msms_main.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logging.info(
        f"DDP: world_size={world_size}, global_rank={global_rank}, local_rank={local_rank}"
    )
    if global_rank == 0:
        logging.info(f"Output directory: {os.getcwd()}")
        logging.info(cfg)

    # ------------------------------------------------------------------
    # 2. Dataset name validation
    # ------------------------------------------------------------------
    dataset_name = cfg["dataset"]["name"]
    valid_datasets = (
        "canopus",
        "msg",
        "neims",
        "franklin",
        "neims_tms",
        "gecko_atmomaccs",
        "msg_neims",
        "mixed_augment_test",
        "mixed_augment",
        "mixed_atmomaccs",
        "gecko_new_atmomaccs",
        "gecko_new",
        "gecko_new_mixed_augment_atmomaccs_test",
    )
    if dataset_name not in valid_datasets:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")

    # ------------------------------------------------------------------
    # 3. Ensure stat files exist (sentinel-file coordination, no dist init)
    # ------------------------------------------------------------------
    _ckpt("before _ensure_stats_ready")
    _ensure_stats_ready(cfg, global_rank, is_ddp)
    _ckpt("after _ensure_stats_ready")

    # ------------------------------------------------------------------
    # 4. Build DataModule (lightweight — no I/O in __init__)
    # ------------------------------------------------------------------
    datamodule = Spec2MolDataModule(cfg)
    _ckpt("DataModule created")

    # ------------------------------------------------------------------
    # 5. Build DatasetInfos from pre-existing stat files
    #    (pure file reads + static dim computation, no DataLoader iteration)
    # ------------------------------------------------------------------
    dataset_infos = Spec2MolDatasetInfos(cfg)
    _ckpt("DatasetInfos created")

    # ------------------------------------------------------------------
    # 6. Extra features (require dataset_infos.max_n_nodes from stat files)
    # ------------------------------------------------------------------
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(
            cfg.model.extra_features, dataset_info=dataset_infos
        )
    else:
        extra_features = DummyExtraFeatures()

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
    }

    # ------------------------------------------------------------------
    # 7. LR scaling: sqrt scaling rule (1 baseline GPU → N total GPUs)
    #
    # The linear rule (lr * N) comes from Goyal et al. for SGD+momentum. For
    # adaptive optimizers (this repo uses adamw/radam) sqrt scaling is the usual
    # choice -- linear overshoots badly at large N. At 16 GPUs linear gave
    # 0.0002 * 16 = 0.0032, over 2x the documented from-scratch LR of 0.0015;
    # sqrt gives 0.0002 * 4 = 0.0008.
    # ------------------------------------------------------------------
    num_nodes = getattr(cfg.general, "num_nodes", 1)
    total_gpus = cfg.general.gpus * num_nodes
    if total_gpus > 1:
        lr_scale = math.sqrt(total_gpus)
        cfg.train.lr = cfg.train.lr * lr_scale
        logging.info(
            f"Scaled LR by sqrt({total_gpus}) = {lr_scale:.2f} → {cfg.train.lr:.6f}"
        )

    # ------------------------------------------------------------------
    # 8. Model construction
    # ------------------------------------------------------------------
    resume: str | None = cfg.general.resume

    _ckpt("before model construction")
    if cfg.general.test_only:
        cfg, model = get_resume(cfg, model_kwargs)
        logging.info("Loaded model for test_only via get_resume()")
    elif resume is not None:
        cfg, model = get_resume_adaptive(cfg, model_kwargs)
        logging.info("Loaded model for adaptive resume via get_resume_adaptive()")
    else:
        model = Spec2MolDenoisingDiffusion(cfg=cfg, **model_kwargs)
    _ckpt("model constructed")

    utils.log_nonstatic_cfg(cfg)

    # ------------------------------------------------------------------
    # 9. Finetuning / weight loading
    # ------------------------------------------------------------------
    apply_encoder_finetuning(model, cfg.general.encoder_finetune_strategy)
    apply_decoder_finetuning(model, cfg.general.decoder_finetune_strategy)

    if cfg.general.load_weights is not None:
        logging.info(f"Loading weights from {cfg.general.load_weights}")
        model = load_weights(model, cfg.general.load_weights)

    # ------------------------------------------------------------------
    # 10. Optional torch.compile (decoder only)
    # ------------------------------------------------------------------
    if (
        torch.cuda.is_available()
        and not cfg.general.test_only
        and getattr(cfg.train, "compile", False)
    ):
        logging.info("Compiling decoder with torch.compile (dynamic=True)")
        model.decoder = torch.compile(model.decoder, dynamic=True)

    # ------------------------------------------------------------------
    # 11. Callbacks
    # ------------------------------------------------------------------
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if cfg.train.save_model:
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"checkpoints/{name}",
                filename="{epoch}",
                monitor="val/NLL",
                save_top_k=1,
                mode="min",
                every_n_epochs=1,
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"checkpoints/{name}",
                filename="last",
                every_n_epochs=1,
            )
        )

    # ------------------------------------------------------------------
    # 12. Loggers
    # ------------------------------------------------------------------
    loggers = [CSVLogger(save_dir=f"logs/{name}", name=name)]

    # ------------------------------------------------------------------
    # 13. DDP strategy selection
    #
    #  - In single-GPU / CPU mode: "auto" lets Lightning decide
    #  - In DDP mode: use ddp_find_unused_parameters_true so that old
    #    checkpoints (which may have unused parameters) load correctly
    #  - Can be overridden via cfg.train.trainer_strategy
    # ------------------------------------------------------------------
    if hasattr(cfg.train, "trainer_strategy") and cfg.train.trainer_strategy != "auto":
        trainer_strategy = cfg.train.trainer_strategy
    else:
        trainer_strategy = "ddp_find_unused_parameters_true" if is_ddp else "auto"

    if global_rank == 0:
        logging.info(f"Trainer strategy: {trainer_strategy}")

    use_gpu = cfg.general.gpus > 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            logging.info("Could not set float32 matmul precision")

    # ------------------------------------------------------------------
    # 14. Profiler (rank 0 only — avoids 16x trace files in DDP)
    # ------------------------------------------------------------------
    profiler = None
    profiler_type = getattr(cfg.train, "profiler", None)
    if profiler_type and global_rank == 0:
        if profiler_type == "simple":
            from pytorch_lightning.profilers import SimpleProfiler
            profiler = SimpleProfiler(dirpath=f"logs/{name}", filename="profiler")
        elif profiler_type == "pytorch":
            from pytorch_lightning.profilers import PyTorchProfiler
            profiler = PyTorchProfiler(
                dirpath=f"logs/{name}/profiler",
                filename="trace",
                emit_nvtx=False,
                export_to_chrome=True,
            )

    # ------------------------------------------------------------------
    # 15. Trainer
    # ------------------------------------------------------------------
    if name == "debug":
        logging.warning("Run is named 'debug' — fast_dev_run will be used.")
    limit_val_batches = getattr(
        cfg.train, "limit_val_batches", None
    )
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=trainer_strategy,
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        num_nodes=getattr(cfg.general, "num_nodes", 1),
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=(name == "debug"),
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        limit_val_batches=limit_val_batches,
        logger=loggers,
        enable_progress_bar=getattr(cfg.train, "progress_bar", False),
        profiler=profiler,
    )

    # ------------------------------------------------------------------
    # 15. Run
    #
    # trainer.fit() calls (on all ranks, coordinated by Lightning):
    #   - datamodule.prepare_data()  [rank 0 only — no-op, sentinel exists]
    #   - datamodule.setup('fit')    [all ranks — staggered I/O]
    #   - training loop
    # ------------------------------------------------------------------
    _ckpt("calling trainer.fit()")
    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume)

        if name not in ["debug", "test"] and not getattr(
            cfg.general, "skip_test", False
        ):
            trainer.test(
                model,
                datamodule=datamodule,
                ckpt_path=cfg.general.checkpoint_strategy,
            )
        else:
            logging.info("Skipped test epoch.")
    else:
        trainer.test(model, datamodule=datamodule)

        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            logging.info(f"Evaluating all checkpoints in: {directory}")
            for file in os.listdir(directory):
                if ".ckpt" in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    logging.info(f"Loading checkpoint: {ckpt_path}")
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
