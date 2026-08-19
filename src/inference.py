"""
inference.py

Inference script for generating molecules from mass spectra where no molecular
structure (SMILES/InChI/fingerprints) is known — only spectra and formulas.

Based on src/eval_test.py but designed for:
- Inference-only data (e.g., Yee et al. dataset)
- No ground-truth comparison (so no NLL/KL/top-k; those need true structures)

BATCHING. Spectra are processed train.eval_batch_size at a time, not one by one.
This does not affect the results: to_dense pads each batch to its largest molecule,
but the padding is masked throughout, and a given molecule's extra-features come
out bit-identical at padding widths 25/40/60/100 (verified directly). The one
consequence of batch size is the RNG stream — sample_discrete_feature_noise draws
a multinomial sized by the padded batch, so a fixed seed yields different noise,
and therefore different candidate molecules, at different eval_batch_size. That is
resampling from the same distribution, not a bias, but it does mean runs are only
reproducible when eval_batch_size is held fixed.

Usage:
    python -m src.inference
    python -m src.inference general.test_samples_to_generate=10

    Config lives in configs/config_inference.yaml and configs/dataset/yee.yaml;
    command-line overrides are forwarded to hydra.compose (see main()).
"""

import sys

sys.path.append("../src")

import numpy as np
import pickle
import hydra
import torch
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem, RDLogger

from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.datasets.inference_dataset import InferenceDataModule, InferenceDatasetInfos
from src import utils
from omegaconf import OmegaConf

from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def move_batch_to_device(batch, device):
    """Recursively move all tensors in a batch dict to the given device."""
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            for subkey in batch[key]:
                if isinstance(batch[key][subkey], torch.Tensor):
                    batch[key][subkey] = batch[key][subkey].to(device)
    return batch


def predict_fingerprint(model, batch):
    """Run the encoder on a batch and predict fingerprints."""
    output, aux = model.encoder(batch)

    if model.merge == "mist_fp":
        y = aux["int_preds"][-1]
    elif model.merge in ("merge-encoder_output-linear", "merge-encoder_output-mlp"):
        y = model.merge_function(aux["h0"])
    elif model.merge == "downproject_4096":
        y = model.merge_function(output)
    else:
        raise ValueError(f"Unknown merge strategy: {model.merge}")

    return y


def generate_molecules(model, data, num_samples):
    """Generate molecules from graph data using the diffusion model.

    Args:
        model: Spec2MolDenoisingDiffusion model.
        data: PyTorch Geometric Batch with .y set to predicted fingerprint.
        num_samples: Number of molecule samples to generate per spectrum.

    Returns:
        List of lists of RDKit Mol objects (one list per spectrum in batch).
    """
    generated_mols = [list() for _ in range(len(data))]
    for _ in range(num_samples):
        mols = model.sample_batch(data)
        for idx, mol in enumerate(mols):
            generated_mols[idx].append(mol)
    return generated_mols


def main():
    # Load config
    hydra.initialize(version_base="1.3", config_path="../configs")
    # compose() does not read sys.argv the way @hydra.main does, so without this
    # any override typed on the command line is accepted and silently ignored.
    overrides = [a for a in sys.argv[1:] if "=" in a]
    if overrides:
        logger.info(f"Config overrides: {overrides}")
    cfg = hydra.compose(config_name="config_inference", overrides=overrides)

    logger.info(f"Dataset config: {cfg.dataset}")

    # Setup output directory
    now = datetime.now()
    date_part = now.strftime("%Y-%m-%d")
    time_part = now.strftime("%H-%M-%S")
    # "outputs", not "../outputs": main() uses the Compose API (hydra.initialize +
    # hydra.compose) rather than @hydra.main, so hydra.job.chdir never fires and the
    # cwd stays wherever the script was launched from. "../outputs" resolved to the
    # parent of the repo. Dataset paths in the config are repo-relative for the same
    # reason, so this must be run from the repo root either way.
    output_path = Path("outputs") / date_part / f"{time_part}-inference"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Initialize data module and dataset infos
    logger.info("Loading inference dataset...")
    datamodule = InferenceDataModule(cfg)
    dataset_infos = InferenceDatasetInfos(datamodule, cfg)

    # Build model infrastructure
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

    # Load model from checkpoint
    checkpoint_path = cfg.dataset.eval_model_path
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    # weights_only=False: save_hyperparameters() embeds the Hydra config, so the
    # checkpoint pickle contains an omegaconf DictConfig, which PyTorch 2.6's
    # weights_only=True default rejects with "Unsupported global: DictConfig".
    # These checkpoints are produced by this repo, so there is no untrusted-pickle
    # exposure. Same reason as the load_from_checkpoint calls in spec2mol_main.py.
    model = Spec2MolDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
        load_pretrained_weights=False,
        **model_kwargs,
    )
    model.eval()

    # Load to CPU, then move to the accelerator. `device = model.device` alone read
    # back cpu (map_location above put every parameter there) and the whole run
    # executed on CPU — ~500 denoising steps per candidate through a 53 M-parameter
    # graph transformer, which is days rather than minutes.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type == "cpu":
        logger.warning(
            "No GPU visible — running on CPU. Generation will take days, not minutes."
        )

    # model.test_num_samples comes from the config embedded in the checkpoint, not
    # from the config being composed now, so it silently differs per checkpoint (10
    # for tms_final_april, 100 for the gecko runs). Let the live config win — this is
    # the only lever on runtime, which scales as
    #     ceil(n_spectra / eval_batch_size) * num_samples * T
    # forward passes on one GPU.
    num_samples = model.test_num_samples
    cfg_samples = getattr(cfg.general, "test_samples_to_generate", None)
    if cfg_samples is not None and cfg_samples != num_samples:
        logger.info(
            f"Overriding checkpoint's test_num_samples ({num_samples}) with "
            f"cfg.general.test_samples_to_generate={cfg_samples}"
        )
        num_samples = cfg_samples

    n_batches = -(-len(datamodule.test_dataset) // cfg.train.eval_batch_size)
    logger.info(f"Model loaded. Device: {device}. Samples per spectrum: {num_samples}")
    logger.info(
        f"Work: {n_batches} batches x {num_samples} candidates x {model.T} steps "
        f"= {n_batches * num_samples * model.T:,} decoder passes"
    )

    # Point-wise inference
    test_loader = datamodule.test_dataloader()
    all_results = []
    all_fingerprints = []
    all_generated_mols = []
    all_names = []

    logger.info(
        f"Starting point-wise inference on {len(datamodule.test_dataset)} spectra..."
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            batch = move_batch_to_device(batch, device)

            # 1. Encode spectrum → predicted fingerprint
            y = predict_fingerprint(model, batch)
            y = y.to(device)

            # 2. Assign predicted fingerprint to graph data
            data = batch["graph"]
            data.y = y
            data = data.to(device)

            # 3. Generate molecules
            generated_mols = generate_molecules(model, data, num_samples)

            # 4. Collect results
            names = batch.get(
                "names", [f"sample_{batch_idx}_{i}" for i in range(len(data))]
            )
            for i in range(len(data)):
                uid = names[i] if i < len(names) else f"sample_{batch_idx}_{i}"
                formula = (
                    data.get_example(i).inchi.replace("InChI=1S/", "")
                    if hasattr(data.get_example(i), "inchi")
                    else "unknown"
                )

                valid_count = sum(1 for mol in generated_mols[i] if mol is not None)
                smiles_list = []
                for mol in generated_mols[i]:
                    if mol is not None:
                        try:
                            smi = Chem.MolToSmiles(mol)
                            smiles_list.append(smi)
                        except Exception:
                            smiles_list.append(None)
                    else:
                        smiles_list.append(None)

                all_results.append(
                    {
                        "uid": uid,
                        "formula": formula,
                        "num_valid": valid_count,
                        "num_samples": num_samples,
                        "smiles": smiles_list,
                    }
                )
                all_names.append(uid)

            all_fingerprints.append(y.detach().cpu())
            all_generated_mols.extend(generated_mols)

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")

    # Save results
    logger.info(f"Saving results to {output_path}")

    # Save generated molecules
    with open(f"{output_path}/generated_mols.pkl", "wb") as f:
        pickle.dump(all_generated_mols, f)

    # Save predicted fingerprints
    all_fingerprints_tensor = torch.cat(all_fingerprints, dim=0)
    torch.save(all_fingerprints_tensor, f"{output_path}/predicted_fingerprints.pt")

    # Save names
    with open(f"{output_path}/batch_names.txt", "w") as file:
        for name in all_names:
            file.write(name + "\n")

    # Save fingerprints as text (thresholded at 0.5)
    with open(f"{output_path}/fp.txt", "w") as file:
        for tens in all_fingerprints_tensor:
            str_out = " ".join(
                str(int(el)) for el in (tens.numpy() >= 0.5).astype(np.uint8)
            )
            file.write(str_out + "\n")

    # Save results summary as CSV
    import pandas as pd

    summary_rows = []
    for result in all_results:
        valid_smiles = [s for s in result["smiles"] if s is not None]
        summary_rows.append(
            {
                "uid": result["uid"],
                "formula": result["formula"],
                "num_valid": result["num_valid"],
                "num_samples": result["num_samples"],
                "validity_rate": (
                    result["num_valid"] / result["num_samples"]
                    if result["num_samples"] > 0
                    else 0
                ),
                "unique_smiles": len(set(valid_smiles)),
                "top_smiles": valid_smiles[0] if valid_smiles else None,
            }
        )
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{output_path}/results_summary.csv", index=False)

    # Save full SMILES results
    with open(f"{output_path}/all_smiles.pkl", "wb") as f:
        pickle.dump({r["uid"]: r["smiles"] for r in all_results}, f)

    logger.info(f"Inference complete. {len(all_results)} spectra processed.")
    logger.info(f"Results saved to: {output_path}")

    # Print summary statistics
    total_valid = sum(r["num_valid"] for r in all_results)
    total_samples = sum(r["num_samples"] for r in all_results)
    logger.info(
        f"Overall validity: {total_valid}/{total_samples} ({100*total_valid/total_samples:.1f}%)"
    )


if __name__ == "__main__":
    main()
