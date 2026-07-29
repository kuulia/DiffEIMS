"""Print the DDP topology (and a few other run-shaped keys) stored in a checkpoint.

Why this matters: get_resume() in spec2mol_main.py replaces the live cfg with the
one embedded in the checkpoint, and update_config_with_new_keys() only fills in
keys that are MISSING. Any key the checkpoint already carries therefore wins over
the command line, including general.gpus / general.num_nodes — the two values the
SLURM script passes explicitly and that Trainer uses to size the process group.
If they disagree with what srun actually launched, DDP construction desyncs.

Usage (inside the container, venv activated):
    python src/slurm/check_ckpt_topology.py /path/to/fine_tuned.ckpt
"""

import sys

import torch


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    path = sys.argv[1]
    # weights_only=False: these checkpoints embed the Hydra config (an omegaconf
    # DictConfig), which PyTorch 2.6's weights_only=True default rejects. Same
    # reasoning as the torch.load calls in spec2mol_main.py.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    hparams = ckpt.get("hyper_parameters", {})
    cfg = hparams.get("cfg")
    if cfg is None:
        print(f"No cfg found under 'hyper_parameters'. Top-level keys: {list(ckpt)}")
        print(f"hyper_parameters keys: {list(hparams)}")
        return 1

    print(f"checkpoint: {path}")
    print(f"  general.gpus       = {cfg.general.get('gpus')}")
    print(f"  general.num_nodes  = {cfg.general.get('num_nodes')}")
    print(f"  train.trainer_strategy  = {cfg.train.get('trainer_strategy')}")
    print(f"  train.ddp_timeout_hours = {cfg.train.get('ddp_timeout_hours')}")
    print(f"  train.eval_batch_size   = {cfg.train.get('eval_batch_size')}")
    print()
    print("Expected for a 4-node x 8-GCD run: gpus=8, num_nodes=4.")
    print("A mismatch means Trainer sized the process group from the checkpoint,")
    print("not from srun — which desyncs DDP's initial parameter broadcast.")

    n_params = sum(
        v.numel() for v in ckpt.get("state_dict", {}).values() if hasattr(v, "numel")
    )
    print(f"\nstate_dict total elements = {n_params:,}")
    print("Compare against NumelIn in the hanging BROADCAST (72,579,445).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
