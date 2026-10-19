#!/bin/bash
# Build a local, CPU-only conda env that matches the LUMI container + pyg_venv the
# data pipeline and training run in, from a snapshot made by envs/dump_lumi_env.sh.
#
#   bash envs/build_local_env.sh                       # envs/lumi_env.txt -> diffms_lumi
#   bash envs/build_local_env.sh path/to/lumi_env.txt
#   DRY_RUN=1 bash envs/build_local_env.sh             # print the plan, install nothing
#
# What "matches" means here: the same Python minor version, and the LUMI versions of
# every package DiffMS imports (numpy/pandas above all, which decide whether the
# .pkl files unpickle). The rest of the container (vLLM, DeepSpeed, ROCm tooling, …)
# is irrelevant to DiffMS and mostly uninstallable off LUMI, so it is not copied;
# pip resolves the remaining transitive dependencies itself.
#
# torch is the one deliberate difference: LUMI ships a ROCm build, this installs the
# CPU wheel of the same X.Y.Z from download.pytorch.org.
#
# Overridable:
#   ENV_NAME  conda env to create             [diffms_lumi]
#   DRY_RUN   1 = show the plan only          [0]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SNAPSHOT="${1:-$REPO/envs/lumi_env.txt}"
ENV_NAME="${ENV_NAME:-diffms_lumi}"
DRY_RUN="${DRY_RUN:-0}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

[[ -r "$SNAPSHOT" ]] || {
    echo "ERROR: snapshot not found: $SNAPSHOT" >&2
    echo "       Create it on LUMI with envs/dump_lumi_env.sh and copy it here." >&2
    exit 1
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Packages DiffMS needs directly (pyproject.toml dependencies + what src/ imports).
# Distribution names, not import names.
NEEDED="numpy pandas scipy rdkit torch_geometric pytorch_lightning lightning_utilities
torchmetrics hydra_core omegaconf h5py scikit_learn joblib tqdm tqdm_joblib wandb
matplotlib seaborn overrides myopic_mces requests setuptools"

# Parse the snapshot with whatever python3 is on PATH (only stdlib needed).
python3 - "$SNAPSHOT" "$WORK" $NEEDED <<'PY'
import re, sys
snapshot, work, *needed = sys.argv[1:]
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()

pyver, pins = None, {}
for line in open(snapshot):
    line = line.strip()
    if line.startswith("# python "):
        pyver = line.split()[2]
    elif line and not line.startswith("#") and "==" in line:
        name, ver = line.split("==", 1)
        pins[norm(name)] = (name, ver)

if pyver is None:
    sys.exit("ERROR: snapshot has no '# python X.Y.Z' header — regenerate it with envs/dump_lumi_env.sh")

# torch: drop the ROCm local version (+rocm7.0...) and any pre-release/dev tag so
# the same X.Y.Z can be fetched as a CPU wheel.
torch = pins.get("torch")
if torch is None:
    sys.exit("ERROR: torch not in snapshot — is this really the LUMI training env?")
torch_base = re.match(r"\d+\.\d+\.\d+", torch[1]).group(0)

reqs, missing = [], []
for n in needed:
    hit = pins.get(norm(n))
    if hit:
        reqs.append(f"{hit[0]}=={hit[1]}")
    else:
        reqs.append(n)          # not on LUMI: let pip pick a compatible version
        missing.append(n)

open(f"{work}/pyver", "w").write(".".join(pyver.split(".")[:2]))
open(f"{work}/torch", "w").write(f"{torch_base}\n{torch[1]}")
open(f"{work}/requirements.txt", "w").write("\n".join(reqs) + "\n")
open(f"{work}/missing", "w").write(" ".join(missing))
PY

PYVER="$(cat "$WORK/pyver")"
TORCH_BASE="$(sed -n 1p "$WORK/torch")"
TORCH_LUMI="$(sed -n 2p "$WORK/torch")"
MISSING="$(cat "$WORK/missing")"

echo "Snapshot : $SNAPSHOT"
grep '^#' "$SNAPSHOT" | sed 's/^/           /'
echo "Env      : $ENV_NAME  (python $PYVER)"
echo "torch    : ${TORCH_BASE}+cpu   (LUMI: $TORCH_LUMI)"
echo "Pinned to LUMI versions:"
grep '==' "$WORK/requirements.txt" | sed 's/^/           /'
[[ -n "$MISSING" ]] && echo "Not on LUMI, unpinned: $MISSING"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1 — nothing installed."
    exit 0
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: conda env '$ENV_NAME' already exists. Remove it first:" >&2
    echo "         conda env remove -n $ENV_NAME" >&2
    exit 1
fi

conda create -y -n "$ENV_NAME" "python=$PYVER"
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"

"$PY" -m pip install --upgrade pip

# torch first, from the CPU index. PyPI's Linux torch wheels are CUDA builds
# (several GB); the constraint below stops later installs from swapping it for one.
"$PY" -m pip install "torch==$TORCH_BASE" --index-url "$TORCH_CPU_INDEX"
"$PY" -c "import torch; print(f'torch=={torch.__version__}')" > "$WORK/constraints.txt"

"$PY" -m pip install -r "$WORK/requirements.txt" -c "$WORK/constraints.txt" \
    --extra-index-url "$TORCH_CPU_INDEX"

# The repo itself, editable, so `from src ...` resolves exactly as in the diffms env.
# --no-deps / --ignore-requires-python: pyproject.toml pins python 3.9 and
# numpy==1.23 for the original DiffMS release; this env deliberately follows LUMI.
"$PY" -m pip install --no-deps --ignore-requires-python -e "$REPO"

# ── Verify ───────────────────────────────────────────────────────────────────
echo
echo "Verifying against the snapshot…"
"$PY" - "$SNAPSHOT" "$REPO" <<'PY'
import importlib.metadata as md, pickle, re, sys
from pathlib import Path
snapshot, repo = sys.argv[1], Path(sys.argv[2])
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
lumi = {norm(l.split("==")[0]): l.split("==")[1].strip()
        for l in open(snapshot) if "==" in l and not l.startswith("#")}

bad = 0
for dist in ["numpy", "pandas", "scipy", "rdkit", "torch", "torch_geometric",
             "pytorch_lightning", "torchmetrics", "hydra_core", "omegaconf"]:
    try:
        local = md.version(dist)
    except md.PackageNotFoundError:
        local = "MISSING"
    want = lumi.get(norm(dist), "-")
    same = local.split("+")[0] == want.split("+")[0]
    bad += (not same) and want != "-"
    print(f"  {'ok ' if same or want == '-' else 'DIFF'} {dist:20s} local={local:22s} lumi={want}")

import src, torch, torch_geometric, pytorch_lightning, rdkit, pandas  # noqa: F401
print(f"  imports ok; torch CUDA available: {torch.cuda.is_available()} (expected False)")

# The point of the whole exercise: can the pipeline's pickles be read?
pkls = sorted((repo / "data/neims").glob("*/mist_inputs/*.pkl"), key=lambda p: p.stat().st_size)
if pkls:
    with open(pkls[0], "rb") as f:
        pickle.load(f)
    print(f"  unpickled {pkls[0].relative_to(repo)} ok")
else:
    print("  (no data/neims pkls found locally — skipped unpickle check)")
sys.exit(1 if bad else 0)
PY

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
