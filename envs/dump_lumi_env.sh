#!/bin/bash
# Snapshot the Python environment the NEIMS pipeline and DiffMS training use on LUMI,
# so a matching local environment can be built with envs/build_local_env.sh.
#
# Why this exists: the collated/subform .pkl files are written inside the LUMI
# container + pyg_venv (NumPy 2.x). The repo's pinned local env (numpy==1.23)
# cannot unpickle them ("No module named 'numpy._core'"), so local runs need an
# environment that matches LUMI's versions rather than pyproject.toml's pins.
#
# Run on a LUMI login node (cheap, a few seconds, no allocation needed):
#
#   bash envs/dump_lumi_env.sh                # writes envs/lumi_env.txt
#   bash envs/dump_lumi_env.sh /path/out.txt
#
# Then copy the file into the local checkout, e.g.
#   scp lumi:/scratch/project_462001155/$USER/DiffEIMS/envs/lumi_env.txt envs/
#
# Overridable (same defaults as neims-pipeline/run_pipeline_lumi.sh):
#   PROJECT_ROOT  [/scratch/project_462001155/$USER]
#   VENV          [$PROJECT_ROOT/pyg_venv]
#   SIF           [LUMI AI Factory multitorch image used by the pipeline]

set -euo pipefail

OUT="${1:-$(dirname "$0")/lumi_env.txt}"

module load Local-LAIF lumi-aif-singularity-bindings

SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif}"
REAL="$(realpath "${PROJECT_ROOT:-/scratch/project_462001155/$USER}")"
VENV="${VENV:-$REAL/pyg_venv}"

for p in "$SIF" "$VENV"; do
    [[ -e "$p" ]] || { echo "ERROR: not found: $p (override via the env vars above)" >&2; exit 1; }
done

export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$REAL:$REAL:rw"

# Header lines start with '#', one package per line after that. `pip list
# --format=freeze` (not `pip freeze`) always prints name==version, even for
# packages the container built from source, and — because pyg_venv is layered on
# the container — it includes the container's system site-packages too, which is
# where numpy/pandas/torch actually come from.
singularity exec "$SIF" bash -c "
    source $VENV/bin/activate
    echo \"# python \$(python -c 'import platform; print(platform.python_version())')\"
    echo \"# sif $SIF\"
    echo \"# venv $VENV\"
    echo \"# \$(grep -i include-system-site-packages $VENV/pyvenv.cfg || echo 'include-system-site-packages = unknown')\"
    echo \"# dumped \$(date -u +%Y-%m-%dT%H:%M:%SZ) on \$(hostname)\"
    python -m pip list --format=freeze 2>/dev/null
" > "$OUT"

echo "Wrote $OUT ($(grep -vc '^#' "$OUT") packages)"
grep '^# python' "$OUT"
grep -iE '^(numpy|pandas|torch|torch[-_]geometric|pytorch[-_]lightning|rdkit)==' "$OUT" || true
