#!/bin/bash
#SBATCH --job-name=inf_diffms
#SBATCH --output=outfiles/inference_%j.out
#SBATCH --time=20:00:00        # see the runtime note below -- 2h is not enough at
                               # test_samples_to_generate=100
#SBATCH --account=project_462001155
#SBATCH --partition=small-g        # partial-node partition: standard-g bills whole nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # single process, no DDP -- see note below
#SBATCH --gpus-per-node=1          # one GCD is enough
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G

# ===========================================================================
# Pure inference on spectra with no known structure (e.g. the Yee dataset):
# formula in, generated molecules out.
#
# WHY SINGLE GPU
# --------------
# src/inference.py does not use the Lightning Trainer at all. It loads the
# checkpoint directly and loops over the dataloader itself, so there is no
# DistributedSampler, no DDP wrapper and no collectives -- launching it with
# more than one task would just run N identical copies over the same data and
# overwrite each other's output.
#
# RUNTIME scales as ceil(n_spectra / eval_batch_size) * num_samples * T decoder
# passes, all sequential on one GCD. For Yee at eval_batch_size=128 and
# test_samples_to_generate=100 that is 4 * 100 * 500 = 200,000 passes. Measured
# per-candidate cost on MI250X during the DDP eval runs was 83-374 s at batch 256,
# so roughly 40-190 s at batch 128 -> 4-21 h. Hence the 24 h walltime.
#
# The one lever is the candidate count, which inference.py now takes from the live
# config rather than the checkpoint:
#   sbatch --export=ALL,HYDRA_OVERRIDES=general.test_samples_to_generate=10 \
#          src/slurm/inference_lumi.sh
# That is a straight 10x and finishes inside an hour.
#
# Do NOT export RANK / LOCAL_RANK / WORLD_SIZE here. Nothing in this path reads
# them, and setting them only invites library code to believe it is distributed.
#
# The Lightning test path (src/slurm/test_model_lumi_ddp.sh) is the wrong tool
# for this data: test_step computes an NLL against ground-truth edges and builds
# true molecules from data.inchi, but for inference-only data that field holds
# "InChI=1S/<formula>" -- a formula string, not a parseable InChI. Every accuracy
# and similarity metric would be meaningless.
# ===========================================================================

module load Local-LAIF lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

# ---------------------------------------------------------------------------
# ROCm / MIOpen kernel caches. Same reasoning as the training script, minus the
# per-rank separation -- there is only one process.
# ---------------------------------------------------------------------------
export XDG_CACHE_HOME=$REAL/.rocm_cache/inference
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $XDG_CACHE_HOME $MIOPEN_USER_DB_PATH

# Single process owns all 7 CPUs, so let the math libraries use them.
export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7

# ---------------------------------------------------------------------------
# Singularity binds — MUST merge, not override. The module exports
# SINGULARITY_BIND with the host ROCm/libfabric paths; passing --bind on the
# command line replaces that variable instead of adding to it.
# ---------------------------------------------------------------------------
export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$REAL:$REAL:rw"

# ---------------------------------------------------------------------------
# Hydra overrides. src/inference.py now forwards argv to hydra.compose(), so
# these work the same way as for the other entry points, e.g.
#   sbatch --export=ALL,HYDRA_OVERRIDES="dataset.spec_folder=data/yee" \
#          src/slurm/inference_lumi.sh
# ---------------------------------------------------------------------------
HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-}"

start_time=$(date +%s)
echo "Starting inference on 1 GCD"
echo "Overrides: ${HYDRA_OVERRIDES:-<none>}"

srun singularity exec $SIF \
    bash -c "
        source $VENV/bin/activate
        cd $REAL/DiffEIMS

        export XDG_CACHE_HOME=$XDG_CACHE_HOME
        export MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH
        export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR
        export OMP_NUM_THREADS=$OMP_NUM_THREADS
        export MKL_NUM_THREADS=$MKL_NUM_THREADS

        python -c 'import torch; print(\"GPUs visible:\", torch.cuda.device_count())'

        # Must run from the repo root: inference.py uses the Compose API, so Hydra
        # never chdirs and every dataset path in the config stays repo-relative.
        python -m src.inference $HYDRA_OVERRIDES
    "

end_time=$(date +%s)
echo "Total runtime: $((end_time - start_time)) seconds"
