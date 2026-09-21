#!/bin/bash
#SBATCH --job-name=preprocess_fp2mol
#SBATCH --output=outfiles/preprocess_%j.out
#SBATCH --partition=small          # LUMI-C partial node; this job is single-process
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # no MPI, no DDP — build_guarded_splits forks a Pool
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                  # 2G/core — the ratio above which billing jumps
#SBATCH --time=12:00:00
#SBATCH --account=project_462001448 # CPU project; the GPU project is 462001155
#
# Build the preprocessed InChI splits for a spectral dataset on a compute node.
# Same container / venv as train_model_lumi_ddp.sh, minus everything GPU and
# multi-node: no NCCL, no CXI tuning, no ROCm caches, one rank.
#
#   sbatch src/slurm/preprocess_fp2mol_lumi.sh
#   sbatch --export=ALL,DATASET=gecko_tms src/slurm/preprocess_fp2mol_lumi.sh
#   sbatch --export=ALL,DATASET=ALL src/slurm/preprocess_fp2mol_lumi.sh
#
# DATASET=ALL runs the full 03_preprocess_fp2mol.sh pipeline (every spectral set
# plus hmdb/dss/coconut/moses/combined) and takes many hours. A single DATASET
# runs only that one, but with the exclusion list of a full run (see below).

module load Local-LAIF lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

DATASET="${DATASET:-gecko_tms_mixed_augment_tms_atmomaccs_tms_test}"

# RDKit and the multiprocessing Pool must not oversubscribe the allocated cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Merge, don't override: the module above exports SINGULARITY_BIND, and passing
# --bind on the command line would replace it. Harmless here (no fabric needed),
# but keeping the same idiom as the training script avoids a surprise if this
# script is ever copied.
export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$REAL:$REAL:rw"

start_time=$(date +%s)
echo "Preprocessing dataset: $DATASET  |  ${SLURM_CPUS_PER_TASK} CPUs"

srun singularity exec $SIF bash -c "
    source $VENV/bin/activate
    cd $REAL/DiffEIMS

    set -euo pipefail

    if [ \"$DATASET\" = ALL ]; then
        bash data_processing/03_preprocess_fp2mol.sh
        exit \$?
    fi

    mkdir -p data/neims/$DATASET/mist_inputs/{preprocessed,processed,stats}
    cd data_processing

    # A full run excludes the test/val molecules of EVERY spectral dataset from
    # every train split; a single -d run excludes nothing but the atmomaccs
    # leakage guard. Passing the whole registry as -e reproduces the full run's
    # exclusion set for this one dataset, without rebuilding the structure-only
    # datasets. Read from SPECTRAL_LAYOUT so the list cannot drift.
    EXCLUDE=\$(python -c \"from build_fp2mol_datasets import SPECTRAL_LAYOUT; print(' '.join(n for n, _ in SPECTRAL_LAYOUT))\")
    echo \"Exclusion sources: \$EXCLUDE\"

    python build_guarded_splits.py --data-dir ../data \
        --workers ${SLURM_CPUS_PER_TASK} \
        -d $DATASET

    python build_fp2mol_datasets.py --data-dir ../data \
        -d $DATASET \
        -e \$EXCLUDE
"
rc=$?

end_time=$(date +%s)
echo "Total runtime: $((end_time - start_time)) seconds (exit $rc)"
exit $rc
