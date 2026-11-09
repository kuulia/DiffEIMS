#!/bin/bash
#SBATCH --job-name=combined_smiles
#SBATCH --output=outfiles/combined_smiles_%j.out
#SBATCH --partition=small          # LUMI-C partial node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=project_462001448 # CPU project
#
# Convert data/fp2mol/combined/preprocessed/combined_{train,val}.csv to one file of
# 2D SMILES, data/fp2mol/combined/combined_2d_smiles.csv.
#
#   mkdir -p outfiles && sbatch src/slurm/combined_to_smiles_lumi.sh

module load Local-LAIF lumi-aif-singularity-bindings

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd "$REAL/DiffEIMS"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Merge, don't override: the module above exports SINGULARITY_BIND.
export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$REAL:$REAL:rw"

# Newer Slurm does not always pass --cpus-per-task from sbatch down to srun.
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" singularity exec "$SIF" bash -c "
    source $VENV/bin/activate
    set -euo pipefail
    cd $REAL/DiffEIMS/data_processing
    python combined_inchi_to_smiles.py --data-dir ../data --workers $SLURM_CPUS_PER_TASK
"
