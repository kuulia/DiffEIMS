#!/bin/bash
#SBATCH --job-name=prebuild_decoder
#SBATCH --output=outfiles/prebuild_decoder_%j.out
#SBATCH --partition=small          # LUMI-C partial node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # 4 joblib workers, then 6 DataLoader workers for stats
#SBATCH --mem=96G                  # ~26 GB of graphs in a list, then ~17 GB collated
#SBATCH --time=08:00:00
#SBATCH --account=project_462001448 # CPU project; the GPU project is 462001155
#
# Build the combined dataset's graph cache and stat files on CPU, ahead of decoder
# pretraining. Otherwise rank 0 of the GPU job does it while the other 31 ranks wait,
# and they abort after _SENTINEL_TIMEOUT (2 h) in src/fp2mol_main.py.
#
# Run after rebuild_combined_lumi.sh has finished:
#
#   mkdir -p outfiles && sbatch src/slurm/prebuild_decoder_cache_lumi.sh

module load Local-LAIF lumi-aif-singularity-bindings

set -euo pipefail
shopt -s nullglob

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd "$REAL/DiffEIMS"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

start_time=$(date +%s)

COMBINED=data/fp2mol/combined
SENTINEL=data/fp2mol/.dataset_ready__fp2mol__r2__n2048
CSV=$COMBINED/preprocessed/combined_train.csv

[[ -s "$CSV" ]] || die "$CSV is missing; run rebuild_combined_lumi.sh first."

# Existing files are reused rather than rebuilt, so anything older than the CSVs would
# carry the pre-guard molecules into training. Newer files are from an earlier attempt.
for f in "$COMBINED"/processed/*/train.pt "$COMBINED"/processed/*/val.pt "$COMBINED"/stats/*.txt; do
    [[ "$f" -nt "$CSV" ]] || die "$f is older than $CSV; delete it and resubmit."
done

mkdir -p "$COMBINED/stats"

# The marker tells the DDP ranks the cache is finished, so it must only exist afterwards.
rm -f "$SENTINEL"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Merge, don't override: the module above exports SINGULARITY_BIND.
export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$REAL:$REAL:rw"

# Newer Slurm does not always pass --cpus-per-task from sbatch down to srun.
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" singularity exec "$SIF" bash -c "
    source $VENV/bin/activate
    set -euo pipefail
    cd $REAL/DiffEIMS
    python src/fp2mol_main.py --config-name=config_decoder_pretrain \
        general.gpus=0 general.num_nodes=1 +general.prepare_data_only=true
"

for f in "$COMBINED"/processed/*/train.pt "$COMBINED"/processed/*/val.pt \
    "$COMBINED"/stats/{n_counts,atom_types,edge_types,valencies}.txt "$SENTINEL"; do
    [[ -e "$f" ]] || die "$f was not produced."
    ls -l "$f"
done
compgen -G "$COMBINED/processed/*/train.pt" > /dev/null || die "no train.pt was produced."

echo "Total runtime: $(($(date +%s) - start_time)) seconds"
