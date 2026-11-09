#!/bin/bash
#SBATCH --job-name=rebuild_combined
#SBATCH --output=outfiles/rebuild_combined_%j.out
#SBATCH --partition=small          # LUMI-C partial node; the build is one process
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # billed as 16 cores either way: 2G/core
#SBATCH --time=12:00:00
#SBATCH --account=project_462001448 # CPU project; the GPU project is 462001155
#
# Download the raw structure databases and rebuild the leakage-guarded combined
# fp2mol dataset that the decoder pretraining reads (configs/dataset/fp2mol.yaml).
#
#   mkdir -p outfiles && sbatch src/slurm/rebuild_combined_lumi.sh
#
# Needs on LUMI before submitting:
#   notebooks/data/yee_new_0810/reference_inchi.csv   (gitignored, copy it over)
#   data/fp2mol/raw/structures.zip                    (hmdb.ca blocks wget, copy it over)
#
# Downloads already on disk are skipped. If this compute node cannot download, run
# `bash data_processing/00_download_fp2mol_data.sh` on a login node and resubmit.

module load Local-LAIF lumi-aif-singularity-bindings

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd "$REAL/DiffEIMS"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

start_time=$(date +%s)

# Fail now rather than after hours in the queue. The second check matters most: the
# old build_fp2mol_datasets.py would run to completion and leave the Yee molecules in.
[[ -s notebooks/data/yee_new_0810/reference_inchi.csv ]] \
    || die "notebooks/data/yee_new_0810/reference_inchi.csv is missing; copy it from the machine that has the annotations."
grep -q load_yee_reference_keys data_processing/build_fp2mol_datasets.py \
    || die "data_processing/build_fp2mol_datasets.py has no Yee guard; update this checkout."

bash data_processing/00_download_fp2mol_data.sh \
    || die "raw data is incomplete. See the message above, then resubmit."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Merge, don't override: the module above exports SINGULARITY_BIND.
export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$REAL:$REAL:rw"

# -d combined rebuilds only what the decoder reads and leaves the spectral CSVs alone.
# -e with the whole registry keeps the exclusion list of a full run.
srun singularity exec "$SIF" bash -c "
    source $VENV/bin/activate
    set -euo pipefail
    cd $REAL/DiffEIMS/data_processing

    # pd.read_excel needs this for DSSTox, and the venv did not have it.
    python -c 'import openpyxl' 2>/dev/null || pip install openpyxl

    EXCLUDE=\$(python -c \"from build_fp2mol_datasets import SPECTRAL_LAYOUT; print(' '.join(n for n, _ in SPECTRAL_LAYOUT))\")
    echo \"Exclusion sources: \$EXCLUDE\"

    python build_fp2mol_datasets.py --data-dir ../data -d combined -e \$EXCLUDE
"

# The build reports a dataset it could not process as [SKIP] and still exits 0, so
# confirm the CSVs were really written by this job.
for split in train val; do
    csv=data/fp2mol/combined/preprocessed/combined_$split.csv
    [[ -s "$csv" && $(stat -c %Y "$csv") -ge $start_time ]] \
        || die "$csv was not rewritten by this job; check the log for [SKIP]."
done

# FP2MolDataset reuses these whenever they exist, which would bring the old molecules
# straight back into training. Only reached once the build and its verification passed.
rm -fv data/fp2mol/combined/processed/*/train.pt data/fp2mol/combined/processed/*/val.pt

echo "Total runtime: $(($(date +%s) - start_time)) seconds"
