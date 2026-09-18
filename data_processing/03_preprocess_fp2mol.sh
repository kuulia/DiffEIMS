#!/usr/bin/env bash
# Build the preprocessed InChI train/val/test splits consumed by FP2MolDataset and
# NeimsDataset. Safe to run from anywhere -- all paths are anchored to the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$REPO_ROOT/data"

# Structure-only fp2mol datasets: data/fp2mol/<dataset>/
for dataset in hmdb dss coconut moses combined
do
    mkdir -p "$DATA/fp2mol/$dataset/preprocessed"
    mkdir -p "$DATA/fp2mol/$dataset/processed"
    mkdir -p "$DATA/fp2mol/$dataset/stats"
done

# Spectral datasets: labels.tsv and split_random.tsv live inside a mist_inputs dir
# whose depth varies per dataset, and the preprocessed CSVs are written next to them.
# Keep this list in sync with SPECTRAL_LAYOUT in build_fp2mol_datasets.py.
for rel in \
    atmomaccs_new/mist_inputs \
    atmomaccs_new/mist_inputs_tms \
    mixed_augment/mist_inputs/mixed_augment \
    mixed_augment_tms/mist_inputs/mixed_augment_tms \
    gecko_new/mist_inputs \
    gecko_tms/mist_inputs \
    gecko_new_atmomaccs_test/mist_inputs \
    gecko_new_mixed_augment_atmomaccs_test/mist_inputs \
    gecko_tms_mixed_augment_atmomaccs_tms_test/mist_inputs \
    combined_atmomaccs_test/mist_inputs
do
    mkdir -p "$DATA/neims/$rel/preprocessed"
    mkdir -p "$DATA/neims/$rel/processed"
    mkdir -p "$DATA/neims/$rel/stats"
done

cd "$REPO_ROOT/data_processing"

# Guarded split files first: build_fp2mol_datasets.py reads split_random_guarded.tsv
# for every non-atmomaccs dataset, and spec2mol trains from the same files.
python build_guarded_splits.py --data-dir "$DATA"

python build_fp2mol_datasets.py --data-dir "$DATA" "$@"
