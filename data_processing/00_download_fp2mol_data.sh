#!/usr/bin/env bash
# Download the raw structure databases for the fp2mol datasets into data/fp2mol/raw/.
# Safe to run from anywhere -- paths are anchored to the repo root rather than $PWD,
# which previously dropped the data into data_processing/data/ when run from there.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$REPO_ROOT/data/fp2mol/raw"

mkdir -p "$RAW"
cd "$RAW"

# Skip whatever is already here, so a re-run picks up where it stopped and a file that
# cannot be fetched automatically can be copied in by hand first.
fetch() {
    local url="$1" out="$2"
    if [[ -s "$out" ]]; then
        echo "[skip] $out already present"
        return 0
    fi
    if ! wget -O "$out.part" "$url"; then
        rm -f "$out.part"
        echo "ERROR: could not download $out from $url" >&2
        echo "       Copy it to $RAW/$out by hand and re-run." >&2
        return 1
    fi
    mv "$out.part" "$out"
}

# HMDB -> structures.sdf
# hmdb.ca is behind a bot check that rejects wget (403), so expect to copy this one in.
fetch https://hmdb.ca/system/downloads/current/structures.zip structures.zip
unzip -o structures.zip

# DSSTox -> DSSToxDump1..13.xlsx  (downloads as a file literally named "blob")
fetch https://clowder.edap-cluster.com/api/files/6616d8d7e4b063812d70fc95/blob blob
unzip -o blob

# COCONUT -> coconut_csv-03-2025.csv
fetch https://coconut.s3.uni-jena.de/prod/downloads/2025-03/coconut_csv-03-2025.zip coconut_csv-03-2025.zip
unzip -o coconut_csv-03-2025.zip

# MOSES -> moses.csv
fetch https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv moses.csv
