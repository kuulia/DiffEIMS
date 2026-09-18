#!/usr/bin/env bash
# Download the raw structure databases for the fp2mol datasets into data/fp2mol/raw/.
# Safe to run from anywhere -- paths are anchored to the repo root rather than $PWD,
# which previously dropped the data into data_processing/data/ when run from there.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$REPO_ROOT/data/fp2mol/raw"

mkdir -p "$RAW"
cd "$RAW"

# HMDB -> structures.sdf
wget https://hmdb.ca/system/downloads/current/structures.zip
unzip -o structures.zip

# DSSTox -> DSSToxDump1..13.xlsx  (downloads as a file literally named "blob")
wget https://clowder.edap-cluster.com/api/files/6616d8d7e4b063812d70fc95/blob
unzip -o blob

# COCONUT -> coconut_csv-03-2025.csv
wget https://coconut.s3.uni-jena.de/prod/downloads/2025-03/coconut_csv-03-2025.zip
unzip -o coconut_csv-03-2025.zip

# MOSES -> moses.csv
wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv
mv dataset_v1.csv moses.csv
