#!/bin/bash
#SBATCH --job-name=parse_results
#SBATCH --output=outfiles/%A_parse.out
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Load environment
cd $WRKDIR || exit 1
module load mamba
source activate diffms

# Navigate to project directory
cd ms/DiffMS || exit 1

# Collect matching .out files (edit glob patterns to adjust the range)
FILES=$(ls outfiles/1*.out 2>/dev/null)

if [ -z "$FILES" ]; then
    echo "No matching .out files found."
    exit 1
fi

echo "Parsing files:"
echo "$FILES"
echo ""

python parse_outfiles.py $FILES -o results.csv

echo "Done. Output: results.csv"
