#!/bin/bash
#SBATCH --job-name=tr_diffms_e2e
#SBATCH --output=outfiles/e2e_ft_%j.out
#SBATCH --time=00:29:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dev-g

# Load environment
cd $WRKDIR || exit 1
module load mamba
source activate diffms

# Navigate to project directory
cd ms/DiffMS || exit 1

# Record start time
start_time=$(date +%s)

# Run training
echo "srun python src/spec2mol_main.py"
srun python src/spec2mol_main.py

# Record end time and report runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"
