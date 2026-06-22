#!/bin/bash
#SBATCH --job-name=tr_diffms_e2e
#SBATCH --output=outfiles/e2e_ft_%A.out
#SBATCH --time=32:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0

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
