#!/bin/bash
#SBATCH --job-name=eval_diffms
#SBATCH --output=%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0

# Load environment
cd $WRKDIR || exit 1
module load mamba
source activate diffms

# Navigate to project directory
cd ms/DiffMS/src || exit 1

# Record start time
start_time=$(date +%s)

# Run training
srun python eval_test.py

# Record end time and report runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"
