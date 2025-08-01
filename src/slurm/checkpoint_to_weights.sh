#!/bin/bash
#SBATCH --job-name=checkpoints
#SBATCH --output=%A_%a.out
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --array=0

# Load environment
cd $WRKDIR || exit 1
module load mamba
source activate diffms

# Navigate to project directory
cd ms/DiffMS || exit 1

# Record start time
start_time=$(date +%s)

# Run script
srun python src/checkpoint_to_weights.py outputs/2025-08-01/09-23-40-dev/checkpoints/dev/last-v1.ckpt data/checkpoints/checkpoints/ --encoder

# Record end time and report runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"
