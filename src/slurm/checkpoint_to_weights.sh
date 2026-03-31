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

# Run script 09-03-01-dev  09-07-21-dev  09-09-00-dev  09-12-03-dev
srun python src/checkpoint_to_weights.py data/checkpoints/checkpoints/tms/no_gecko/1/decoder.ckpt data/checkpoints/checkpoints/tms/no_gecko/1/ --decoder

# Record end time and report runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"
