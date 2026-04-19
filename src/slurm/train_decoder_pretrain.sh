#!/bin/bash
#SBATCH --job-name=tr_diffms_dec
#SBATCH --output=%A_%a.out
#SBATCH --time=35:00:00
#SBATCH --mem=128G
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

# Start GPU logging
(
while true; do
    echo "==== $(date) ====" >> gpu_log_${SLURM_JOB_ID}.txt
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits >> gpu_log_${SLURM_JOB_ID}.txt
    sleep 10
done
) &
LOG_PID=$!

# Run training
srun python src/fp2mol_main.py

kill $LOG_PID 2>/dev/null
wait $LOG_PID 2>/dev/null

# Record end time and report runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"
