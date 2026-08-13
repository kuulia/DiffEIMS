#!/bin/bash
#SBATCH --job-name=checkpoints
#SBATCH --output=outfiles/chk_%j.out
#SBATCH --time=00:15:00
#SBATCH --account=project_462001448
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb

module load Local-LAIF lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

start_time=$(date +%s)


srun --cpu-bind=cores \
    singularity exec $SIF \
    bash -c "
        source $VENV/bin/activate
        cd $REAL/DiffEIMS
	python src/checkpoint_to_weights.py data/checkpoints/checkpoints/fine-tuned/20940289-dev/last.ckpt data/checkpoints/checkpoints/fine-tuned/20940289-dev/
    "
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"

