#!/bin/bash
#SBATCH --job-name=tr_diffms_ddp
#SBATCH --output=outfiles/ddp_%j.out
#SBATCH --time=00:29:00
#SBATCH --account=project_462001155
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --gpus-per-node=8            # 8 GCDs per node (4 x MI250X)
#SBATCH --cpus-per-task=56           # all CPUs per node for torchrun to distribute

module load Local-LAIF lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# MIOpen: per-user cache to avoid cross-node conflicts
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

export OMP_NUM_THREADS=1

start_time=$(date +%s)

echo "Running spec2mol_main.py (DDP: ${SLURM_NNODES} nodes x 8 GCDs)"
srun singularity run --bind $REAL:$REAL:rw $SIF bash -c "
source $VENV/bin/activate
cd $REAL/DiffEIMS
python -m torch.distributed.run \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    src/spec2mol_main.py \
    general.gpus=8 \
    general.num_nodes=${SLURM_NNODES}
"

end_time=$(date +%s)
echo "Total runtime: $((end_time - start_time)) seconds"
