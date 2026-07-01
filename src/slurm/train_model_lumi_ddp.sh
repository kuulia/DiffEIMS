#!/bin/bash
#SBATCH --job-name=tr_diffms_ddp
#SBATCH --output=outfiles/ddp_%j.out
#SBATCH --time=24:00:00
#SBATCH --account=project_462001155
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8          # 8 GCDs per node (4 x MI250X, 2 GCDs each)
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7            # 56 CPUs / 8 tasks

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

# --- DDP environment (read by PL's env:// init via srun) ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# --- LUMI / RCCL ---
export NCCL_SOCKET_IFNAME=hsn        # Slingshot-11 high-speed interconnect
export RCCL_CROSS_NIC=1              # allow RCCL to use multiple NICs across nodes
export MIOPEN_DISABLE_CACHE=1        # avoid MIOpen cache corruption across nodes
export NCCL_DEBUG=WARN

start_time=$(date +%s)

echo "Running spec2mol_main.py (DDP: ${SLURM_NNODES} nodes x ${SLURM_NTASKS_PER_NODE} GCDs)"
srun singularity exec --rocm --bind $REAL:$REAL:rw $SIF bash -c "
source $VENV/bin/activate
cd $REAL/DiffEIMS
python src/spec2mol_main.py \
    general.gpus=8 \
    general.num_nodes=${SLURM_NNODES}
"

end_time=$(date +%s)
echo "Total runtime: $((end_time - start_time)) seconds"
