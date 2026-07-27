#!/bin/bash
#SBATCH --job-name=tr_diffms_ddp
#SBATCH --output=outfiles/ddp_%j.out
#SBATCH --time=24:00:00
#SBATCH --account=project_462001155
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8        # one task per GCD; torchrun is NOT used
#SBATCH --gpus-per-node=8          # 8 GCDs per node (4 x MI250X)
#SBATCH --cpus-per-task=7          # 56 usable CPUs / 8 GCDs = 7 per GCD (8 cores reserved by ROCm driver)
#SBATCH --mem=0                    # all available node memory (~512 GB / node)

# ---------------------------------------------------------------------------
# LUMI: load modules
# ---------------------------------------------------------------------------
module load Local-LAIF lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

# ---------------------------------------------------------------------------
# Rendezvous — use the first hostname in the job's node list
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# ---------------------------------------------------------------------------
# NCCL — tell it to use LUMI's Slingshot high-speed network (hsn*)
# Without this NCCL falls back to the slow management network and hangs on
# collective operations.
# ---------------------------------------------------------------------------
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3          # enable GPU Direct RDMA over RoCE
export NCCL_DEBUG=WARN               # raise to INFO only for debugging hangs

# ---------------------------------------------------------------------------
# CXI provider limits — REQUIRED for multi-node.
# With 16 ranks x 16 channels the default CXI completion-queue and hardware
# match-list sizes are exhausted during RCCL *tree* setup. The symptom is a
# silent hang inside ncclCommInitRank: "Connected all rings" appears for every
# rank, "Connected all trees" never does, and no NCCL WARN is emitted. Verified
# by src/slurm/ddp_smoketest.sh — without these, 0/16 ranks reach Init COMPLETE;
# with them, 16/16 do (init ~19 s).
# ---------------------------------------------------------------------------
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_RX_MATCH_MODE=hybrid   # fall back to software matching on overflow
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1

# ---------------------------------------------------------------------------
# ROCm / MIOpen — keep per-rank kernel caches in /tmp (node-local, fast).
# Include SLURM_LOCALID so 8 ranks on the same node don't share one cache dir
# (MIOpen file-locks the directory; sharing causes contention).
# ---------------------------------------------------------------------------
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}-${SLURM_LOCALID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# ---------------------------------------------------------------------------
# Threading — each task owns 7 CPUs; set to 1 to avoid over-subscription
# with DataLoader workers (cfg.train.num_workers=6, plus the main process).
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Torch distributed — WORLD_SIZE is the same for every task; compute it now.
# RANK and LOCAL_RANK are per-task and must be derived inside srun (where
# SLURM_PROCID / SLURM_LOCALID are set correctly for each task).
# IMPORTANT: do NOT pre-set RANK or LOCAL_RANK here — every task would see
# rank 0 because SLURM_PROCID is 0 in the batch-script environment.
# ---------------------------------------------------------------------------
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# ---------------------------------------------------------------------------
# Singularity binds — MUST merge, not override.
# lumi-aif-singularity-bindings exports SINGULARITY_BIND containing the host
# libfabric / libcxi / /dev/cxi* paths that the aws-ofi-nccl plugin needs to
# open a CXI fabric domain. Passing --bind on the command line REPLACES that
# variable, so the CXI stack goes missing, fi_domain() fails with ENOSYS, and
# RCCL silently falls back to NET/Socket. That is invisible on one node (all
# GCDs talk via P2P/IPC) and hangs forever on two.
# ---------------------------------------------------------------------------
export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$REAL:$REAL:rw"

# ---------------------------------------------------------------------------
# Run
#
# We use --ntasks-per-node=8 (one srun task per GCD) rather than
# --ntasks-per-node=1 + torchrun because:
#   - srun gives each task its own CPU affinity mask (7 CPUs per GCD)
#   - SLURM sets SLURM_PROCID / SLURM_LOCALID per task; we map them to
#     RANK / LOCAL_RANK inside the bash -c (note the \$ to defer evaluation)
#   - Lightning picks up RANK / LOCAL_RANK / WORLD_SIZE and uses DDPStrategy
#     without spawning additional processes
#   - No rendezvous port conflicts between torchrun instances
#
# Variables that are the same for all tasks (MASTER_ADDR, NCCL_*, WORLD_SIZE,
# VENV, REAL, ...) are expanded early by the outer shell (no backslash).
# Variables that differ per task (SLURM_PROCID, SLURM_LOCALID) are escaped
# with \$ so they are evaluated at runtime inside each srun task.
# ---------------------------------------------------------------------------
# Pre-create per-node per-rank cache dirs so ROCm doesn't race on mkdir.
# One task per node creates that node's subtree using its SLURM_NODEID (0, 1, ...).
srun --ntasks-per-node=1 bash -c "
    for i in \$(seq 0 7); do mkdir -p $REAL/.rocm_cache/\$SLURM_NODEID/\$i; done
    for i in \$(seq 0 7); do mkdir -p /tmp/${USER}-miopen-${SLURM_JOB_ID}-\$i; done
"

start_time=$(date +%s)
echo "Starting DDP training: ${WORLD_SIZE} ranks (${SLURM_NNODES} nodes x ${SLURM_NTASKS_PER_NODE} GCDs)"
echo "MASTER: ${MASTER_ADDR}:${MASTER_PORT}  |  WORLD_SIZE: ${WORLD_SIZE}"

srun --cpu-bind=cores \
    singularity exec $SIF \
    bash -c "
        source $VENV/bin/activate
        cd $REAL/DiffEIMS

        # Per-task rank vars: evaluated at runtime in each srun task
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID

        # ROCm / RCCL kernel cache — persistent on scratch so compiled kernels
        # survive across jobs (first-time RCCL compilation takes 10-20 min).
        # MUST include SLURM_NODEID: LOCAL_RANK is 0-7 on BOTH nodes, so without
        # it node 0 and node 1 write to the same Lustre path simultaneously,
        # causing ROCm file-lock deadlock and hanging all of node 1's ranks.
        export XDG_CACHE_HOME=$REAL/.rocm_cache/\$SLURM_NODEID/\$SLURM_LOCALID
        # MIOpen user DB — per-rank on /tmp (node-local, fast; MIOpen re-compiles
        # per run anyway and the files are small enough not to matter).
        export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}-\$SLURM_LOCALID
        export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH

        # Same for all tasks — expanded by outer shell
        export WORLD_SIZE=$WORLD_SIZE
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
        export NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL
        export NCCL_DEBUG=$NCCL_DEBUG
        export FI_CXI_DEFAULT_CQ_SIZE=$FI_CXI_DEFAULT_CQ_SIZE
        export FI_CXI_DEFAULT_TX_SIZE=$FI_CXI_DEFAULT_TX_SIZE
        export FI_CXI_RX_MATCH_MODE=$FI_CXI_RX_MATCH_MODE
        export FI_MR_CACHE_MONITOR=$FI_MR_CACHE_MONITOR
        export FI_CXI_DISABLE_HOST_REGISTER=$FI_CXI_DISABLE_HOST_REGISTER
        export OMP_NUM_THREADS=$OMP_NUM_THREADS
        export MKL_NUM_THREADS=$MKL_NUM_THREADS

        echo \"Task RANK=\$RANK LOCAL_RANK=\$LOCAL_RANK MIOPEN=\$MIOPEN_USER_DB_PATH\"

        python src/spec2mol_main.py \
            general.gpus=8 \
            general.num_nodes=$SLURM_NNODES
    "

end_time=$(date +%s)
echo "Total runtime: $((end_time - start_time)) seconds"
