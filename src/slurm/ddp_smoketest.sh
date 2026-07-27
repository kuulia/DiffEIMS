#!/bin/bash
#SBATCH --job-name=ddp_smoke
#SBATCH --output=outfiles/smoke_%j.out
#SBATCH --time=00:15:00
#SBATCH --account=project_462001155
#SBATCH --partition=dev-g
#SBATCH --nodes=2                  # <-- run once with 1, then with 2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0

# Mirrors train_model_lumi_ddp.sh exactly, except:
#   - runs ddp_smoketest.py (no DiffEIMS imports) instead of spec2mol_main.py
#   - NCCL_DEBUG=INFO so we can see which transport RCCL selects
#   - 180 s init timeout inside the script, so it fails fast instead of hanging

module load Local-LAIF lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export NCCL_DEBUG=INFO             # <-- the point of this run
export NCCL_DEBUG_SUBSYS=INIT,NET

# --- CXI provider limits ---------------------------------------------------
# With 16 ranks x 16 channels the default CXI completion-queue and hardware
# match-list sizes are exhausted during tree setup, which hangs silently
# (rings connect, trees never do, no WARN emitted). These are the documented
# LUMI values for aws-ofi-nccl over CXI.
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_RX_MATCH_MODE=hybrid   # fall back to software matching on overflow
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# The lumi-aif-singularity-bindings module exports SINGULARITY_BIND with the host
# libfabric / libcxi / /dev/cxi* paths the aws-ofi-nccl plugin needs. Passing
# --bind on the command line OVERRIDES that variable instead of merging, which
# silently drops the CXI stack and makes RCCL fall back to NET/Socket. Append
# our bind to the variable and drop the flag.
export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$REAL:$REAL:rw"

echo "SMOKETEST: ${WORLD_SIZE} ranks (${SLURM_NNODES} nodes x ${SLURM_NTASKS_PER_NODE} GCDs)"
echo "MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "--- interfaces on the batch node ---"
ip -o link show | awk -F': ' '{print $2}' | tr '\n' ' '; echo
echo "--- SINGULARITY_BIND ---"
echo "$SINGULARITY_BIND"
echo "--- /dev/cxi devices ---"
ls -1 /dev/cxi* 2>/dev/null || echo "(none on batch node)"

srun --cpu-bind=cores \
    singularity exec $SIF \
    bash -c "
        source $VENV/bin/activate
        cd $REAL/DiffEIMS

        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID

        export XDG_CACHE_HOME=$REAL/.rocm_cache/\$SLURM_NODEID/\$SLURM_LOCALID
        export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}-\$SLURM_LOCALID
        export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH

        export WORLD_SIZE=$WORLD_SIZE
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
        export NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL
        export NCCL_DEBUG=$NCCL_DEBUG
        export NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS
        export FI_CXI_DEFAULT_CQ_SIZE=$FI_CXI_DEFAULT_CQ_SIZE
        export FI_CXI_DEFAULT_TX_SIZE=$FI_CXI_DEFAULT_TX_SIZE
        export FI_CXI_RX_MATCH_MODE=$FI_CXI_RX_MATCH_MODE
        export FI_MR_CACHE_MONITOR=$FI_MR_CACHE_MONITOR
        export FI_CXI_DISABLE_HOST_REGISTER=$FI_CXI_DISABLE_HOST_REGISTER
        export OMP_NUM_THREADS=$OMP_NUM_THREADS
        export MKL_NUM_THREADS=$MKL_NUM_THREADS

        python src/slurm/ddp_smoketest.py
    "

echo "Smoketest exit code: $?"
