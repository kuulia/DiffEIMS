#!/bin/bash
#SBATCH --job-name=te_diffms_ddp
#SBATCH --output=outfiles/test_%j.out
#SBATCH --time=24:00:00
#SBATCH --account=project_462001155
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8        # one task per GCD; torchrun is NOT used
#SBATCH --gpus-per-node=8          # 8 GCDs per node (4 x MI250X)
#SBATCH --cpus-per-task=7          # 56 usable CPUs / 8 GCDs = 7 per GCD (8 cores reserved by ROCm driver)
#SBATCH --mem=0                    # all available node memory (~512 GB / node)

# ===========================================================================
# Evaluate a trained checkpoint on the test set.
#
# WHY THIS DOES NOT USE general.test_only
# ---------------------------------------
# Setting general.test_only sends Lightning down its evaluation-only branch,
# where DDPStrategy.setup() calls _sync_module_states() directly instead of
# constructing DistributedDataParallel. On 2026-07-28/29 that path hung on every
# attempt at 32 ranks:
#
#   WorkNCCL(SeqNum=9, OpType=BROADCAST, NumelIn=72579445) ran for 28800014 ms
#
# The NCCL flight recorder showed all 32 ranks enqueued that identical broadcast
# (same seq, same op, same size, all reporting "last enqueued 10, last completed
# 8"), so it is not a rank desync — one agreed-upon collective simply never
# completes. Root cause still unknown.
#
# trainer.fit() reaches DDP through DistributedDataParallel.__init__ instead, and
# that path clears the same broadcast reliably. So this script runs a deliberately
# trivial fit (one batch, lr=0) purely to get through DDP setup, and then lets
# main() fall through to its usual post-fit trainer.test().
#
# lr=0 rather than frozen weights: with every parameter frozen, DDP raises
# "DistributedDataParallel is not needed when a module doesn't have any parameter
# that requires a gradient" and configure_optimizers gets an empty list. AdamW at
# lr=0 applies no update and no decoupled weight decay, so the weights that get
# tested are bit-identical to the checkpoint.
#
# main() then calls trainer.test(ckpt_path=cfg.general.checkpoint_strategy), i.e.
# the 'last' checkpoint this job just wrote — same weights, by the argument above.
#
# Revert to plain test_only once the SeqNum=9 broadcast is understood.
# ===========================================================================

# ---------------------------------------------------------------------------
# What to evaluate. Override either from the command line, e.g.
#   sbatch --export=ALL,CKPT=/path/to/other.ckpt src/slurm/test_model_lumi_ddp.sh
# ---------------------------------------------------------------------------
REAL=$(realpath /scratch/project_462001155/lindl)
CKPT="${CKPT:-$REAL/DiffEIMS/data/checkpoints/checkpoints/gecko_new_ddp/fine_tuned.ckpt}"

# Sampling batch size. Cost per batch scales with bs * max_nodes^2 because
# to_dense pads every batch to its largest molecule, so a single outsized molecule
# taxes its whole batch. At 256 that blast radius made rank 8 run 4.5x slower than
# rank 17 (374 vs 83 s per candidate, 2026-07-29). Paired with
# train.sort_test_by_size, which makes batches near-homogeneous in size.
#
# Drop to 64 if the skew is still bad. Simulated over the full sampler+batching
# pipeline (18k molecules, lognormal sizes, world=32), slowest-rank cost measured
# against one fixed reference — the unsorted bs=256 setup that ran on 2026-07-29:
#     reference: bs=256 unsorted -> 1.00x, imbalance 2.75x, 2 batches/rank
#     bs=256 sorted              -> 0.86x, imbalance 2.62x, 2 batches/rank
#     bs=128 sorted              -> 0.46x, imbalance 2.39x, 4 batches/rank
#     bs=64  sorted              -> 0.26x, imbalance 2.05x, 8 batches/rank
# Note most of the win is total cost, not balance: imbalance barely moves, because
# whichever rank draws the single largest molecule still pays max(n)^2 across its
# batch. Smaller bs shrinks how many molecules that outlier taxes.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

# ---------------------------------------------------------------------------
# LUMI: load modules
# ---------------------------------------------------------------------------
module load Local-LAIF lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

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
# NCCL Flight Recorder — a ring buffer of the last N collectives per rank.
# On a watchdog timeout PyTorch dumps it, which is the only way to see WHICH
# rank enqueued something other than the expected collective. Without it the
# timeout message says only "Stack trace of the failed collective not found".
# Cost is a fixed-size in-memory buffer per rank; safe to leave on permanently.
# ---------------------------------------------------------------------------
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$REAL/nccl_trace/rank_

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

mkdir -p $REAL/nccl_trace

# ---------------------------------------------------------------------------
# Hydra overrides. The fixed block below is what makes this an eval run; anything
# in $HYDRA_OVERRIDES is appended and therefore wins on conflict, e.g.
#   sbatch --export=ALL,HYDRA_OVERRIDES=general.num_test_samples=4096 \
#          src/slurm/test_model_lumi_ddp.sh
# ---------------------------------------------------------------------------
EVAL_OVERRIDES="general.test_only=null \
    general.load_weights=$CKPT \
    train.lr=0 \
    train.n_epochs=1 \
    train.limit_train_batches=1 \
    train.limit_val_batches=1 \
    train.scheduler=const \
    train.eval_batch_size=$EVAL_BATCH_SIZE"
HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-}"

# Pre-create per-node per-rank cache dirs so ROCm doesn't race on mkdir.
# One task per node creates that node's subtree using its SLURM_NODEID (0, 1, ...).
# --ntasks is needed too: without it SLURM warns that --ntasks-per-node=1 does
# not match the job's requested task count.
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
    for i in \$(seq 0 7); do mkdir -p $REAL/.rocm_cache/\$SLURM_NODEID/\$i; done
    for i in \$(seq 0 7); do mkdir -p /tmp/${USER}-miopen-${SLURM_JOB_ID}-\$i; done
"

start_time=$(date +%s)
echo "Starting DDP evaluation: ${WORLD_SIZE} ranks (${SLURM_NNODES} nodes x ${SLURM_NTASKS_PER_NODE} GCDs)"
echo "MASTER: ${MASTER_ADDR}:${MASTER_PORT}  |  WORLD_SIZE: ${WORLD_SIZE}"
echo "CHECKPOINT: ${CKPT}"
echo "EVAL_BATCH_SIZE: ${EVAL_BATCH_SIZE}"

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
        # MUST include SLURM_NODEID: LOCAL_RANK is 0-7 on every node, so without
        # it two nodes write to the same Lustre path simultaneously, causing a
        # ROCm file-lock deadlock that hangs all of the second node's ranks.
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
        export TORCH_NCCL_TRACE_BUFFER_SIZE=$TORCH_NCCL_TRACE_BUFFER_SIZE
        export TORCH_NCCL_DUMP_ON_TIMEOUT=$TORCH_NCCL_DUMP_ON_TIMEOUT
        export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$TORCH_NCCL_DEBUG_INFO_TEMP_FILE
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
            general.num_nodes=$SLURM_NNODES \
            $EVAL_OVERRIDES \
            $HYDRA_OVERRIDES
    "

end_time=$(date +%s)
echo "Total runtime: $((end_time - start_time)) seconds"
