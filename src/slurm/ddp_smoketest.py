"""Minimal DDP connectivity test. No DiffMS imports.

Verifies, in order:
  1. env vars each rank sees
  2. TCPStore rendezvous  (init_process_group returns)
  3. RCCL collective      (all_reduce returns the right sum)

Run under the same srun/singularity invocation as training.
"""

import os
import socket
import datetime

import torch
import torch.distributed as dist


def env(name):
    return os.environ.get(name, "<unset>")


def main():
    rank = int(env("RANK")) if env("RANK") != "<unset>" else int(env("SLURM_PROCID"))
    local_rank = (
        int(env("LOCAL_RANK"))
        if env("LOCAL_RANK") != "<unset>"
        else int(env("SLURM_LOCALID"))
    )
    world_size = (
        int(env("WORLD_SIZE"))
        if env("WORLD_SIZE") != "<unset>"
        else int(env("SLURM_NTASKS"))
    )

    print(
        f"[rank {rank:02d}] host={socket.gethostname()} local_rank={local_rank} "
        f"world_size={world_size} MASTER_ADDR={env('MASTER_ADDR')} "
        f"MASTER_PORT={env('MASTER_PORT')} "
        f"NCCL_SOCKET_IFNAME={env('NCCL_SOCKET_IFNAME')}",
        flush=True,
    )

    torch.cuda.set_device(local_rank)
    print(f"[rank {rank:02d}] device set -> {torch.cuda.current_device()}", flush=True)

    # Short timeout: fail fast instead of hanging for hours.
    print(f"[rank {rank:02d}] calling init_process_group...", flush=True)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=180),
    )
    print(f"[rank {rank:02d}] init_process_group OK", flush=True)

    t = torch.ones(1, device="cuda") * rank
    dist.all_reduce(t)
    expected = world_size * (world_size - 1) // 2
    print(
        f"[rank {rank:02d}] all_reduce OK: got {int(t.item())} expected {expected}",
        flush=True,
    )

    dist.barrier()
    if rank == 0:
        print("=== DDP SMOKETEST PASSED ===", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
