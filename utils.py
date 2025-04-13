import os
import torch
import torch.distributed as dist

def init_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_id = os.environ.get("SLURM_NODEID", "N/A")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    print(f"[Distributed Init] Rank {rank} initialized on {node_id} on GPU {local_rank}.")
    dist.barrier()
    if rank == 0:
        print(f"[Rank {rank}] All ranks ready!")
    return rank, local_rank, world_size
