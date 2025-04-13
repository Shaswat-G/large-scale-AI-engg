import os
import torch
import torch.distributed as dist

def init_distributed():
    """
    Initialise the distributed environment.
    Assumes that environment variables RANK, LOCAL_RANK, and WORLD_SIZE are set.
    """
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


def create_batch(batch_size, input_dim, output_dim, seed=42, device="cuda"):
    """
    Create synthetic input and target tensors for a batch.
    Parameters:
    batch_size (int): Number of examples in the batch
    input_dim (int): Dimension of each input example
    seed (int): Random seed for reproducibility
    device (str): Device to create the tensors on ("cuda" or "cpu")
    Returns:
    tuple: (inputs, targets)
    - inputs: Tensor of shape (batch_size, input_dim) containing random values
    - targets: Tensor of shape (batch_size,) for regression tasks
    """
    torch.manual_seed(seed)
    # Create input tensor
    inputs = torch.randn(batch_size, input_dim, device=device)
    # Create regression targets
    targets = torch.randn(batch_size, output_dim, device=device)
    return inputs, targets
