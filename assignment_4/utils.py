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

def create_batch(batch_size, input_dim, output_dim, seed=42, device="cuda"):
    torch.manual_seed(seed)
    inputs = torch.randn(batch_size, input_dim, device=device)
    targets = torch.randn(batch_size, output_dim, device=device)
    return inputs, targets

def check(tensor, expected_shape):
    if tensor is None:
        return
    assert isinstance(tensor, torch.Tensor), "Provided tensor is not a torch.Tensor!"
    actual_shape = list(tensor.detach().shape)
    assert isinstance(expected_shape, (list, tuple)), "expected_shape must be list or tuple!"
    assert len(expected_shape) == len(actual_shape), (
        f"Expected shape length {len(expected_shape)}, got {len(actual_shape)}: {actual_shape}"
    )
    for idx, (a, e) in enumerate(zip(actual_shape, expected_shape)):
        if e <= 0:
            continue
        assert a == e, f"At dim {idx}: expected {e}, got {a} (shape: {actual_shape})"

def compare_tensors(tensor1, tensor2, tol=1e-5, prefix=""):
    abs_diff = (tensor1 - tensor2).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    is_close = torch.allclose(tensor1, tensor2, rtol=tol, atol=tol)
    rank = dist.get_rank() if dist.is_initialized() else 0
    prefix = f"[{prefix}]" if prefix else ""
    print(f"{prefix}[Rank {rank}] Tensors match: {is_close} | Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}", flush=True)
