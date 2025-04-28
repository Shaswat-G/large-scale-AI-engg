#!/usr/bin/env python3
import os
import time
import socket
import torch
import torch.distributed as dist

# 1. Parse environment variables set by torchrun
rank       = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])  # torchrun should set this as well

# 2. Initialize the default (global) process group
dist.init_process_group(backend="nccl")

# 3. Limit GPU allocation of this process to only one GPU
torch.cuda.set_device(local_rank)

# 4. Create a large tensor on each rank directly on the GPU
N = 2 ** 30  # ~1.07 billion elements (roughly 4GB if float32)
tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

# 5. (Optional) Warmup runs to avoid overhead from e.g. connection set-up
for _ in range(5):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # We won't time the warmups; they just ensure the next steps measure raw throughput.


# Re-initialize after warmups
tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

# 6. Force a CUDA sync, measure start time
torch.cuda.synchronize()
start = time.time()

# Perform the actual all-reduce we care about
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Another sync and measure end time
torch.cuda.synchronize()
end = time.time()
elapsed_seconds = end - start

# 7. Verify correctness
#    Since each rank's tensor originally held "rank", the sum is sum(0..world_size-1) = world_size*(world_size-1)/2
expected_val = world_size * (world_size - 1) / 2
# The first element of "tensor" should be "expected_val"
# The entire tensor should also be that same repeated value:
assert torch.allclose(
    tensor,
    torch.full_like(tensor, expected_val)
), f"[Python] rank={rank} | all-Reduce mismatch: expected {expected_val}, got {tensor[0].item()} in first element."

# 8. Calculate throughput (GB/s)
#    Each float32 is 4 bytes, so total data = N * 4 bytes
total_bytes = tensor.nelement() * 4
total_gbs = total_bytes / (1024 ** 3)  # convert to GB
throughput = total_gbs / elapsed_seconds

print(f"[Python] rank={rank} | transferred {total_gbs:.2f}GB | "
      f"elapsed={elapsed_seconds:.4f}s | throughput={throughput:.4f}GB/s")

# 9. Cleanup
dist.destroy_process_group()