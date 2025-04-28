# relevant imports

from utils import init_distributed, create_batch, check, compare_tensors

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed import ReduceOp


rank, local_rank, world_size = init_distributed()

# Define global parameters
global_batch_size = 128  # Must be divisible by world size
local_batch_size = global_batch_size // world_size
input_dim = 64
output_dim = 32
seed = 42

class BroadCastParallel(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        if world_size ==1:
            return grad_output
        dist.all_reduce(grad_output, op=ReduceOp.SUM)
        return grad_output
        
class GatherParallel(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        if world_size == 1:
            return x
        
        x = x.contiguous()
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        x_list[rank] = x  # place this rank's shard in x_list

        # Gather all shards
        dist.all_gather(x_list, x)
        out = torch.cat(x_list, dim=-1).contiguous()

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        if world_size == 1:
            return grad_output
        
        # Split the incoming big gradient across the last dimension
        local_dim = grad_output.shape[-1] // world_size
        grad_output_split = torch.split(grad_output, local_dim, dim=-1)

        # Return only this rank's slice
        return grad_output_split[rank].contiguous()
    
class FullColumnParallelLinear(nn.Module):
    
    def __init__(self, weight: torch.Tensor, world_size: int, rank: int):
        super(FullColumnParallelLinear, self).__init__()
        in_dim, out_dim = weight.shape
        assert out_dim % world_size == 0, "out_dim must be divisible by world_size"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.world_size = world_size
        self.rank = rank
        self.local_out_dim = out_dim // world_size
        
        start = rank * self.local_out_dim
        end   = start + self.local_out_dim

        # Copy the local shard. 
        # The shape is [in_dim, out_dim // world_size].
        self.W = nn.Parameter(weight[:, start:end].clone().contiguous())
        
    def forward(self, X):
        
        local_bsz = X.shape[0]
        check(X, [local_bsz, self.in_dim])
        
        X = BroadCastParallel.apply(X)
        
        local_out = torch.einsum("bi,ij->bj", X, self.W).contiguous()
        check(local_out, [local_bsz, self.local_out_dim])
        
        out = GatherParallel.apply(local_out)
        check(out, [local_bsz, self.out_dim])
        
        return out
    
    
def full_column_parallel_single_step(seed=42, device="cuda"):
    
    torch.manual_seed(seed)
    initial_weight = torch.randn(input_dim, output_dim)
    model = FullColumnParallelLinear(initial_weight, world_size, rank).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.MSELoss(reduction="mean")
    full_inputs, full_targets = create_batch(batch_size=global_batch_size, input_dim=input_dim, output_dim=output_dim, seed=seed, device=device)

    outputs = model(full_inputs)
    check(outputs, [global_batch_size, output_dim])
    
    loss = loss_fn(outputs, full_targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # --- Each rank has a local shard of shape [input_dim, out_dim // world_size] => Gather all shards
    local_updated_weight = model.W.detach()
    check(local_updated_weight, (input_dim, output_dim // world_size))
    
    weight_shards = [torch.zeros_like(local_updated_weight) for _ in range(world_size)]
    dist.all_gather(weight_shards, local_updated_weight)
    global_updated_weight = torch.cat(weight_shards, dim=1)
    check(global_updated_weight, (input_dim, output_dim))
    
    return global_updated_weight

class CustomLinearLayer(nn.Module):
    """
    A linear layer.
    weight matrix W has shape [in_dim, out_dim]
    activation matrix X has shape [bsz, in_dim]
    out = X @ W which as shape [bsz, out_dim]
    """
    
    def __init__(self, weight: torch.Tensor):
        super(CustomLinearLayer, self).__init__()
        self.W = nn.Parameter(weight)
        self.in_dim = weight.shape[0]
        self.out_dim = weight.shape[1]
        
        
    def forward(self, X):
        """
        Forward pass of the custom linear layer.
        :param X: Input tensor of shape [bsz, in_dim]
        :return: Output tensor of shape [bsz, out_dim]
        """
        local_bsz = X.shape[0]
        # Check that the batch size is correct
        check(X, [local_bsz, self.in_dim])
        
        # Batched matrix-vector multiplication
        X = torch.einsum("bi,ij->bj", X, self.W)
        
        check(X, [local_bsz, self.out_dim])
        return X
    
    
def single_step(seed = 42, device = "cuda") -> torch.Tensor:
    
    torch.manual_seed(seed)
    initial_weight = torch.randn(input_dim, output_dim)
    model = CustomLinearLayer(initial_weight).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.MSELoss(reduction="mean")
    
    inputs, targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
    check(inputs, [global_batch_size, input_dim])
    check(targets, [global_batch_size, output_dim])

    outputs = model(inputs)
    check(outputs, [global_batch_size, output_dim])
    
    loss = loss_fn(outputs, targets)
    check(loss, [])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return initial_weight, model.W.detach()

def single_step_with_grad_accumulation(seed=42, device="cuda", accumulation_steps=4):
    
    torch.manual_seed(seed)
    initial_weight
    model = CustomLinearLayer(initial_weight).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.MSELoss(reduction="mean")
    inputs, targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
    check(inputs, [global_batch_size, input_dim])
    check(targets, [global_batch_size, output_dim])
    
    micro_batch_size = global_batch_size // accumulation_steps
    
    optimizer.zero_grad()
    
    for i in range(accumulation_steps):
        
        start_idx = i * micro_batch_size
        end_idx = start_idx + micro_batch_size
        
        micro_inputs = inputs[start_idx:end_idx]
        micro_targets = targets[start_idx:end_idx]
        check(micro_inputs, [micro_batch_size, input_dim])
        check(micro_targets, [micro_batch_size, output_dim])
        
        micro_outputs = model(micro_inputs)
        micro_loss = loss_fn(micro_outputs, micro_targets)
        check(micro_loss, [])
        
        scaled_loss = micro_loss / accumulation_steps
        scaled_loss.backward()
    
    optimizer.step()
    
    return model.W.detach()


def data_parallel_single_step(seed=42, device="cuda"):
    
    torch.manual_seed(seed)
    initial_weight = torch.randn(input_dim, output_dim)
    model = CustomLinearLayer(initial_weight).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.MSELoss(reduction="mean")
    inputs, targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
    check(inputs, [global_batch_size, input_dim])
    check(targets, [global_batch_size, output_dim])
    
    full_inputs, full_targets = create_batch(batch_size=global_batch_size, input_dim=input_dim, output_dim=output_dim, seed=seed, device=device)
    
    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size
    
    local_inputs = full_inputs[start_idx:end_idx]
    local_targets = full_targets[start_idx:end_idx]
    
    check(local_inputs, [local_batch_size, input_dim])
    check(local_targets, [local_batch_size, output_dim])
    
    optimizer.zero_grad()
    local_outputs = model(local_inputs)
    check(local_outputs, [local_batch_size, output_dim])
    
    local_loss = loss_fn(local_outputs, local_targets)
    check(local_loss, [])
    
    local_loss.backward()
    
    # sync gradients across all ranks using all_reduce
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)
        
    optimizer.step()
    
    return model.W.detach()
    
    

    
if rank == 0:
    print(f"[Rank {rank}] Compute the updated matrix which should be different from the initial weight matrix.")
    initial_weight, updated_weight = single_step()
    compare_tensors(initial_weight, updated_weight.cpu())
else:
    updated_weight = torch.zeros(input_dim, output_dim, device="cuda")



dist.broadcast(updated_weight, src=0)



if rank == 0:
    print(f"[Rank {rank}] Compute the updated weight using batch accumulation. They should match.")
    batch_accum_weight = single_step_with_grad_accumulation()
    compare_tensors(updated_weight.cpu(), batch_accum_weight.cpu())

if rank == 0:
    print(f"[Rank {rank}] Compute the updated weight using data parallelism.")
data_parallel_weight = data_parallel_single_step()
compare_tensors(updated_weight.cpu(), data_parallel_weight.cpu(), prefix="DataParallel")


if rank == 0:
    print(f"[Rank {rank}] Compute the updated weight using tensor parallelism (FullColumnParallelLinear).")
column_parallel_weight = full_column_parallel_single_step()
compare_tensors(updated_weight.cpu(), column_parallel_weight.cpu(), prefix="FullColumnParallel")



dist.destroy_process_group()
print(f"[Rank {rank}] done")
