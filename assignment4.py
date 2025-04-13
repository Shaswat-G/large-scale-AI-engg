# relevant imports

from utils import init_distributed, create_batch, check, compare_tensors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist


rank, local_rank, world_size = init_distributed()

# Define global parameters
global_batch_size = 128  # Must be divisible by world size
local_batch_size = global_batch_size // world_size
input_dim = 64
output_dim = 32
seed = 42


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
    """
    Educational example of performing a single gradient step
    """
    
    # set seed
    torch.manual_seed(seed)
    
    # Generate a weight matrix
    initial_weight = torch.randn(input_dim, output_dim)
    
    # create custom linera model
    model = CustomLinearLayer(initial_weight).to(device)
    
    # Set up SGD optimizer with lr 0.5
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    # Create Loss function
    loss_fn = nn.MSELoss(reduction="mean")
    
    # create a synthetic batch of data with gloabl_batch_size
    inputs, targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
    check(inputs, [global_batch_size, input_dim])
    check(targets, [global_batch_size, output_dim])
    
    # Forward pass
    outputs = model(inputs)
    check(outputs, [global_batch_size, output_dim])
    
    # Compute MSE Loss
    loss = loss_fn(outputs, targets)
    check(loss, [])
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Compute gradients
    loss.backward()
    
    # Parameter update
    optimizer.step()
    
    # return updated weights detached from the graph
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
    # Rank 0 does the single-step baseline:
    print(f"[Rank {rank}] Compute the updated matrix which should be different from the initial weight matrix.")
    initial_weight, updated_weight = single_step()
    compare_tensors(initial_weight, updated_weight.cpu())
else:
    # On all other ranks, create a placeholder for the final weight
    updated_weight = torch.zeros(input_dim, output_dim, device="cuda")

# Distribute updated_weight to all ranks so they can compare with single-step approach
dist.broadcast(updated_weight, src=0)

if rank == 0:
    print(f"[Rank {rank}] Compute the updated weight using batch accumulation. They should match.")
    batch_accum_weight = single_step_with_grad_accumulation()
    # Compare with single-step approach
    compare_tensors(updated_weight.cpu(), batch_accum_weight.cpu())

if rank == 0:
    print(f"[Rank {rank}] Compute the updated weight using data parallelism.")
data_parallel_weight = data_parallel_single_step()

# Compare on all ranks to the baseline `updated_weight` from single_step
compare_tensors(updated_weight.cpu(), data_parallel_weight.cpu(), prefix="DataParallel")


dist.destroy_process_group()
print(f"[Rank {rank}] done")
