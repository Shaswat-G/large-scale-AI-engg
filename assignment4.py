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
    check(loss, [1])
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Compute gradients
    loss.backward()
    
    # Parameter update
    optimizer.step()
    
    # return updated weights detached from the graph
    return initial_weight, model.W.detach()    
    
    
    
if rank == 0:
    # Rank 0 does the single-step baseline:
    print(f"[Rank {rank}] Compute the updated matrix which should be different from the initial weight matrix.")
    initial_weight, updated_weight = single_step()
    compare_tensors(initial_weight, updated_weight.cpu())
else:
    # On all other ranks, create a placeholder for the final weight
    updated_weight = torch.zeros(input_dim, output_dim, device="cuda")

# Distribute updated_weight to all ranks so they can compare to the baseline
dist.broadcast(updated_weight, src=0)


# Cleanup
dist.destroy_process_group()
print(f"[Rank {rank}] done")
