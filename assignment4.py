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
ouput_dim = 32
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