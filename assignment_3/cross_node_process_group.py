#!/usr/bin/env python3
import os
import time
import socket
import torch
import torch.distributed as dist

def main():

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"]) 

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    cross_groups = [
        dist.new_group([0,4,8,12]),
        dist.new_group([1,5,9,13]),
        dist.new_group([2,6,10,14]),
        dist.new_group([3,7,11,15]),
    ]

    cross_group_index = rank % 4
    cross_pg = cross_groups[cross_group_index]

    # Warm Up
    exponent = 30
    N = 2 ** exponent
    tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=cross_pg)


    # Re-initialize after warmup
    tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

    for exponent in range(12, 31, 2):
        
        N = 2 ** exponent
        tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
        
        torch.cuda.synchronize()
        start = time.time()
        dist.all_reduce(tensor,  op=dist.ReduceOp.SUM, group=cross_pg)
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed_seconds = end - start
        total_bytes = tensor.nelement() * 4
        total_gbs = total_bytes / (1024 ** 3)  # convert to GB
        throughput = total_gbs / elapsed_seconds

        # print(f"[Python] rank={rank} | transferred {total_gbs:.2f}GB | elapsed={elapsed_seconds:.4f}s | throughput={throughput:.4f}GB/s")
        print(f"[Python] exponent={exponent} | rank={rank} | transferred {total_gbs:.2f}GB | elapsed={elapsed_seconds:.4f}s | throughput={throughput:.4f}GB/s")
        

    dist.destroy_process_group()
    
    
if __name__ == "__main__":
    main()