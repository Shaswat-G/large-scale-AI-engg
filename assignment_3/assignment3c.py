import os
import time
import torch
import torch.distributed as dist

def main():
    # 1. Parse environment variables set by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # 2. Initialize the global process group
    dist.init_process_group(backend="nccl")
    
    # 3. Limit GPU allocation to one GPU per process
    torch.cuda.set_device(local_rank)
    
    # 4. Create parameters and gradients
    N = 2 ** 30 # ~1.07 billion elements, ~4GB float32
    parameters = torch.ones((N,), dtype = torch.float32, device="cuda")
    gradients = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    
    # sanity check
    print(f"[Python] rank={rank} | Initial paramters[0] = {parameters[0].item()} | Initial gradients[0] = {gradients[0].item()}")
    
    # 5. Define root rank and learning rate
    ROOT_RANK = 0
    LEARNING_RATE = 0.1
    
    # 6. Warmup operations and reinitialize parameters
    for _ in range(5):
        dist.all_reduce(parameters, op=dist.ReduceOp.SUM)
    parameters = torch.ones((N,), dtype=torch.float32, device="cuda")
    
    # 7. Measure time for reduce + broadcast
    torch.cuda.synchronize()
    start = time.time()
    
    #8. Reduce all gradients to the root rank
    dist.reduce(gradients, dst=ROOT_RANK, op=dist.ReduceOp.SUM)
    
    # 9. On root rank, compute average gradient and update parameters
    if rank == ROOT_RANK:
        gradients /= world_size
        parameters -= LEARNING_RATE * gradients
        
    # 10. Broadcast updated parameters to all ranks
    dist.broadcast(parameters, src=ROOT_RANK)
    torch.cuda.synchronize()
    end = time.time()
    elapsed_seconds = end - start
    
    # 11. Verify correctness of the parameter value on *every* rank
    #     We said: initial = 1.0, minus LEARNING_RATE * average(grad)
    #     The average(grad) if we sum ranks 0..(world_size-1) is 
    #     ( (world_size-1)*world_size / 2 ) / world_size = (world_size-1)/2
    #     => final param = 1.0 - LEARNING_RATE * ( (world_size - 1)/2 )
    expected_param = 1.0 - LEARNING_RATE * ((world_size - 1) / 2)
    assert torch.allclose(
        parameters[0],
        torch.tensor(expected_param, device="cuda")
    ), f"[Python] rank={rank} | Parameter mismatch: expected {expected_param}, got {parameters[0].item()}"
    
    
        # 12. Calculate and print throughput or time
    #     We want to measure how big was the data transfer. 
    #     - The reduce operation sends data from every rank to root
    #     - The broadcast operation sends data from root to every rank
    # For a rough estimate, let's just consider 2*N float32 elements crossing the network.
    
    total_bytes = parameters.nelement() * 4  # single pass of N
    # In a "reduce" of size N, the data from all ranks is conceptually shipped to root, but 
    # the physical data movement can be more complicated. We'll do a naive approach:
    total_bytes_transferred = total_bytes + total_bytes  # reduce + broadcast
    total_gbs = total_bytes_transferred / (1024 ** 3)
    throughput = total_gbs / elapsed_seconds

    print(f"[Python] rank={rank} | elapsed={elapsed_seconds:.4f}s | throughput={throughput:.4f}GB/s")

    # 13. Cleanup
    dist.destroy_process_group()
    

if __name__ == "__main__":
    main()