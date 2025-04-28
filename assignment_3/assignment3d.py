import os
import time
import torch
import torch.distributed as dist

def main():
    # Initialize distributed
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    # Create send and receive tensors
    N = 2 ** 28 # ~256MB and ~1GB of float32
    send_tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    recv_tensor = torch.zeros((N,), dtype=torch.float32, device="cuda")
    
    # Implement ring communication topology
    send_rank = (rank+1) % world_size
    recv_rank = (rank-1) % world_size
    
    # Warm up
    WARMUP_ROUNDS = 5
    for _ in range(WARMUP_ROUNDS):
        
        if rank %2 == 0:
            dist.send(send_tensor, dst=send_rank)
            dist.recv(recv_tensor, src=recv_rank)
        else:
            dist.recv(recv_tensor, src=recv_rank)
            dist.send(send_tensor, dst=send_rank)
            
            
    torch.cuda.synchronize()
    
    start = time.time()

    if rank %2 == 0:
        dist.send(send_tensor, dst=send_rank)
        dist.recv(recv_tensor, src=recv_rank)
    else:
        dist.recv(recv_tensor, src=recv_rank)
        dist.send(send_tensor, dst=send_rank)
        
    torch.cuda.synchronize()
    end = time.time()
    elapsed_blocking = end - start
    
    expected_val = float(recv_rank)
    assert torch.allclose(recv_tensor, torch.full_like(recv_tensor, fill_value=expected_val)),  f"[Blocking ring] rank={rank} data mismatch: expected {expected_val}, got {recv_tensor[0].item()}"
                          
    total_bytes = recv_tensor.nelement()*4
    total_gbs = total_bytes / (1024**3)
    
    blocking_throughput = total_gbs / elapsed_blocking
    print(f"[Python] rank={rank} | Blocking: {elapsed_blocking:.4f}s, throughput={blocking_throughput:.4f}GB/s")
    
    # Async version
    send_tensor.fill_(rank)
    recv_tensor.fill_(0)
    
    # Warm up
    for _ in range(WARMUP_ROUNDS):
        send_req  = dist.isend(send_tensor, dst=send_rank)
        recv_req = dist.irecv(recv_tensor, src=recv_rank)
        send_req.wait()
        recv_req.wait()
        
    torch.cuda.synchronize()
    start = time.time()

    send_req = dist.isend(tensor=send_tensor, dst=send_rank)
    recv_req = dist.irecv(tensor=recv_tensor, src=recv_rank)
    send_req.wait()
    recv_req.wait()

    torch.cuda.synchronize()
    end = time.time()
    elapsed_async = end - start
    
    # Check correctness again
    assert torch.allclose(
        recv_tensor,
        torch.full_like(recv_tensor, float(recv_rank))
    ), f"[Async ring] rank={rank} data mismatch: expected {float(recv_rank)}, got {recv_tensor[0].item()}"

    async_throughput = total_gbs / elapsed_async
    print(f"[Python] rank={rank} | Async: {elapsed_async:.4f}s, throughput={async_throughput:.4f}GB/s")

    dist.destroy_process_group()
    

if __name__ == "__main__":
    main()