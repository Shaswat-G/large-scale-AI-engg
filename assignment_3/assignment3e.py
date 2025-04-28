import os
import time
import torch
import torch.distributed as dist


def main():

    # Initialize environment

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    N = 2 ** 30  # ~1.07 billion floats (~4GB if float32)
    tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    
    dist.barrier()
    test_process_groups(rank, world_size, tensor)
    dist.destroy_process_group()
    
def test_process_groups(rank, world_size, tensor):
    # Node based grouping
    node_groups = [
        dist.new_group([0,1,2,3]),
        dist.new_group([4,5,6,7]),
        dist.new_group([8,9,10,11]),
        dist.new_group([12,13,14,15]),
    ]
    
    group_index = rank // 4
    pg = node_groups[group_index]
    
    torch.cuda.synchronize()
    start = time.time()
    dist.all_reduce(tensor,  op=dist.ReduceOp.SUM, group=pg)
    torch.cuda.synchronize()
    end = time.time()
    elapsed = end - start
    
    group_sums = [6, 22, 38, 54]
    expected_val = float(group_sums[group_index])
    first_val = tensor[0].item()
    if not torch.allclose(tensor[0], torch.tensor(expected_val, device="cuda")):
        raise RuntimeError(f"[Node-based PG] rank={rank}, group={group_index} mismatch: expected {expected_val}, got {first_val}")
    
    
# 3. Throughput (rough measure)
    total_bytes = tensor.nelement() * 4  # N * 4 bytes
    total_gbs = total_bytes / (1024**3)
    throughput = total_gbs / elapsed
    
    print(f"[Node-based] rank={rank} | group_idx={group_index} | time={elapsed:.4f}s | throughput={throughput:.4f}GB/s")
    
    
    dist.barrier()
    
    # B. Less-natural grouping
    cross_groups = [
        dist.new_group([0,4,8,12]),
        dist.new_group([1,5,9,13]),
        dist.new_group([2,6,10,14]),
        dist.new_group([3,7,11,15]),
    ]

    # Re-init the tensor (otherwise it’s already summed)
    # or you can do a barrier first
    dist.barrier()
    tensor.fill_(float(rank))

    cross_group_index = rank % 4
    cross_pg = cross_groups[cross_group_index]

    torch.cuda.synchronize()
    start = time.time()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=cross_pg)

    torch.cuda.synchronize()
    end = time.time()
    elapsed_cross = end - start

    # Calculate expected sum for each cross group
    # e.g. group [0,4,8,12]: sum=0+4+8+12=24
    #      group [1,5,9,13]=28, [2,6,10,14]=32, [3,7,11,15]=36
    cross_sums = [24, 28, 32, 36]
    expected_val_2 = float(cross_sums[cross_group_index])
    first_val_2 = tensor[0].item()
    if not torch.allclose(tensor[0], torch.tensor(expected_val_2, device="cuda")):
        raise RuntimeError(f"[Cross PG] rank={rank}, group={cross_group_index} mismatch: "
                           f"expected {expected_val_2}, got {first_val_2}")

    # Throughput
    total_bytes = tensor.nelement() * 4
    total_gbs = total_bytes / (1024**3)
    throughput_cross = total_gbs / elapsed_cross
    
    print(f"[Cross-based] rank={rank} | group_idx={cross_group_index} | time={elapsed_cross:.4f}s | throughput={throughput_cross:.4f}GB/s")

if __name__ == "__main__":
    main()