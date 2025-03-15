import os
import torch
import math
from torch import distributed as dist

def dist_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])


    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])


    group_rank = int(os.environ["GROUP_RANK"])
    group_world_size = int(os.environ["GROUP_WORLD_SIZE"])
    return local_rank, local_world_size, rank, world_size, group_rank, group_world_size

def dist_info1():

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])


    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    return local_rank, local_world_size, rank, world_size, rank // world_size, world_size // local_world_size

 
def new_infer_pg(rank, local_infer_world_size, local_world_size, world_size, backend="nccl"):
    group_world_size = world_size // local_world_size
    group_rank = rank // local_world_size
    infer_pgs = []
    
    infer_pg_world_size = int(math.ceil(local_world_size / local_infer_world_size))
    assert infer_pg_world_size > 1, "infer_pg_world_size must be greater than 1"
    if local_world_size % local_infer_world_size != 0:
        print(f"WARN: local_world_size({local_world_size}) not be divisible by local_infer_world_size({local_infer_world_size}), infer_pg_world_size is {infer_pg_world_size}")
    # temp = (rank % local_world_size) // infer_pg_world_size
    current_infer_pg = None
    current_infer_pg_ranks = None
    for group_rank in range(group_world_size):
        ranks = [x for x in range(group_rank * local_world_size, group_rank * local_world_size + local_world_size)]
        for x in range(local_infer_world_size):
            infer_pg_ranks = ranks[x * infer_pg_world_size : (x + 1) * infer_pg_world_size]
            assert len(infer_pg_ranks) > 1, "infer_pg_world_size must be greater than 1"
            # print(f"****** rank: {rank}, infer_pg_ranks: {infer_pg_ranks}")
            infer_pg = dist.new_group(ranks=infer_pg_ranks, backend=backend)
            infer_pgs.append(infer_pg)
            if rank in infer_pg_ranks:
                current_infer_pg = infer_pg
                current_infer_pg_ranks = infer_pg_ranks
    # infer_pg = None
    infer_rank = current_infer_pg_ranks[0]
    is_infer_rank = rank == infer_rank
    # print(f"rank: {rank}, infer_pg_ranks: {current_infer_pg_ranks}, infer_rank: {infer_rank}, is_infer_rank: {is_infer_rank}")
    return current_infer_pg, infer_rank, is_infer_rank, infer_pg_world_size, infer_pgs


def new_train_pg(rank, local_infer_world_size, local_world_size, world_size, backend="nccl"):
    group_world_size = world_size // local_world_size
    ranks = [x for x in range(world_size)]
    infer_pg_world_size = int(math.ceil(local_world_size / local_infer_world_size))
    assert infer_pg_world_size > 1, "infer_pg_world_size must be greater than 1"
    real_local_infer_world_size = int(math.ceil(local_world_size / local_infer_world_size))
    if local_world_size % local_infer_world_size != 0:
        print(f"WARN: local_world_size not be divisible by local_infer_world_size, real local_infer_world_size is {real_local_infer_world_size}")
    infer_ranks = []
    for group_rank in range(group_world_size):
        for i in range(local_infer_world_size):
            local_infer_rank = min(infer_pg_world_size * i, local_world_size - 1)
            infer_ranks.append(group_rank * local_world_size + local_infer_rank)
    train_ranks = list(set(ranks) - set(infer_ranks))
    train_pg = dist.new_group(ranks=train_ranks, backend=backend)
    # print(f"train_ranks: {train_ranks}")
    return train_pg, real_local_infer_world_size


def send_to_infer_device(data, infer_pg, is_infer_rank, infer_rank):
    gather_list = None
    tmp = None
    org_data_shape0 = None

    # print(f"infer_pg: {dist.get_process_group_ranks(infer_pg)}, is_infer_rank: {is_infer_rank}, infer_rank: {infer_rank}")

    if is_infer_rank:
        infer_world_size = dist.get_world_size(infer_pg)
        shape = list(data.shape)
        org_data_shape0 = shape[0]
        shape[0] = org_data_shape0 * infer_world_size
        # print(f"rank: {infer_rank}, send_to_infer_device data.device: {data.device}")
        tmp = torch.empty(size=shape, dtype=data.dtype, device=data.device)
        # gather_list = list(tmp.chunk(infer_world_size, dim=0))
        gather_list = [tmp for tmp in tmp.chunk(infer_world_size, dim=0)]

        # print(f"gather_list: {type(gather_list)}, tensor: {type(gather_list[0])}")
    # print(f"****** infer_pg: {dist.get_process_group_ranks(infer_pg)}, is_infer_rank: {is_infer_rank}, infer_rank: {infer_rank}, rank: {dist.get_rank()}")
    dist.gather(tensor=data, gather_list=gather_list, dst=infer_rank, group=infer_pg)
    # print(f"****** success gather, infer_pg: {dist.get_process_group_ranks(infer_pg)}, is_infer_rank: {is_infer_rank}, infer_rank: {infer_rank}, rank: {dist.get_rank()}")
    if tmp is not None:
        tmp = tmp[1*org_data_shape0 : ]
    # print(f"rank: {rank}, tmp: {tmp}")
    return tmp


def receive_from_infer_device(data, infer_pg, is_infer_rank, infer_rank):
    scatter_list = None
    tmp = None
    real_data = data

    if is_infer_rank:
        infer_world_size = dist.get_world_size(infer_pg)
        scatter_list = [tmp for tmp in data.chunk(infer_world_size - 1, dim=0)]
        tmp = scatter_list[0]
        # print(f"rank: {infer_rank}, receive_from_infer_device tmp.device: {tmp.device}")
        real_data = torch.empty(size=tmp.shape, dtype=tmp.dtype, device=tmp.device)
        # print(f"real_data = {real_data}")
        scatter_list = [real_data] + scatter_list
    dist.scatter(tensor=real_data, scatter_list=scatter_list, src=infer_rank, group=infer_pg)
    # print(f"rank: {rank}, real_data: {real_data}")
    return real_data

if __name__ == '__main__':
    local_infer_world_size = 1
    
    torch.distributed.init_process_group(backend="nccl")
    
    local_rank, local_world_size, rank, world_size, group_rank, group_world_size = dist_info()
    torch.cuda.set_device(local_rank)
    
    infer_pg, infer_rank, is_infer_rank, real_local_infer_world_size, _ = new_infer_pg(rank, local_infer_world_size, local_world_size, world_size)
    train_pg, real_local_infer_world_size = new_train_pg(rank, local_infer_world_size, local_world_size, world_size)
    
    data = torch.ones(size=(2,2), dtype=torch.bfloat16, device="cuda") * rank
    ret = send_to_infer_device(data, infer_pg, is_infer_rank, infer_rank)
    print(f"rank: {rank}, send_ret: {ret}")
    
    data = ret if is_infer_rank else torch.empty(size=(2,2), dtype=torch.bfloat16, device="cuda")
    ret = receive_from_infer_device(data, infer_pg, is_infer_rank, infer_rank)
    print(f"rank: {rank}, receive_ret: {ret}")
    
    dist.destroy_process_group(infer_pg)
    dist.destroy_process_group(train_pg)



