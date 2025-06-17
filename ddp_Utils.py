"""Distributed Data Parallel utilities for multi-GPU training."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS
import diffdist.functional as DF


def all_gather(tensors):
    """Gather tensors from all processes and concatenate."""
    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_gather2(tensors):
    """Gather tensors from all processes without concatenation."""
    gather_list = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    return gather_list


def all_gather_grad(tensors):
    """Gather tensors with gradient support."""
    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        DF.all_gather(tensor_placeholder, tensor)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_gather_grad2(tensors):
    """Gather tensors with gradient support, no concatenation."""
    gather_list = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        DF.all_gather(tensor_placeholder, tensor)
        gather_list.append(tensor_placeholder)
    return gather_list


def all_reduce(tensors, average=True):
    """All-reduce tensors across processes."""
    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def setup(rank, world_size, port='12345'):
    """Initialize distributed training process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    
def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()
    