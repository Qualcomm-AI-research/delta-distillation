# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
#
# This code is from https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import torch
import torch.distributed as torch_dist
from torch.utils.data.distributed import DistributedSampler


def is_distributed():
    return torch_dist.is_initialized()


def get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()


def reduce_tensor(inp):
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def get_sampler(dataset):
    if is_distributed():
        return DistributedSampler(dataset)
    else:
        return None


def get_rank():
    if not torch_dist.is_initialized():
        return 0
    return torch_dist.get_rank()
