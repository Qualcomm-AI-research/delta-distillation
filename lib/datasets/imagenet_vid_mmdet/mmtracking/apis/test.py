# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------
import os.path as osp
import shutil
import tempfile
from collections import defaultdict

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def collect_results_cpu(result_part, tmpdir=None):
    """Collect results on cpu mode.

    Saves the results on different gpus to 'tmpdir' and collects them by the
    rank 0 worker.

    Args:
        result_part (dict[list]): The part of prediction results.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. If is None, use `tempfile.mkdtemp()`
            to make a temporary path. Defaults to None.

    Returns:
        dict[str, list]: The prediction results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = defaultdict(list)
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_file = mmcv.load(part_file)
            for k, v in part_file.items():
                part_list[k].extend(v)
        shutil.rmtree(tmpdir)
        return part_list
