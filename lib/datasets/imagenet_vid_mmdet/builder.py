# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmdet.datasets.samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedVideoSampler(_DistributedSampler):
    """Put videos to multi gpus during testing.

    Args:
        dataset (Dataset): Test dataset that must has `data_infos` attribute.
            Each data_info in `data_infos` record information of one frame,
            and each video must has one data_info that includes
            `data_info['frame_id'] == 0`.
        num_replicas (int): The number of gpus. Defaults to None.
        rank (int): Gpu rank id. Defaults to None.
        shuffle (bool): If True, shuffle the dataset. Defaults to False.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        assert not self.shuffle, 'Specific for video sequential testing.'
        self.num_samples = len(dataset)

        first_frame_indices = []
        for i, img_info in enumerate(self.dataset.data_infos):
            if img_info['frame_id'] == 0:
                first_frame_indices.append(i)

        if len(first_frame_indices) < num_replicas:
            raise ValueError(f'only {len(first_frame_indices)} videos loaded,'
                             f'but {self.num_replicas} gpus were given.')

        chunks = np.array_split(first_frame_indices, self.num_replicas)
        split_flags = [c[0] for c in chunks]
        split_flags.append(self.num_samples)

        self.indices = [
            list(range(split_flags[i], split_flags[i + 1]))
            for i in range(self.num_replicas)
        ]

    def __iter__(self):
        """Put videos to specify gpu."""
        indices = self.indices[self.rank]
        return iter(indices)


def reduce_dataset_train(dataset, reduction_rate=1.):
    if reduction_rate == 1:
        return

    for d in dataset.datasets:
        data_infos = d.data_infos
        data_infos = data_infos[0:len(data_infos) // reduction_rate]
        d.data_infos = data_infos
    dataset.cumulative_sizes = list(np.cumsum([len(d) for d in dataset.datasets]))
    dataset.flag = dataset.flag[0:len(dataset.flag) // reduction_rate]


def reduce_dataset_test(dataset, reduction_rate=1.):
    if reduction_rate == 1:
        return

    data_infos = dataset.data_infos
    data_infos = data_infos[0:len(data_infos) // reduction_rate]
    dataset.data_infos = data_infos
    dataset.img_ids = [_['id'] for _ in data_infos]


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     validation=False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if hasattr(dataset, 'datasets'):
        reduce_dataset_train(dataset, reduction_rate=1)
    else:
        reduce_dataset_test(dataset, reduction_rate=(20 if validation else 1))

    rank, world_size = get_dist_info()
    if dist:
        if shuffle:
            sampler = DistributedGroupSampler(dataset, samples_per_gpu,
                                              world_size, rank)
        else:
            if hasattr(dataset, 'load_as_video') and dataset.load_as_video:
                sampler = DistributedVideoSampler(
                    dataset, world_size, rank, shuffle=False)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
