# ------------------------------------------------------------------------------
# Copyright 2020 Ross Wightman
# Licensed under the Apache License 2.0.
#
# This modified starting from https://github.com/rwightman/efficientdet-pytorch
# ------------------------------------------------------------------------------
import itertools
import pickle

import numpy as np
import torch


def clip_collate(batch, max_num_instances=100):
    """
    Collate function for videos.
    :param batch: Batch of video clips
    :param max_num_instances: maximum number of detection instances per frame.
    :return:
    A tuple like (image_tensor, detection targets).
    """
    # merge the time into batch dimension
    frames = np.concatenate([_[0] for _ in batch])
    annotations = list(itertools.chain(*[pickle.loads(_[1]) for _ in batch]))
    batch = list(zip(frames, annotations))

    batch_size = len(batch)

    target = dict()
    for k, v in batch[0][1].items():
        if isinstance(v, np.ndarray):
            # if a numpy array, assume it relates to object instances, pad to MAX_NUM_INSTANCES
            target_shape = (batch_size, max_num_instances)
            if len(v.shape) > 1:
                target_shape = target_shape + v.shape[1:]
            target_dtype = torch.float32
        elif isinstance(v, (tuple, list)):
            # if tuple or list, assume per batch
            target_shape = (batch_size, len(v))
            target_dtype = torch.float32 if isinstance(v[0], float) else torch.int32
        else:
            # scalar, assume per batch
            target_shape = (batch_size,)
            target_dtype = torch.float32 if isinstance(v, float) else torch.int64
        target[k] = torch.zeros(target_shape, dtype=target_dtype)

    tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
    for i in range(batch_size):
        tensor[i] += torch.from_numpy(batch[i][0])
        for tk, tv in batch[i][1].items():
            if isinstance(tv, np.ndarray) and len(tv.shape):
                target[tk][i, 0 : tv.shape[0]] = torch.from_numpy(tv)
            else:
                target[tk][i] = torch.tensor(tv, dtype=target[tk].dtype)

    return tensor, target
