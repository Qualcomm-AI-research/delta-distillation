# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is adapted from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------
# gpu setting
seed = 304
deterministic = False
local_rank = -1
cuda_devices = 'all'
dist_params = dict(backend='nccl')
find_unused_parameters = True

# logger setting
log_name = 'log/train/detection/faster_rcnn'
log_level = 'INFO'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
log_activations = True

# checkpoint setting
load_from = None
resume_from = None
checkpoint = dict(init=None, mac=None)
checkpoint_config = dict(interval=1)
