# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is adapted from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------
_base_ = [
    './mmdetection/base.py',
]

model = dict(detector=dict(backbone=dict(frozen_stages=0, )))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0001 / 2,
    lr_g=0.1,
    momentum=0.9,
    weight_decay=0.0001,
)
optimizer_config = dict(grad_clip=dict(max_norm=3., norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup=None,
    step=[4])

workflow = [('train', 1)]
total_epochs = 7

loss = dict(
    alpha=1000.,
    beta=10.,
)

custom_hooks = [
    dict(type='LossWeightsSchedulingHook', at_epoch=2, new_lr=None, new_lr_g=None, new_alpha=0.),
]

# evaluation setting
run_validation = True
evaluation = dict(metric=['bbox'], interval=1)

stage = 'train'