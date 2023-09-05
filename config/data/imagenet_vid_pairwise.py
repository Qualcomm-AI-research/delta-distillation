# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is adapted from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------
_base_ = ['./imagenet_vid_base.py']

from config.data.imagenet_vid_base import data_root
from config.data.imagenet_vid_base import dataset_type
from config.data.imagenet_vid_base import test_pipeline
from config.data.imagenet_vid_base import train_pipeline

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,

    type='pairwise',
    num_frames=2,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'Annotations/imagenet_vid_train.json',
            img_prefix=data_root + 'Data/VID',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=9,
                filter_key_img=False,
                method='uniform'),
            pipeline=train_pipeline),

        # you may not to use image data for fine-tuning as it is a video model
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + 'Annotations/imagenet_det_30plus1cls.json',
            img_prefix=data_root + 'Data/DET',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=0,
                filter_key_img=False,
                method='uniform'),
            pipeline=train_pipeline)
    ],

    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=None,
        pipeline=test_pipeline,
        test_mode=True),

    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'Data/VID',
        ref_img_sampler=None,
        pipeline=test_pipeline,
        test_mode=True)
)
