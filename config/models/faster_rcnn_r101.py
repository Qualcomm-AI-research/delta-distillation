# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
_base_ = [
    "./mmdetection/faster_rcnn_r50_dc5.py",
]

model = dict(
    type="VideoFasterRCNN",
    detector=dict(
        backbone=dict(
            depth=101, init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101")
        ),
        train_cfg=dict(rpn_proposal=dict(max_per_img=1000), rcnn=dict(sampler=dict(num=512))),
    ),
    train_cfg=None,
    test_cfg=dict(key_frame_interval=10),
)

MODEL = dict(
    grad_block_type="none",
    factor_ch=1,
    factor_wh=1,
)
