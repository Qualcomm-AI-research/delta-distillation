# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
_base_ = [
    "./mmdetection/base.py",
]

fuse_conv_bn = False

stage = "test"
