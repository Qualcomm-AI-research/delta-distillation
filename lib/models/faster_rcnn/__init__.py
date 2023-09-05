# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
from .builder import build_model
from .faster_rcnn import VideoFasterRCNN

__all__ = [
    'build_model',
    'VideoFasterRCNN',
]
