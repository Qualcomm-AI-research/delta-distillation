# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from mmdet.models.builder import DETECTORS as MMDET_DETECTORS
from mmdet.models.builder import HEADS as MMDET_HEADS

MODELS = Registry('models', parent=MMCV_MODELS)
DETECTORS = MMDET_DETECTORS
HEADS = MMDET_HEADS


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is None and test_cfg is None:
        return MODELS.build(cfg)
    else:
        return MODELS.build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)
