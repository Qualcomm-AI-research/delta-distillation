# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
from torch import nn

import lib.models.ddrnet.common as ddrnet_modules
import lib.models.hrnet.seg_hrnet as hrnet_modules
from lib.utils.dist import is_distributed

from .ddrnet.DDRNet_23 import DualResNet23
from .ddrnet.DDRNet_23_slim import DualResNet23s
from .ddrnet.DDRNet_39 import DualResNet39
from .delta_distillation.convertor import convert_convs, init_weights_convs
from .hrnet.seg_hrnet import HighResolutionNet


def get_seg_model(cfg, stage):
    """
    Instantiates a semantic segmentation model, transforms its convolutions into Delta Distillation modules and
    initializes every module.
    :param cfg: A configuration object for the experiment
    :param stage: Stage for which the model will be employed (`train` or `test`).
    :return:
    The semantic segmentation model.
    """
    if is_distributed():
        ddrnet_modules.BatchNorm2d = hrnet_modules.BatchNorm2d = nn.SyncBatchNorm
    else:
        ddrnet_modules.BatchNorm2d = hrnet_modules.BatchNorm2d = nn.BatchNorm2d

    if cfg.MODEL.type == "hrnet-w18-small":
        model = HighResolutionNet(cfg.serialize())
    elif cfg.MODEL.type == "ddrnet-23s":
        model = DualResNet23s()
    elif cfg.MODEL.type == "ddrnet-23":
        model = DualResNet23()
    elif cfg.MODEL.type == "ddrnet-39":
        model = DualResNet39()
    else:
        raise NotImplementedError

    conv_blocks = [
        layer for layer in list(model.named_modules()) if isinstance(layer[1], nn.Conv2d)
    ]
    conv_blocks = conv_blocks[1:-1]  # exclude the first and last conv

    model = convert_convs(model, cfg, conv_blocks)
    model = init_weights_convs(model, cfg, deploy=stage == "test")

    return model
