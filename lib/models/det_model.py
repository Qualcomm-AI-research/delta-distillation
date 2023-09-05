# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
from torch import nn

from lib.models.delta_distillation.convertor import convert_convs, init_weights_convs
from lib.models.efficientdet.effdet import (
    DetBenchPredict,
    DetBenchTrain,
    EfficientDet,
    get_efficientdet_config,
)
from lib.models.efficientdet.effdet.efficientdet import HeadNet
from lib.models.faster_rcnn import build_model


def get_efficientdet(cfg, stage):
    """
    Instantiates an EfficientDet-D0 model, transforms its convolutions into Delta Distillation modules and
    initializes every module.
    :param cfg: A configuration object for the experiment
    :param stage: Stage for which the model will be employed (`train` or `test`).
    :return:
    The EfficientDet-D0 Delta Distillation model.
    """
    config = get_efficientdet_config(cfg.model.type)
    net = EfficientDet(config)

    config.num_classes = cfg.data.num_classes
    net.class_net = HeadNet(
        config, num_outputs=config.num_classes, norm_kwargs=dict(eps=0.001, momentum=0.01)
    )

    if stage == "train":
        model = DetBenchTrain(net, config)
    else:
        model = DetBenchPredict(net, config)

    # disable random drop path to prevent mismatch between dz_ref and dz_pred
    for module in model.modules():
        if hasattr(module, "drop_path_rate"):
            module.drop_path_rate = 0.0

    conv_blocks = [
        layer for layer in list(model.named_modules()) if isinstance(layer[1], nn.Conv2d)
    ]
    conv_blocks = conv_blocks[1:-2]  # exclude the first (3 channel) and final prediction convs
    conv_blocks = [_ for _ in conv_blocks if ".se." not in _[0]]
    conv_blocks = [_ for _ in conv_blocks if _[1].groups != _[1].in_channels]

    model = convert_convs(model, cfg, conv_blocks)
    model.model = init_weights_convs(model.model, cfg, convert=stage == "train")

    return model


def get_faster_rcnn(cfg, stage):
    """
    Instantiates an FasterRCNN model, transforms its convolutions into Delta Distillation modules and
    initializes every module.
    :param cfg: A configuration object for the experiment
    :param stage: Stage for which the model will be employed (`train` or `test`).
    :return:
    The FasterRCNN Delta Distillation model.
    """
    model = build_model(cfg.model)
    model.init_weights()

    if stage == "train":
        model.train_cfg = dict(
            delta_distillation=cfg.delta_distillation, loss=cfg.loss, checkpoint=cfg.checkpoint
        )

    conv_blocks = [
        layer for layer in list(model.named_modules()) if isinstance(layer[1], nn.Conv2d)
    ]
    conv_blocks = conv_blocks[1:-2]  # exclude the first (3 channel) and final prediction convs

    model = convert_convs(model, cfg, conv_blocks)
    model = init_weights_convs(model, cfg, convert=stage == "train")

    return model


def get_det_model(model_name, cfg, stage):
    """
    Instantiates an object detection model.
    :param model_name: Specifies which models to instantiate (`efficientdet` or `faster_rcnn`).
    :param cfg: A configuration object for the experiment
    :param stage: Stage for which the model will be employed (`train` or `test`).
    :return:
    An object detection Delta Distillation model.
    """
    if model_name == "efficientdet":
        model = get_efficientdet(cfg, stage)
    elif model_name == "faster_rcnn":
        model = get_faster_rcnn(cfg, stage)
    else:
        raise ValueError(f"Detection model {model_name} is unknown.")

    return model
