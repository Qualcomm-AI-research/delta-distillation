# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import os
import warnings

import torch
from torch import nn

from lib.models.delta_distillation.base_ga_module import GAModule


def set_module_attr(model, layer_name, value):
    """
    Operates surgery on a model, replacing the module at `layer_name` with a new one.
    :param model: The original model.
    :param layer_name: A string specifiying which module should be replaced.
    :param value: The new module that is plugged inplace.
    """
    split = layer_name.split(".")

    this_module = model
    for mod_name in split[:-1]:
        if mod_name.isdigit():
            this_module = this_module[int(mod_name)]
        else:
            this_module = getattr(this_module, mod_name)

    last_mod_name = split[-1]
    if last_mod_name.isdigit():
        this_module[int(last_mod_name)] = value
    else:
        setattr(this_module, last_mod_name, value)


def set_module_attr_efficientdet(model, layer_name, value):
    """
    Operates surgery on a model, replacing the module at `layer_name` with a new one.
    Specific instantiation for EfficientDet.
    :param model: The original model.
    :param layer_name: A string specifiying which module should be replaced.
    :param value: The new module that is plugged inplace.
    """
    split = layer_name.split(".")

    this_module = model
    for mod_name in split[:-1]:
        this_module = getattr(this_module, mod_name)

    last_mod_name = split[-1]
    setattr(this_module, last_mod_name, value)


def load_state_dict(cfg):
    """
    Loads a state dictionary from disk.
    :param cfg: A configuration object for the experiment.
    :return:
    A state dictionary.
    """
    pretrained = cfg.checkpoint.init
    map_location = {f"cuda:{0}": f"cuda:{cfg.local_rank}"} if cfg.local_rank != -1 else None
    if os.path.isfile(pretrained):
        state_dict = torch.load(pretrained, map_location=map_location)
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = None
        warnings.warn(f"checkpoint does not exist at {pretrained}")

    return state_dict


def init_weights_normal(model, g_std=0.01):
    """
    Initializes the weights of the model by sampling from normal distributions.
    :param model: The model to be initialized.
    :param g_std: The standard deviation of student parameters.
    :return:
    The initialized model.
    """
    print("=> init weights from normal distribution")
    for m_name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if "g_" in m_name:  # init gradient apprx convs to almost zero
                nn.init.normal_(m.weight, std=g_std)
            else:
                nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return model


def set_delta_distillation_schedule(model, T):
    """
    Sets the Delta Distillation period T for every distilled layer in a model.
    :param model: The delta distillation model.
    :param T: The number of frames before reinstantiating a new keyframe.
    """
    _ = [m.set_T(T) for m in model.modules() if isinstance(m, GAModule)]
