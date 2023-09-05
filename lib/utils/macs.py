# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import torch
from ptflops import flops_counter, get_model_complexity_info
from ptflops.flops_counter import flops_to_string
from timm.models.layers.activations_me import SwishMe
from timm.models.layers.conv2d_same import Conv2dSame
from timm.models.layers.pool2d_same import MaxPool2dSame
from torch.nn import UpsamplingNearest2d

from lib.utils.nn import set_delta_distillation_schedule

default_constructor = lambda size: {"x": torch.zeros(*size)}

faster_rcnn_constructor = lambda size: {
    "return_loss": False,
    "rescale": True,
    "img": torch.zeros(size),
    "img_metas": [
        [
            {
                "img_shape": (size[2:]) + (3,),
                "scale_factor": [1.0],
                "frame_id": 0,
            }
        ]
        * size[0]
    ],
}


def swishme_flops_counter_hook(module, input_tensor, output_tensor):
    flops_counter.relu_flops_counter_hook(module, input_tensor, output_tensor)
    flops_counter.relu_flops_counter_hook(module, input_tensor, output_tensor)


CUSTOM_MODULE_HOOKS = {
    Conv2dSame: flops_counter.conv_flops_counter_hook,
    SwishMe: swishme_flops_counter_hook,
    MaxPool2dSame: flops_counter.pool_flops_counter_hook,
    UpsamplingNearest2d: flops_counter.upsample_flops_counter_hook,
}


@torch.no_grad()
def compute_macs(model, clip_size, input_constructor=default_constructor):
    """
    Function computing the MAC count of a given model.
    :param model: The model to compute the cost for.
    :param clip_size: The shape of the clip for which the cost need to be computed (T,C,H,W).
    :param input_constructor: Function building a dummy input to the model, according to clip_size.
    :return:
    A float specifying the per-frame MACS of the model, amortized over the clip.
    """
    print(f"Computing macs for clip size {clip_size}")
    T, c, h, w = clip_size
    set_delta_distillation_schedule(model, T)

    macs, _ = get_model_complexity_info(
        model,
        input_res=clip_size,
        input_constructor=input_constructor,
        as_strings=False,
        print_per_layer_stat=False,
        custom_modules_hooks=CUSTOM_MODULE_HOOKS,
        ignore_modules=[torch.nn.BatchNorm2d],
    )

    amortized_macs = macs / T
    print(f"Cost per frame (amortized): {flops_to_string(amortized_macs, units=None)}")

    return amortized_macs
