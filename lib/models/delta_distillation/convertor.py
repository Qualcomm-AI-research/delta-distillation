# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import copy
import warnings

import numpy as np
import torch

from lib.utils.nn import init_weights_normal, load_state_dict
from lib.utils.nn import set_module_attr_efficientdet as set_module_attr

from .conv import Conv


def convert_convs(model, cfg, conv_blocks):
    """
    Convert convolutions within a given model to Delta Distillation functions, implementing teacher and student
    functionalities.
    :param model: The model to be converted.
    :param cfg: A configuration object for the experiment.
    :param conv_blocks: The list of modules (nn.Conv2d) to be replaced by Delta Distillation counterparts.
    :return:
    The transformed model.
    """
    for layer_name, conv in conv_blocks:
        converted_block = Conv(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            cfg.delta_distillation.factor_ch,
            cfg.delta_distillation.channel_reduction_type,
            cfg.delta_distillation.gate_bias_init,
        )
        set_module_attr(model, layer_name, converted_block)

    return model


def init_weights_train(model, cfg, convert=True):
    """
    Initializes weights for a Delta Distillation model. It supports loading the checkpoint of a regular model and
    plugging its parameters to the teacher of the Delta Distillation model.
    The student is initialized via SVD decomposition of the checkpointed weights (see Fig.3).
    :param model: The model for which parameters need initialization.
    :param cfg: A configuration object for the experiment. Can contain the path of a checkpoint to load.
    :param convert: Flag identifying whether the model already in Delta Distillation mode. If so, the loaded checkpoint
    will be converted to fit the model state dictionary.
    :return:
    The initialized model.
    """
    model = init_weights_normal(model, g_std=0.001)
    state_dict = load_state_dict(cfg)
    model_dict = model.state_dict()

    if state_dict is None:
        warnings.warn("using a randomly initialized model")
        return model

    if convert:
        to_be_converted = [key for key in model_dict.keys() if ".W." in key]
        state_dict_converted = {}
        for k, v in state_dict.items():
            k_ = ".".join(k.split(".")[0:-1] + ["W"] + k.split(".")[-1:])
            if k_ in to_be_converted:
                state_dict_converted[k_] = v

                # initialize Wa
                state_dict_converted[k_.replace(".W.", ".g_Wa.")] = v

                # initialize Wb
                if k_.split(".")[-1] == "weight":
                    g_Wb0 = model_dict[k_.replace(".W.", ".g_Wb.0.")]
                    g_Wb1 = model_dict[k_.replace(".W.", ".g_Wb.1.")]
                    rank = g_Wb0.shape[0]

                    g_Wb_init = spatial_svd(v, rank)

                    assert g_Wb0.shape == g_Wb_init[0].shape
                    assert g_Wb1.shape == g_Wb_init[1].shape

                    state_dict_converted[k_.replace(".W.", ".g_Wb.0.")] = g_Wb_init[0]
                    state_dict_converted[k_.replace(".W.", ".g_Wb.1.")] = g_Wb_init[1]

                elif k_.split(".")[-1] == "bias":
                    state_dict_converted[k_.replace(".W.", ".g_Wb.1.")] = v

            else:
                state_dict_converted[k] = v
    else:
        state_dict_converted = state_dict

    model.load_state_dict(state_dict_converted, strict=False)

    return model


def init_weights_convs(model, cfg, convert=True, deploy=False):
    """
    Initializes weights for a model.
    :param model: The model to be intialized.
    :param cfg: A configuration object for the experiment. Can contain the path of a checkpoint to load.
    :param convert: Boolean specifying whether the loaded state dictionary has to be converted
    for a Delta Distillation model.
    :param deploy: Boolean specifying whether to put the model in deploy mode. This performs hard selection of
    the student architecture based on the boolean gate for the layer.
    :return:
    The initialized model.
    """
    model = init_weights_train(model, cfg, convert)

    if deploy:
        conv_blocks = [_ for _ in model.modules() if isinstance(_, Conv)]
        for block in conv_blocks:
            if block.g_gate.bias > 0:
                block.g_W = copy.deepcopy(block.g_Wa)
            else:
                block.g_W = copy.deepcopy(block.g_Wb)

            delattr(block, "g_Wa")
            delattr(block, "g_Wb")
            delattr(block, "g_gate")
    return model


def spatial_svd(weight, rank):
    """
    Performs SVD decomposition of the weight tensor, with numpy.
    :param weight: The weight tensor.
    :param rank: The desired rank.
    :return:
    A tuple like (V,H).
    """
    W = to_numpy(weight)  # n c h w
    n, c, h, w = W.shape
    W = np.transpose(W, [1, 2, 0, 3])  # c h n w
    W = W.reshape(c * h, n * w)

    assert rank <= c * h

    V, S, H = np.linalg.svd(W, full_matrices=False)

    V = V[:, :rank]
    S = S[:rank]
    H = H[:rank, :]
    sqrt_S = np.sqrt(S)

    V = V * sqrt_S
    H = sqrt_S.reshape(sqrt_S.shape[0], 1) * H

    # rank nw -> rank n w 1
    H = H.reshape([rank, n, w, 1])
    # rank n w 1 -> n rank 1 w
    H = np.transpose(H, [1, 0, 3, 2])
    # ch rank -> c 1 h rank  -> rank c h 1
    V = V.reshape((c, 1, h, rank))
    V = np.transpose(V, [3, 0, 2, 1])

    return torch.FloatTensor(V), torch.FloatTensor(H)


def to_numpy(tensor):
    """
    Helper function that turns the given tensor into a numpy array.
    :param tensor: The original torch tensor.
    :returns
    A numpy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "is_cuda"):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    return np.array(tensor)
