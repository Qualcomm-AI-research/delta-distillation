# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import torch
from torch import nn
from torch.nn import init

from lib.models.delta_distillation.base_ga_module import GAModule
from lib.models.delta_distillation.gate import Gate
from lib.utils.tensor import roll_time, unroll_time


class Conv(GAModule):
    """
    Implements a Deta Distillation convolution.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        factor_ch,
        channel_reduction_type,
        gate_bias_init,
    ):
        """
        Class constructor.
        :param in_planes: Number of channels in the input image.
        :param out_planes: Number of channels produced by the convolution.
        :param kernel_size: Size of the convolving kernel.
        :param stride: Stride of the convolution.
        :param padding: Padding added to all four sides of the input.
        :param dilation: Spacing between kernel elements.
        :param groups: Number of blocked connections from input channels to output channels.
        :param bias: If True, adds a learnable bias to the output.
        :param factor_ch: Compression factor for channels in the student (gamma in the paper).
        :param channel_reduction_type: Type of channel reduction (`out` or `min`).
        :param gate_bias_init: Initialization value for the gate.
        """
        super().__init__()
        conv_params = in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias

        self.factor_ch = factor_ch
        self.W = nn.Conv2d(*conv_params)

        self.g_W = None
        self.g_Wa = nn.Conv2d(*conv_params)
        self.g_Wb = _CH_REDUCTION_FN[channel_reduction_type](factor_ch, *conv_params)

        if self.g_Wa.bias is not None:
            init.constant_(self.g_Wa.bias.data, 0.0)
        if self.g_Wb[1].bias is not None:
            init.constant_(self.g_Wb[1].bias.data, 0.0)

        self.g_gate = Gate()
        init.constant_(self.g_gate.bias, gate_bias_init)

        self.gradient_mode = False

    def forward_gradient_mode(self, x):
        """
        Forward propagation that stores the temporal delta of the teachers, to be then used as groundtruth for
        the student module.
        :param x: The input tensor (with time rolled on the batch axis).
        :return:
        The output of the convolution (with time rolled on the batch axis).
        """
        z = self.W(x)
        # Store reference dz
        z_unrolled = unroll_time(z, self.T)
        z0 = z_unrolled[:, 0]
        dzt = z_unrolled[:, 1:] - z0.unsqueeze(dim=1)
        self.dz_ref = roll_time(dzt).clone().detach()
        return z

    def forward_train(self, x):
        """
        Forward propagation during training. The student regresses the delta for all frames except the first.
        A binary gate is sampled and decides what student deltas to employ (compressed or uncompressed).
        :param x: The input tensor (with time rolled on the batch axis).
        :return:
        The output of the convolution (with time rolled on the batch axis).
        """
        if self.T == 1:
            return self.W(x)

        # Compute predicted dz
        x_unrolled = unroll_time(x, self.T)
        x0 = x_unrolled[:, 0]
        dxt = x_unrolled[:, 1:] - x0.unsqueeze(dim=1)

        z0 = self.W(x0)
        dzt_a = self.g_Wa(roll_time(dxt))
        dzt_b = self.g_Wb(roll_time(dxt))
        g = self.g_gate().view(1)
        dzt = g * dzt_a + (1 - g) * dzt_b
        self.dz_pred = dzt
        self.g = g

        # Forward integration
        zt = z0.unsqueeze(dim=1) + unroll_time(dzt, self.T - 1)

        z = torch.cat((z0.unsqueeze(dim=1), zt), dim=1)
        z = roll_time(z)

        return z

    def forward_test(self, x):
        """
        Forward propagation during inference.
        :param x: The input tensor (with time rolled on the batch axis).
        :return:
        The output of the convolution (with time rolled on the batch axis).
        """
        if self.T == 1:
            return self.W(x)

        x_unrolled = unroll_time(x, self.T)
        x0 = x_unrolled[:, 0]
        dxt = x_unrolled[:, 1:] - x0.unsqueeze(dim=1)

        z0 = self.W(x0)
        dzt = self.g_W(roll_time(dxt))

        zt = z0.unsqueeze(dim=1) + unroll_time(dzt, self.T - 1)

        z = torch.cat((z0.unsqueeze(dim=1), zt), dim=1)
        z = roll_time(z)

        return z

    def train(self, mode: bool = True):
        """
        Sets the model in training mode as specified by `mode`.
        :param mode: boolean, True for training, False for inference.
        """
        super(Conv, self).train(mode)

        if not mode:  # .eval()
            if self.g_W is None:
                # Not deployed, select between g_Wa and g_Wb
                self.g_W = self.g_Wa if self.g_gate.bias > 0 else self.g_Wb

    def gradient_mode_on(self):
        """
        Activates gradient mode.
        """
        self.gradient_mode = True

    def gradient_mode_off(self):
        """
        Deactivates gradient mode.
        """
        self.gradient_mode = False


def channel_scaling_out_kernel_split(
    factor_ch, in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias
):
    """
    Instantiates a student architecture with the out policy.
    The bottleneck channels will be computed as out_planes / factor_ch.
    :param factor_ch: Compression factor for channels in the student (gamma in the paper).
    :param in_planes: Number of channels in the input image.
    :param out_planes: Number of channels produced by the convolution.
    :param kernel_size: Size of the convolving kernel.
    :param stride: Stride of the convolution.
    :param padding: Padding added to all four sides of the input.
    :param dilation: Spacing between kernel elements.
    :param groups: Number of blocked connections from input channels to output channels.
    :param bias: If True, adds a learnable bias to the output.
    :return:
    A torch module with student architecture.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
        stride = (stride,) * 2
        padding = (padding,) * 2
        dilation = (dilation,) * 2

    ch_reduced = out_planes // factor_ch if out_planes > factor_ch else out_planes
    ch_reduced = int(ch_reduced)

    g_W = nn.Sequential(
        *[
            nn.Conv2d(
                in_planes,
                ch_reduced,
                kernel_size=(kernel_size[0], 1),
                stride=(stride[0], 1),
                padding=(padding[0], 0),
                dilation=(dilation[0], 1),
                groups=groups,
                bias=False,
            ),
            nn.Conv2d(
                ch_reduced,
                out_planes,
                kernel_size=(1, kernel_size[1]),
                stride=(1, stride[1]),
                padding=(0, padding[1]),
                dilation=(1, dilation[1]),
                bias=bias,
            ),
        ]
    )
    return g_W


def channel_scaling_min_kernel_split(
    factor_ch, in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias
):
    """
    Instantiates a student architecture with the min policy.
    The bottleneck channels will be computed as min(in_planes,out_planes) / factor_ch.
    :param factor_ch: Compression factor for channels in the student (gamma in the paper).
    :param in_planes: Number of channels in the input image.
    :param out_planes: Number of channels produced by the convolution.
    :param kernel_size: Size of the convolving kernel.
    :param stride: Stride of the convolution.
    :param padding: Padding added to all four sides of the input.
    :param dilation: Spacing between kernel elements.
    :param groups: Number of blocked connections from input channels to output channels.
    :param bias: If True, adds a learnable bias to the output.
    :return:
    A torch module with student architecture.
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
        stride = (stride,) * 2
        padding = (padding,) * 2
        dilation = (dilation,) * 2

    min_planes = min(in_planes, out_planes)
    ch_reduced = min_planes // factor_ch if min_planes >= factor_ch else min_planes
    ch_reduced = int(ch_reduced)

    g_W = nn.Sequential(
        *[
            nn.Conv2d(
                in_planes,
                ch_reduced,
                kernel_size=(kernel_size[0], 1),
                stride=(stride[0], 1),
                padding=(padding[0], 0),
                bias=False,
                dilation=(dilation[0], 1),
                groups=groups,
            ),
            nn.Conv2d(
                ch_reduced,
                out_planes,
                kernel_size=(1, kernel_size[1]),
                stride=(1, stride[1]),
                padding=(0, padding[1]),
                dilation=(1, dilation[1]),
                bias=bias,
            ),
        ]
    )
    return g_W


_CH_REDUCTION_FN = {
    "out": channel_scaling_out_kernel_split,
    "min": channel_scaling_min_kernel_split,
}
