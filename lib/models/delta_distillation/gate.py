# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import torch
from torch import nn

from .gumble_module import GumbelSoftmax


class Gate(nn.Module):
    """
    This class implements a simple binary gate learnable through Gumble Softmax optimization.
    It is used for the Student Architecture Search described in Sec. 3.2.
    """

    def __init__(self):
        """
        Class constructor.
        """
        super().__init__()
        self.gs = GumbelSoftmax()
        self.pi_log = None

        self.bias = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self):
        """
        Samples a binary value with probability guided by self.bias.
        :return:
        A binary torch tensor.
        """
        temperature = 2 / 3
        gi = self.gs(self.bias, temperature=temperature, force_hard=True)

        # to calculate loss
        self.pi_log = gi

        return gi
