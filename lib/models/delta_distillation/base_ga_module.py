# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.

from abc import abstractmethod

from torch import nn


class GAModule(nn.Module):
    """
    Base class for Delta Distillation modules.
    """

    def __init__(self):
        """
        Class constructor.
        """
        super().__init__()

        self.mac = None
        self.x, self.z = None, None
        self.dz_pred, self.dz_ref = None, None
        self.g = None
        self.T = 1
        self.gradient_mode = False

    @abstractmethod
    def forward_gradient_mode(self, x):
        raise NotImplementedError

    @abstractmethod
    def forward_train(self, x):
        raise NotImplementedError

    @abstractmethod
    def forward_test(self, x):
        raise NotImplementedError

    def forward(self, x):
        """
        Forward function. Simply routes the computational graph depending on whether the model is training or
        collecting groundtruth deltas.
        :param x: The input tensor.
        :return:
        The output tensor.
        """
        if self.gradient_mode:
            return self.forward_gradient_mode(x)
        if self.training:
            return self.forward_train(x)
        return self.forward_test(x)

    def set_T(self, T):
        """
        Sets the amount of frames T before reinstantiating a new keyframe.
        :param T: The maximum clip length.
        """
        self.T = T
