# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class LossWeightsSchedulingHook(Hook):
    """
    Hook that schedules the loss weights.
    """

    def __init__(self, at_epoch, new_lr, new_lr_g, new_alpha):
        """
        Class constructor.
        :param at_epoch: Epoch at which this hook triggers.
        :param new_lr: Updated learning rate for the Delta Distillation teacher.
        :param new_lr_g: Updated learning rate for the Delta Distillation student.
        :param new_alpha: Updated loss weight for the Delta Distillation loss.
        """
        self.at_epoch = at_epoch
        self.new_lr = new_lr
        self.new_lr_g = new_lr_g
        self.new_alpha = new_alpha

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        """
        Hook operating before Every epoch. Adjusts the loss weights depending on the current epoch number.
        :param runner: Experiment runner.
        """

        if runner.epoch >= self.at_epoch:
            runner.model.module.train_cfg["loss"].alpha = self.new_alpha

            if self.new_lr is not None:
                runner.optimizer.param_groups[0]["lr"] = self.new_lr

            if self.new_lr_g is not None:
                runner.optimizer.param_groups[1]["lr"] = self.new_lr_g

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
