# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is modified from https://github.com/open-mmlab/mmcv
#
# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------------
import os.path as osp
import time

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook
from mmcv.utils import TORCH_VERSION
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import digit_version
from mmcv.utils import get_logger as get_base_logger


def get_logger(cfg, exp_name):
    """
    Initialize the logger for training object detection experiments.
    :param cfg: A configuration object for the experiment.
    :param exp_name: A string identifying the experiment.
    :return:
    A tuple like (logger, metadata, timestamp).
    """
    # create work_dir
    cfg.dump(osp.join(cfg.work_dir, osp.basename(exp_name)))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_base_logger("mmtrack", log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    meta["env_info"] = "\n".join([(f"{k}: {v}") for k, v in collect_base_env().items()])
    meta["seed"] = cfg.seed
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + meta["env_info"] + "\n" + dash_line)
    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Set random seed to {cfg.seed}, " f"deterministic: {cfg.deterministic}")

    return logger, meta, timestamp


@HOOKS.register_module()
class TensorboardHook(LoggerHook):
    """
    This class implements a training hook for tensorboard logging.
    """

    def __init__(
        self, log_dir=None, interval=10, ignore_last=True, reset_flag=False, by_epoch=True
    ):
        """
        Class constructor.
        :param log_dir: Directory under which to save tensorboard logs.
        :param interval: Logging interval (every k iterations).
        :param ignore_last: Ignore the log of last iterations in each epoch if less than `interval`.
        :param reset_flag: Whether to clear the output buffer after logging.
        :param by_epoch: Whether EpochBasedRunner is used.
        """
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir
        self.writer = None

    @master_only
    def before_run(self, runner):
        """
        Hook operating before the training starts. Instantiates the SummaryWriter.
        :param runner: Experiment runner.
        """
        super().before_run(runner)
        if TORCH_VERSION == "parrots" or digit_version(TORCH_VERSION) < digit_version("1.1"):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError("Please install tensorboardX to use " "TensorboardLoggerHook.")
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, "tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        """
        Hook operating every `interval` training batches. Adds losses and metrics.
        :param runner: Experiment runner.
        """
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        """
        Hook operating after the training ends. Closes the SummaryWriter.
        :param runner: Experiment runner.
        """
        self.writer.close()
