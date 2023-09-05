# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import fire
import torch
import torch.optim

from lib.handlers.segmentation.test import test
from lib.handlers.segmentation.train import setup_training, train_epoch
from lib.utils.config import Config
from lib.utils.dist import get_rank


def main_train(*config_paths, **kwargs):
    """
    Entrypoint for training semantic segmentation models on Cityscapes.
    :param config_paths: list of paths of configuration files.
    :param kwargs: dictionary of command line arguments. Each one overrides the corresponding entry in the config file.
    """
    config = Config.load(*config_paths, **kwargs)

    trainloader, testloader, model, optimizer, tb_writer = setup_training(config)

    if get_rank() == 0:
        print(config)

    for epoch in range(config.train.num_epochs):
        if trainloader.sampler is not None and hasattr(trainloader.sampler, "set_epoch"):
            trainloader.sampler.set_epoch(epoch)

        mean_IoU, _ = test(config, testloader, model, is_model_wrapped=True)
        if get_rank() == 0:
            print(f"Val mIoU: {mean_IoU:4.4f}")
            tb_writer.add_scalar("val/mIoU", mean_IoU, epoch)

        train_epoch(config, epoch, model, trainloader, optimizer, tb_writer)
        print(f"Training epoch {epoch + 1} finished.")

        if get_rank() == 0:
            torch.save(model.module.state_dict(), f"{config.log.dir}/latest.pth")

    if get_rank() == 0:
        tb_writer.close()


if __name__ == "__main__":
    fire.Fire(main_train)
