# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
from os.path import join

import fire
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from lib.datasets.imagenet_vid.imagenet_vid_multi_frame import (
    ImagenetVIDDatasetMultiFrame,
)
from lib.datasets.imagenet_vid.imagenet_vid_pairwise import ImagenetVIDDatasetPairwise
from lib.datasets.imagenet_vid.loader import create_loader
from lib.datasets.imagenet_vid.utils import clip_collate
from lib.handlers.detection.test.efficientdet import validate_epoch
from lib.handlers.detection.train.efficientdet import get_optimizer, train_epoch
from lib.models.det_model import get_det_model
from lib.models.efficientdet.effdet import get_efficientdet_config, unwrap_bench
from lib.utils.config import Config
from lib.utils.dist import get_rank


def train(*config_paths, **kwargs):
    """
    Entrypoint for training EfficientDet-D0 for object detection on ImagenetVID.
    :param config_paths: list of paths of configuration files.
    :param kwargs: dictionary of command line arguments.
    Each one overrides the corresponding entry in the config file.
    """
    config = Config.load(*config_paths, **kwargs)

    cuda_devices = (
        config.cuda_devices
        if config.cuda_devices != "all"
        else list(range(torch.cuda.device_count()))
    )
    torch.cuda.set_device(config.local_rank)
    # setup data parallel
    init_process_group(
        "nccl", init_method="env://", world_size=len(cuda_devices), rank=config.local_rank
    )

    if get_rank() == 0:
        print(config)
        tb_writer = SummaryWriter(config.log.dir)
    else:
        tb_writer = None

    # creat model and switch to SyncBatchNorm
    model = get_det_model("efficientdet", config, stage="train")
    config_model = get_efficientdet_config(config.model.type)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # send each model to its device
    model = model.to(torch.device(f"cuda:{config.local_rank}"))
    model = DistributedDataParallel(
        model, device_ids=[config.local_rank], find_unused_parameters=True
    )

    train_dataset = ImagenetVIDDatasetPairwise(config.data, is_train=True)
    train_dataloader = create_loader(
        train_dataset,
        is_training=True,
        collate_fn=clip_collate,
        input_size=config_model.image_size,
        distributed=True,
        batch_size=config.data.batch_size.train,
        num_workers=config.data.num_workers,
    )

    config.data.dataset_sampling_rate = 30
    validate_dataset = ImagenetVIDDatasetMultiFrame(config.data, is_train=False)
    validate_dataloader = create_loader(
        validate_dataset,
        is_training=False,
        collate_fn=clip_collate,
        input_size=config_model.image_size,
        distributed=False,
        batch_size=config.data.batch_size.val,
        num_workers=config.data.num_workers,
    )

    optimizer = get_optimizer(model, config.train.optim)
    scheduler = MultiStepLR(optimizer, milestones=[4], gamma=0.1)

    for epoch in range(config.train.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        if get_rank() == 0:
            mAP = validate_epoch(config, model, validate_dataloader)
            tb_writer.add_scalar("val/mAP", mAP, epoch)
            print(f"Validation mAP: {mAP:04f} ")

        train_epoch(config, model, train_dataloader, optimizer, tb_writer, epoch)

        print(f"Epoch {epoch + 1} finished")
        scheduler.step()

        if get_rank() == 0:
            # save checkpoints
            path = join(config.log.dir, "latest.pth")
            torch.save(unwrap_bench(model).state_dict(), path)
            print(f"Checkpoint saved at : {path}")


if __name__ == "__main__":
    fire.Fire(train)
