# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import fire
from mmcv.runner import init_dist

from lib.datasets.imagenet_vid_mmdet import build_dataset
from lib.handlers.detection.train.faster_rcnn import get_optimizer, train_model
from lib.models.det_model import get_det_model
from lib.utils.mmdet_config import get_config
from lib.utils.mmdet_logger import get_logger


def main_train(*config_paths, **kwargs):
    """
    Entrypoint for training FasterRCNN (ResNet-101) for object detection on ImagenetVID.
    :param config_paths: list of paths of configuration files.
    :param kwargs: dictionary of command line arguments.
    Each one overrides the corresponding entry in the config file.
    """
    cfg = get_config(*config_paths, **kwargs)

    distributed = cfg.local_rank >= 0
    if distributed:
        init_dist("pytorch", **cfg.dist_params)

    _, meta, timestamp = get_logger(cfg, exp_name=config_paths[0])

    model = get_det_model("faster_rcnn", cfg, stage="train")

    optimizer = get_optimizer(model, cfg.optimizer)

    datasets = [build_dataset(cfg.data.train)]

    train_model(
        model,
        datasets,
        cfg,
        optimizer=optimizer,
        distributed=distributed,
        validate=cfg.run_validation,
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    fire.Fire(main_train)
