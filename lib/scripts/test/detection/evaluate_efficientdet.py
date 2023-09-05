# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import fire
import torch
from ptflops.flops_counter import flops_to_string

from lib.datasets.imagenet_vid import imagenet_vid_multi_frame
from lib.datasets.imagenet_vid.loader import create_loader
from lib.datasets.imagenet_vid.utils import clip_collate
from lib.handlers.detection.test.efficientdet import test
from lib.models.det_model import get_det_model
from lib.models.efficientdet.effdet import get_efficientdet_config
from lib.utils.config import Config
from lib.utils.macs import compute_macs


def main_test(*config_paths, **kwargs):
    """
    Entrypoint for evaluating EfficientDet-D0 for object detection on ImagenetVID.
    :param config_paths: list of paths of configuration files.
    :param kwargs: dictionary of command line arguments. Each one overrides the corresponding entry in the config file.
    """
    cfg = Config.load(*config_paths, **kwargs)
    cfg_model = get_efficientdet_config(cfg.model.type)

    # Model
    model = get_det_model("efficientdet", cfg, stage="test")

    # Macs
    clip_size = (cfg.data.num_frames_test, 3, cfg_model.image_size, cfg_model.image_size)
    macs = compute_macs(model.model, clip_size)

    cuda_devices = (
        cfg.cuda_devices if cfg.cuda_devices != "all" else list(range(torch.cuda.device_count()))
    )
    model = torch.nn.DataParallel(model.cuda(), device_ids=cuda_devices)

    # Dataset
    val_dataset = imagenet_vid_multi_frame.ImagenetVIDDatasetMultiFrame(cfg.data, is_train=False)
    val_dataloader = create_loader(
        val_dataset,
        is_training=False,
        input_size=cfg_model.image_size,
        collate_fn=clip_collate,
        batch_size=cfg.data.batch_size.val,
        num_workers=cfg.data.num_workers,
    )

    # Test
    coco_eval = test(cfg, model, val_dataloader)

    print()
    print(f"mAP_50: {coco_eval.stats[1]}")
    print(f"Cost per frame (amortized): {flops_to_string(macs, units=None)}")


if __name__ == "__main__":
    fire.Fire(main_test)
