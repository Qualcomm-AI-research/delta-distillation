# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import fire
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from ptflops.flops_counter import flops_to_string

from lib.datasets.imagenet_vid_mmdet import build_dataloader, build_dataset
from lib.handlers.detection.test.faster_rcnn import multi_gpu_test, single_gpu_test
from lib.models.det_model import get_det_model
from lib.utils.macs import compute_macs, faster_rcnn_constructor
from lib.utils.mmdet_config import get_config


def main_test(*config_paths, **kwargs):
    """
    Entrypoint for evaluating FasterRCNN (ResNet-101) for object detection on ImagenetVID.
    :param config_paths: list of paths of configuration files.
    :param kwargs: dictionary of command line arguments. Each one overrides the
    corresponding entry in the config file.
    """
    cfg = get_config(*config_paths, **kwargs)
    cfg.data.test.test_mode = True

    if hasattr(cfg.model, "detector"):
        cfg.model.detector.pretrained = None

    distributed = cfg.local_rank >= 0
    if distributed:
        init_dist("pytorch", **cfg.dist_params)

    # Model
    model = get_det_model("faster_rcnn", cfg, stage="test")

    # Macs
    mac_clip_size = (cfg.model.test_cfg.key_frame_interval, 3, 584, 966)
    macs = compute_macs(model, mac_clip_size, input_constructor=faster_rcnn_constructor)

    # Dataset
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=distributed, shuffle=False
    )

    if not hasattr(model, "CLASSES"):
        model.CLASSES = dataset.CLASSES

    # Test
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, cfg.model.test_cfg.key_frame_interval)
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, cfg.model.test_cfg.key_frame_interval)

    rank, _ = get_dist_info()
    if rank == 0:
        print(f"\nwriting results to {cfg.work_dir}")
        mmcv.dump(outputs, f"{cfg.work_dir}/results.pkl")
        metr_dict = dataset.evaluate(outputs, metric=["bbox"])

        print()
        print(f'mAP_50: {metr_dict["bbox_mAP_50"]}')
        print(f"Cost per frame (amortized): {flops_to_string(macs, units=None)}")


if __name__ == "__main__":
    fire.Fire(main_test)
