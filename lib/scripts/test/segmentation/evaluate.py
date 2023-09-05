# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import random
from copy import deepcopy

import fire
import numpy as np
import torch.optim
from ptflops.flops_counter import flops_to_string
from torch.backends import cudnn
from torch.utils.data import DataLoader

from lib.datasets.cityscapes.cityscapes_pairwise import CityscapesPairwise
from lib.handlers.segmentation.test import test
from lib.models.seg_model import get_seg_model
from lib.utils.config import Config
from lib.utils.macs import compute_macs

random.seed(304)
torch.manual_seed(304)

cudnn.benchmark = True
cudnn.deterministic = True
cudnn.enabled = True


def main_test(*config_paths, **kwargs):
    """
    Entrypoint for evaluating semantic segmentation models on Cityscapes.
    :param config_paths: list of paths of configuration files.
    :param kwargs: dictionary of command line arguments.
    Each one overrides the corresponding entry in the config file.
    """
    cfg = Config.load(*config_paths, **kwargs)
    print(cfg)

    # always run the test on a single gpu
    assert cfg.local_rank == -1

    model = get_seg_model(cfg, stage="test")
    print(model)

    macs = compute_macs(model, clip_size=(cfg.data.TEST.keyframe_distance, 3, 1024, 2048))

    IoU_per_timestep = []
    for t in range(0, cfg.data.TEST.keyframe_distance):
        _cfg = deepcopy(cfg)
        _cfg.data.TEST.keyframe_distance = t

        # setup data loaders
        dataset = CityscapesPairwise(config=_cfg, mode="test")
        testloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=_cfg.data.num_workers,
            shuffle=_cfg.data.TEST.shuffle,
            pin_memory=True,
        )

        mean_IoU, _ = test(_cfg, testloader, model)
        IoU_per_timestep.append(mean_IoU)
        print(f"MeanIoU: {mean_IoU:.4f}")

    print("=" * 50)
    amort_IoU = np.mean(IoU_per_timestep)

    print()
    print(f'IoU per timestep: {["{:.4f}".format(x) for x in IoU_per_timestep]}')
    print(f'Amortized IoU: {f"{amort_IoU:.4f}"}')
    print(f"Cost per frame (amortized): {flops_to_string(macs, units=None)}")


if __name__ == "__main__":
    fire.Fire(main_test)
