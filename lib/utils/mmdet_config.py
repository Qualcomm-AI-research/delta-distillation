# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import os.path as osp

import mmcv
import torch
from mmcv import Config
from mmdet.apis import set_random_seed


def config_args(cfg, kwargs):
    """
    Updates args in a configuration file with args provided by command line.
    :param cfg: A configuration object (from file).
    :param kwargs: Dictionary of command line arguments.
    """
    for key in kwargs.keys():
        key_s = key.split(".")
        if len(key_s) == 1:
            setattr(cfg, key_s[0], kwargs[key_s[0]])
        elif len(key_s) == 2:
            setattr(getattr(cfg, key_s[0]), key_s[1], kwargs[key])
        else:
            raise ValueError(f"{key} can not be converted")


def config_dataset(cfg):
    """
    Checks and updates the configuration for the dataset.
    :param cfg: The original configuration object.
    """
    assert cfg.load_from is None

    dataset_type = cfg.data["type"]
    if dataset_type == "single_frame":
        assert cfg.data.num_frames == 1
    elif dataset_type == "multi_frame":
        assert cfg.data.num_frames >= 2
    elif dataset_type == "pairwise":
        assert cfg.data.num_frames == 2
    else:
        raise ValueError()

    # setup data loader
    if dataset_type == "multi_frame":
        for dataset in cfg.data.train_epoch:
            dataset["ref_img_sampler"]["num_ref_imgs"] = cfg.data.num_frames - 1

            if cfg.data.frame_range is None:
                dataset["ref_img_sampler"]["frame_range"] = [0, cfg.data.num_frames - 1]
            else:
                dataset["ref_img_sampler"]["frame_range"] = [0, cfg.data.frame_range]

        if cfg.data.val.ref_img_sampler is not None:
            cfg.data.val["ref_img_sampler"]["num_ref_imgs"] = cfg.data.num_frames - 1
            cfg.data.val["ref_img_sampler"]["frame_range"] = [0, cfg.data.num_frames - 1]

        if cfg.data.test.ref_img_sampler is not None:
            cfg.data.test["ref_img_sampler"]["num_ref_imgs"] = cfg.data.num_frames - 1
            cfg.data.test["ref_img_sampler"]["frame_range"] = [0, cfg.data.num_frames - 1]


def get_config(*config_paths, **kwargs):
    """
    Loads configuration files and updates them with command line arguments.
    :param config_paths: a tuple of configuration file paths.
    :param kwargs: dictionary of command line arguments.
    :return:
    A configuration object for the experiment.
    """
    # loading configs
    cfg = Config.fromfile(config_paths[0])
    for file in config_paths[1:]:
        cfg.merge_from_dict(Config.fromfile(file))

    config_args(cfg, kwargs)

    # setup log dir
    cfg.work_dir = osp.join(cfg.work_dir, cfg.log_name)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    config_dataset(cfg)

    # setup misc.
    cfg.gpu_ids = (
        cfg.cuda_devices if cfg.cuda_devices != "all" else list(range(torch.cuda.device_count()))
    )
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    return cfg
