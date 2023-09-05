# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is modified from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Hook,
    build_optimizer,
)
from mmcv.utils import build_from_cfg, get_logger
from mmdet.datasets import build_dataset

from lib.datasets.imagenet_vid_mmdet import build_dataloader
from lib.models.delta_distillation.conv import Conv
from lib.utils.mmdet_eval_hooks import DistEvalHook, EvalHook
from lib.utils.mmdet_train_hooks import LossWeightsSchedulingHook

# to register the hooks
assert issubclass(LossWeightsSchedulingHook, Hook)


def get_optimizer(model, config_opt):
    """
    Instantiates the optimizer to be used in the optimization.
    :param model: The model to be trained.
    :param config_opt: A config object with hyperparametes about optimizers.
    :return:
    The pytorch optimizer.
    """
    optimizer_type = config_opt.type.lower()
    assert optimizer_type in ["sgd", "adam"]

    params_g, params_f = [], []
    for param_name, param in model.named_parameters():
        if "g_" in param_name:
            params_g.append(param)
        else:
            params_f.append(param)
    params = [
        {"params": params_f, "lr": config_opt.lr, "weight_decay": config_opt.weight_decay},
        {"params": params_g, "lr": config_opt.lr_g, "weight_decay": 0.0},
    ]

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config_opt.lr,
            momentum=config_opt.momentum,
            weight_decay=config_opt.weight_decay,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(params, lr=config_opt.lr, weight_decay=config_opt.weight_decay)
    else:
        raise NotImplementedError

    return optimizer


def get_gradients(model, *inputs):
    """
    Utility function to collect groundtruth deltas from the teacher model.
    :param model: The delta distillation model.
    :param images: A batch of training images.
    :return:
    A list of torch.Tensor, containing for every distilled layer the groundtruth deltas.
    """
    mode = model.training
    model.eval()
    _ = [m.gradient_mode_on() for m in model.modules() if isinstance(m, Conv)]

    with torch.no_grad():
        _ = model.forward_train(*inputs)

    model.train(mode)
    _ = [m.gradient_mode_off() for m in model.modules() if isinstance(m, Conv)]

    return [m.dz_ref for m in model.modules() if isinstance(m, Conv)]


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
    optimizer=None,
):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_logger("mmtrack", cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. ' 'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    if optimizer is None:
        optimizer = build_optimizer(model, cfg.optimizer)

    runner = EpochBasedRunner(
        model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta
    )
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    optimizer_config = cfg.optimizer_config
    if "type" not in cfg.optimizer_config:
        optimizer_config.type = "Fp16OptimizerHook" if fp16_cfg else "OptimizerHook"
    if fp16_cfg:
        optimizer_config.update(fp16_cfg)
    if "Fp16" in optimizer_config.type:
        optimizer_config.update(distributed=distributed)

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            validation=True,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got " f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
