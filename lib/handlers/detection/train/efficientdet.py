# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import pickle
import time

import torch
from torch.optim import SGD

from lib.models.delta_distillation.conv import Conv
from lib.utils.dist import get_rank
from lib.utils.nn import set_delta_distillation_schedule


def get_optimizer(model, config_opt):
    """
    Instantiates the optimizer to be used in the optimization.
    :param model: The model to be trained.
    :param config_opt: A config object with hyperparametes about optimizers.
    :return:
    The pytorch optimizer.
    """
    wd_f = config_opt.weight_decay

    params_g, params_f = [], []
    for param_name, param in model.named_parameters():
        if "g_" in param_name:
            params_g.append(param)
        else:
            params_f.append(param)
    params = [
        {"params": params_f, "lr": config_opt.learning_rate, "weight_decay": wd_f},
        {"params": params_g, "lr": config_opt.learning_rate_g, "weight_decay": 0.0},
    ]

    optimizer = SGD(
        params, lr=config_opt.learning_rate, momentum=config_opt.momentum, weight_decay=wd_f
    )
    return optimizer


def get_gradients(model, images, targets):
    """
    Utility function to collect groundtruth deltas from the teacher model.
    :param model: The delta distillation model.
    :param images: A batch of training images.
    :param targets: A batch of groundtruth labels.
    :return:
    A list of torch.Tensor, containing for every distilled layer the groundtruth deltas.
    """
    mode = model.training
    model.eval()
    _ = [m.gradient_mode_on() for m in model.modules() if isinstance(m, Conv)]

    with torch.no_grad():
        _ = model(images, targets)

    model.train(mode)
    _ = [m.gradient_mode_off() for m in model.modules() if isinstance(m, Conv)]

    return [m.dz_ref for m in model.modules() if isinstance(m, Conv)]


coef_loss_distill = None


def train_epoch(config, model, dataloader, optimizer, tb_writer, epoch):
    """
    Implements one epoch of training loop.
    :param config: A configuration object with training hyperparameters.
    :param model: The segmentation model.
    :param dataloader: Dataloader for the training dataset.
    :param optimizer: Optimizer for gradient descent optimization.
    :param tb_writer: Tensorboard writer object.
    :param epoch: The current epoch being trained.
    """
    global coef_loss_distill

    alpha = 0 if epoch >= config.train.num_epochs // 3 else config.loss.alpha
    beta = config.loss.beta

    model.train()
    set_delta_distillation_schedule(model, T=2)

    if beta != 0:
        if config.checkpoint.mac is not None:
            macs = pickle.load(open(config.checkpoint.mac, "rb"))
        else:
            macs = 1 / len(
                [m for m in model.modules() if isinstance(m, Conv)]
            )  # uniform initialization
        macs = macs.cuda()

    data_start = time.time()
    iterations_prev_epochs = epoch * len(dataloader)
    for i, (frames, targets) in enumerate(dataloader):
        # measure data loading time
        data_time = time.time() - data_start

        batch_start = time.time()

        # need to be ran before the main forward pass
        if alpha != 0:
            dz_ref = get_gradients(model, frames, targets)

        # compute output
        output = model(frames, targets)

        # compute task loss
        task_loss = output["loss"]

        # compute gradient distillation loss
        if alpha != 0.0:
            dz_pred = [m.dz_pred for m in model.modules() if isinstance(m, Conv)]
            dist_loss = torch.stack([((x - y) ** 2).mean() for x, y in zip(dz_pred, dz_ref)])
            if coef_loss_distill is None:
                coef_loss_distill = 1.0 / dist_loss.detach().clone()
            dist_loss = (dist_loss * coef_loss_distill).mean()
        else:
            dist_loss = torch.zeros(1).cuda()

        # compute rank selection loss
        if beta != 0.0:
            gates = [m.g for m in model.modules() if isinstance(m, Conv)]
            cost_loss = (torch.stack(gates).flatten() * macs).sum()
        else:
            cost_loss = torch.zeros(1).cuda()

        # combine all three losses
        total_loss = task_loss + alpha * dist_loss + beta * cost_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - batch_start

        if get_rank() == 0 and i % config.log.frequency.iteration == 0:
            msg = (
                f"Epoch: [{i}/{len(dataloader)}]\t"
                f"Data time {data_time:.3f}s "
                f"Batch time {batch_time:.3f}s "
                f"total_loss {total_loss.item():.5f}, "
                f"task_loss {task_loss.item():.5f}, "
                f"dist_loss {dist_loss.item():.5f}, "
                f"cost_loss {cost_loss.item():.5f},"
            )
            print(msg)

            tb_step = iterations_prev_epochs + i
            tb_writer.add_scalar("train/total_loss", total_loss.item(), tb_step)
            tb_writer.add_scalar("train/task_loss", task_loss.item(), tb_step)
            tb_writer.add_scalar("train/dist_loss", dist_loss.item(), tb_step)
            tb_writer.add_scalar("train/cost_loss", cost_loss.item(), tb_step)

        data_start = time.time()
