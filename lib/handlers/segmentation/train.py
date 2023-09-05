# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
#
# This code is modified from https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import pickle
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn
import torch.nn as nn
import torch.optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.datasets.cityscapes.cityscapes_pairwise import CityscapesPairwise
from lib.models.delta_distillation.conv import Conv
from lib.models.delta_distillation.gate import Gate
from lib.models.seg_model import get_seg_model
from lib.utils.criterion_segmentation import OhemCrossEntropy
from lib.utils.dist import get_rank, get_sampler
from lib.utils.nn import set_delta_distillation_schedule
from lib.utils.tensor import unroll_time


class ModelWraper(nn.Module):
    """
    Helper class that wraps the segmentation model and the segmentation objective under a single nn.Module.
    """

    def __init__(self, model, loss):
        """
        Class constructor.
        :param model: The segmentation model.
        :param loss: The optimization objective.
        """
        super(ModelWraper, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, labels, *args, **kwargs):
        """
        Feeds the model with some training examples.
        :param inputs: The input batch of images.
        :param labels: The groundtruth labels.
        :param args: Any positional argument to be used in model forward.
        :param kwargs: Any keyword argument to be used in model forward.
        :return:
        A tuple like (loss_value, prediction_outputs, log_dictionary)
        """
        outputs = self.model(inputs, *args, **kwargs)
        outputs_last_frame = unroll_time(outputs, T=2)[:, -1]
        loss = self.loss(outputs_last_frame, labels)
        return torch.unsqueeze(loss, 0), outputs, self.get_log_dict()

    def get_log_dict(self):
        """
        Inspects the segmentation model and populates a dictionary with utility tensors useful
        to compute the distillation and sparsity loss.
        :return:
        A tuple like (predicted_deltas, reference_deltas, gating_decisions).
        """
        modules = list(self.model.modules())
        dz_pred = [x.dz_pred for x in modules if hasattr(x, "dz_pred") and x.dz_pred is not None]
        dz_ref = [x.dz_ref for x in modules if hasattr(x, "dz_ref") and x.dz_ref is not None]
        gates = [x.g for x in modules if hasattr(x, "g") and x.g is not None]

        output = dict()
        if len(dz_pred):
            output.update({"dz_pred": dz_pred})
        if len(dz_ref):
            output.update({"dz_ref": dz_ref})
        if len(gates):
            output.update({"gates": gates})
        return output


def get_optimizer(model, config_opt):
    """
    Instantiates the optimizer to be used in the optimization.
    :param model: The model to be trained.
    :param config_opt: A config object with hyperparametes about optimizers.
    :return:
    The pytorch optimizer.
    """
    params_g, params_f = [], []
    for param_name, param in model.named_parameters():
        if "g_" in param_name:
            params_g.append(param)
        else:
            params_f.append(param)
    params = [
        {"params": params_f, "lr": config_opt.lr},
        {"params": params_g, "lr": config_opt.lr_g},
    ]

    optimizer = torch.optim.SGD(params, lr=config_opt.lr, momentum=0.9)
    return optimizer


def setup_training(config):
    """
    This function takes care of setting up everything useful for training the segmentation model.
    Specifically, it:
        - creates the dataset objects for training and validation;
        - instantiates the segmentation model to be trained and its loss function;
        - sets up the optimizer;
        - initializes process groups for distributed processing;
        - fixes random seeds;
        - creates the Tensorboard writer;
    :param config: A configuration object with training hyperparameters.
    :return:
    A tuple like (dataloader_training, dataloader_test, model, optimizer, tensorboard_writer).
    """
    random.seed(304)
    torch.manual_seed(304)

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    conf_data, conf_train = config.data, config.train

    if config.local_rank >= 0:
        torch.cuda.set_device(torch.device("cuda:{}".format(config.local_rank)))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    tb_writer = None
    if get_rank() == 0:
        tb_writer = SummaryWriter(config.log.dir)

    # setup data loaders
    gpus = (
        config.cuda_devices
        if config.cuda_devices != "all"
        else list(range(torch.cuda.device_count()))
    )
    batch_size = config.data.TRAIN.batch_size * (1 if config.local_rank >= 0 else len(gpus))
    train_dataset = CityscapesPairwise(config=config, mode="train")
    train_sampler = get_sampler(train_dataset)
    trainloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=conf_data.num_workers,
        shuffle=conf_data.TRAIN.shuffle and train_sampler is None,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = CityscapesPairwise(config=config, mode="test")
    test_sampler = get_sampler(test_dataset)
    testloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=config.data.TEST.batch_size,
        num_workers=conf_data.num_workers,
        shuffle=conf_data.TEST.shuffle,
        pin_memory=True,
    )

    # setup loss
    criterion = OhemCrossEntropy(
        config,
        ignore_label=conf_data.TRAIN.IGNORE_LABEL,
        thres=conf_train.loss.ohem_thresh,
        min_kept=conf_train.loss.ohem_keep,
        weight=train_dataset.class_weights,
    )

    # setup model
    model = get_seg_model(config, stage="train")
    model = ModelWraper(model, criterion)
    if config.local_rank >= 0:
        device = torch.device("cuda:{}".format(config.local_rank))
        model = model.to(device)
        model = DistributedDataParallel(
            model, device_ids=[config.local_rank], output_device=config.local_rank
        )
    else:
        device = list(range(torch.cuda.device_count()))
        model = DataParallel(model, device_ids=device).cuda()

    # setup optimizer
    optimizer = get_optimizer(model, conf_train.optim)

    if get_rank() == 0:
        print(model)
        print(optimizer)
        print(train_dataset)

    return trainloader, testloader, model, optimizer, tb_writer


def get_gradients(model, images, labels):
    """
    Utility function to collect groundtruth deltas from the teacher model.
    :param model: The delta distillation model.
    :param images: A batch of training images.
    :param labels: A batch of groundtruth labels.
    :return:
    A list of torch.Tensor, containing for every distilled layer the groundtruth deltas.
    """
    mode = model.training
    model.eval()
    [m.gradient_mode_on() for m in model.modules() if isinstance(m, Conv)]

    with torch.no_grad():
        _, _, log_dict = model(images, labels)

    model.train(mode)
    [m.gradient_mode_off() for m in model.modules() if isinstance(m, Conv)]

    return log_dict["dz_ref"]


coef_loss_distill = None


def train_epoch(config, epoch, model, trainloader, optimizer, tb_writer):
    """
    Implements one epoch of training loop.
    :param config: A configuration object with training hyperparameters.
    :param epoch: The current epoch being trained.
    :param model: The segmentation model.
    :param trainloader: Dataloader for the training dataset.
    :param optimizer: Optimizer for gradient descent optimization.
    :param tb_writer: Tensorboard writer object.
    """
    alpha = 0.0 if epoch > config.train.num_epochs // 3 else config.train.loss.alpha
    beta = config.train.loss.beta

    global coef_loss_distill

    if beta != 0:
        if config.checkpoint.mac is not None:
            macs = pickle.load(open(config.checkpoint.mac, "rb"))
        else:
            macs = 1 / len(
                [m for m in model.modules() if isinstance(m, Conv)]
            )  # uniform initialization
        macs = macs.cuda()

    model.train()
    set_delta_distillation_schedule(model, T=2)

    if get_rank() == 0:
        print(f"started training epoch {epoch + 1} ...")

    num_epoch = config.train.num_epochs
    gpus = (
        config.cuda_devices
        if config.cuda_devices != "all"
        else list(range(torch.cuda.device_count()))
    )
    epoch_iters = np.int(trainloader.dataset.__len__() / config.data.TRAIN.batch_size / len(gpus))
    num_iters = num_epoch * epoch_iters
    base_lr_f = config.train.optim.lr
    base_lr_g = config.train.optim.lr_g

    tic = time.time()
    cur_iters = epoch * epoch_iters

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch

        if len(images.shape) == 5:  # merge time into batch dim
            images = images.flatten(end_dim=1)
            labels = labels.flatten(end_dim=1)

        images = images.cuda()
        labels = labels.long().cuda()

        if alpha != 0.0:
            dz_ref = get_gradients(model, images, labels)

        task_loss, outputs, log_dict = model(images, labels)

        # compute gradient distillation loss
        if alpha != 0.0:
            dz_pred = log_dict["dz_pred"]
            dist_loss = torch.stack([((x - y) ** 2).mean() for x, y in zip(dz_pred, dz_ref)])
            if coef_loss_distill is None:
                coef_loss_distill = 1.0 / dist_loss.mean().item()
            dist_loss = dist_loss.mean() * coef_loss_distill
        else:
            dist_loss = torch.zeros(1).cuda()

        # compute cost loss
        if beta != 0.0:
            gates = log_dict["gates"]
            cost_loss = (torch.stack(gates).flatten() * macs).sum()
        else:
            cost_loss = torch.zeros(1).cuda()

        # combine all three losses
        total_loss = task_loss + alpha * dist_loss + beta * cost_loss

        # Perform optimizer step
        model.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - tic
        tic = time.time()

        # adjust learning rate
        adjust_learning_rate(optimizer, base_lr_f, base_lr_g, num_iters, i_iter + cur_iters)

        if get_rank() == 0 and i_iter % config.log.frequency.iteration == 0:
            msg = (
                f"Epoch: [{epoch+1}/{num_epoch}] "
                f"Iter:[{i_iter}/{epoch_iters}], "
                f'Time: {batch_time:.2f}, lr: {[x["lr"] for x in optimizer.param_groups]}, '
                f"Loss: {total_loss.item():.6f}, "
                f"Task loss: {task_loss.item():.6f}, "
                f"Dist loss: {dist_loss.item():.6f}, "
                f"Cost loss: {cost_loss.item():.6f}, "
            )
            print(msg)

            step = i_iter + cur_iters
            tb_writer.add_scalar("train/total_loss", total_loss.item(), step)
            tb_writer.add_scalar("train/task_loss", task_loss.item(), step)
            tb_writer.add_scalar("train/dist_loss", dist_loss.item(), step)
            tb_writer.add_scalar("train/cost_loss", cost_loss.item(), step)

            gate_biases = [
                (n, g.bias.item()) for (n, g) in model.named_modules() if isinstance(g, Gate)
            ]
            for i, (n, gb_i) in enumerate(gate_biases):
                tb_writer.add_scalar(f"pi_logs_bias/{i + 1}_{n}", gb_i, step)


def adjust_learning_rate(optimizer, base_lr_f, base_lr_g, max_iters, cur_iters, power=0.9):
    """
    Adjusts the learning rate following a cosine decay.
    :param optimizer: The optimizer object.
    :param base_lr_f: The initial learning rate for the teacher parameters.
    :param base_lr_g: The initial learning rate for the student parameters.
    :param max_iters: The total number of training iterations.
    :param cur_iters: The current iteration number.
    :param power: The power of the polynomial for cosine decay.
    :return:
    The new learning rate for the teacher parameters.
    """
    lr_f = base_lr_f * ((1 - float(cur_iters) / max_iters) ** (power))
    lr_g = base_lr_g * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]["lr"] = lr_f
    optimizer.param_groups[1]["lr"] = lr_g
    return lr_f
