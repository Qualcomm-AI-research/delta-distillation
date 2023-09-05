# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
#
# This code is modified from https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import numpy as np
import torch.optim
from torch.nn import functional as F
from tqdm import tqdm

from lib.utils.dist import is_distributed, reduce_tensor
from lib.utils.metrics_segmentation import get_confusion_matrix
from lib.utils.nn import set_delta_distillation_schedule
from lib.utils.tensor import unroll_time


@torch.no_grad()
def test(cfg, testloader, model, is_model_wrapped=False):
    """
    Conducts test on a segmentation dataset and computes mean Intersection over Union (mIoU).
    :param cfg: A configuration object for the experiment.
    :param testloader: Dataloader for the test dataset.
    :param model: The segmentation model.
    :param is_model_wrapped: Flag indicating whether the model is wrapped in a ModelWraper.
    :return:
    A tuple like (mean_IoU, IoU_per_class).
    """
    model.cuda()
    model.eval()

    T = 2 if cfg.data.TEST.keyframe_distance > 0 else 1
    set_delta_distillation_schedule(model, T)

    confusion_matrix = np.zeros((cfg.data.num_classes, cfg.data.num_classes))
    for idx, batch in enumerate(
        tqdm(testloader, desc=f"Inference for {cfg.data.TEST.keyframe_distance}-frame clips")
    ):
        frames, labels, _, _ = batch

        if len(frames.shape) == 5:  # merge time into batch dim
            frames = frames.flatten(end_dim=1)
            labels = labels.flatten(end_dim=1)

        size = labels.size()
        frames = frames.cuda()
        labels = labels.long().cuda()

        if is_model_wrapped:
            _, pred, _ = model(frames, labels)
        else:
            pred = model(frames)

        # only evaluate on the last frame in the sequence (manually annotated)
        pred = unroll_time(pred, T)[:, -1]

        pred = F.interpolate(
            pred, size=size[-2:], mode="bilinear", align_corners=cfg.test.pred_align_corners
        )
        confusion_matrix += get_confusion_matrix(
            labels, pred, size, cfg.data.num_classes, cfg.data.TEST.IGNORE_LABEL
        )

    if is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = tp / np.maximum(1.0, pos + res - tp)
    mean_IoU = IoU_array.mean()
    return mean_IoU, IoU_array
