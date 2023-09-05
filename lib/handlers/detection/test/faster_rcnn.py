# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import time
from collections import defaultdict

import mmcv
import torch
from mmcv.runner import get_dist_info

from lib.datasets.imagenet_vid_mmdet.mmtracking.apis.test import collect_results_cpu
from lib.utils.nn import set_delta_distillation_schedule


def get_clip_name(img_metas):
    """
    Extracts the clip name from an image metadata.
    :param img_metas: Metadata for a given image
    :return:
    A string containing the filename of the image.
    """
    return img_metas[0].data[0][0]["filename"].split("/")[-2]


def clip_not_ended_yet(frame_count_this_clip, clip_length):
    """
    Determines whether a clip has ended by comparing the frame count with the maximum clip length.
    :param frame_count_this_clip: Number of frames currently in the clip.
    :param clip_length: Desired maximum clip length.
    :return:
    A boolean expressing whether it is time to end this clip or not.
    """
    return frame_count_this_clip < clip_length


def clip_not_changed(cur_clip, old_clip):
    """
    Determines whether the clip has changed by comparing the clip names.
    :param cur_clip: clip name of current frame.
    :param old_clip: clip name of last processed frame.
    :return:
    A boolean expressing whether the new frame is from the same clip or not.
    """
    return cur_clip == old_clip


def clip_feed(model, clip_frames, clip_metas, results):
    """
    Feeds a single clip into the model.
    :param model: The object detection model.
    :param clip_frames: Batch of images to be fed to the model.
    :param clip_metas: Metadata for the images.
    :param results: Dictionary holding results, to be updated with results from this clip.
    """
    clip = torch.cat(clip_frames)
    set_delta_distillation_schedule(model, T=clip.shape[0])
    result_list = model(return_loss=False, rescale=True, img_metas=[clip_metas], img=clip)

    for k, v in result_list.items():
        results[k].extend(v)


@torch.no_grad()
def single_gpu_test(model, data_loader, clip_length):
    """
    Perform test in a non-distributed environment.
    :param model: The object detection model.
    :param data_loader: Dataloader for the test dataset.
    :param clip_length: Maximum length of clips to be fed to the model.
    :return:
    A dictionary holding results of detection.
    """
    model.eval()

    results = defaultdict(list)

    prog_bar = mmcv.ProgressBar(len(data_loader))

    clip_frames = []
    clip_metas = []
    frame_count_this_clip = 0
    old_clip_name = get_clip_name(next(iter(data_loader))["img_metas"])
    for data in data_loader:
        prog_bar.update()

        cur_clip_name = get_clip_name(data["img_metas"])
        if clip_not_ended_yet(frame_count_this_clip, clip_length) and clip_not_changed(
            cur_clip_name, old_clip_name
        ):
            # Keep collecting clip frames
            clip_frames.extend(data["img"])
            clip_metas.extend(data["img_metas"])
            frame_count_this_clip += 1
            continue

        # Clip ended, perform inference and update results
        clip_feed(model, clip_frames, clip_metas, results)

        # Initialize new clip
        clip_frames = data["img"]
        clip_metas = data["img_metas"]
        old_clip_name = cur_clip_name
        frame_count_this_clip = 1

    # perform inference on last clip
    clip_feed(model, clip_frames, clip_metas, results)

    return results


@torch.no_grad()
def multi_gpu_test(model, data_loader, clip_length, tmpdir=None, gpu_collect=False):
    """
    Perform test in a distributed environment.
    :param model: The object detection model.
    :param data_loader: Dataloader for the test dataset.
    :param clip_length: Maximum length of clips to be fed to the model.
    :return:
    A dictionary holding results of detection.
    """
    model.eval()

    results = defaultdict(list)
    rank, world_size = get_dist_info()

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))

    clip_frames = []
    clip_metas = []
    frame_count_this_clip = 0
    old_clip_name = get_clip_name(next(iter(data_loader))["img_metas"])
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for data in data_loader:
        if rank == 0:
            for _ in range(1 * world_size):
                prog_bar.update()

        cur_clip_name = get_clip_name(data["img_metas"])
        if clip_not_ended_yet(frame_count_this_clip, clip_length) and clip_not_changed(
            cur_clip_name, old_clip_name
        ):
            # Keep collecting clip frames
            clip_frames.extend(data["img"])
            clip_metas.extend(data["img_metas"])
            frame_count_this_clip += 1
            continue

        # Clip ended, perform inference and update results
        clip_feed(model, clip_frames, clip_metas, results)

        # Initialize new clip
        clip_frames = data["img"]
        clip_metas = data["img_metas"]
        old_clip_name = cur_clip_name
        frame_count_this_clip = 1

    # perform inference on last clip
    clip_feed(model, clip_frames, clip_metas, results)

    # collect results from all ranks
    if gpu_collect:
        raise NotImplementedError

    results = collect_results_cpu(results, tmpdir)
    return results
