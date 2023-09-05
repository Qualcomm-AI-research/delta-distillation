# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import torch


def roll_time(x):
    """
    Rolls the time on a tensor, that goes from shape (N,T,C,H,W) to (N*T,C,H,W).
    :param x: The tensor to be rolled.
    :return:
    The rolled tensor
    """
    return x.view((-1,) + x.shape[2:])


def unroll_time(x, T):
    """
    Unrolls the time on a tensor, that goes from shape (N*T,C,H,W) to (N,T,C,H,W).
    :param x: The tensor to be unrolled.
    :param T: The amount of frames in each clip, rolled into the batch axis.
    :return:
    The unrolled tensor
    """
    return x.view(
        (
            -1,
            T,
        )
        + x.shape[1:]
    )


def roll_time_data(
    img, img_metas, gt_bboxes, gt_labels, ref_img, ref_img_metas, ref_gt_bboxes, ref_gt_labels
):
    """
    Rolls the time on a series of tensors. Used for object detection training.
    :param img: The current image.
    :param img_metas: The metadata of the current image.
    :param gt_bboxes: The groundtruth bounding boxes of the current image.
    :param gt_labels: The groundtruth class of the current image.
    :param ref_img: The last keyframe.
    :param ref_img_metas: The metadata of the last keyframe.
    :param ref_gt_bboxes: The groundtruth bounding boxes of the last keyframe.
    :param ref_gt_labels: The groundtruth class of the last keyframe.
    :return:
    A tuple like (image, metadata, groundtruth_boxes, groundtruth_classes).
    """
    for ref_img_metas_b in ref_img_metas:
        assert len(ref_img_metas_b) == 1

    num_ref_frames = len(ref_img_metas[0])

    img = torch.cat((img.unsqueeze(dim=1), ref_img), dim=1)
    img_rolled = roll_time(img)

    img_metas = [[_1] + _2 for _1, _2 in zip(img_metas, ref_img_metas)]
    img_metas_rolled = [
        img_metas_frame for img_metas_b in img_metas for img_metas_frame in img_metas_b
    ]

    get_frame_items = lambda items, frame_id: items[items[:, 0] == frame_id][:, 1:]

    ref_gt_bboxes = [
        [get_frame_items(_1, i) for i in range(num_ref_frames)] for _1 in ref_gt_bboxes
    ]
    gt_bboxes = [[_1] + _2 for _1, _2 in zip(gt_bboxes, ref_gt_bboxes)]
    gt_bboxes_rolled = [
        gt_bboxes_frame for gt_bboxes_b in gt_bboxes for gt_bboxes_frame in gt_bboxes_b
    ]

    ref_gt_labels = [
        [get_frame_items(_1, i).squeeze(dim=1) for i in range(num_ref_frames)]
        for _1 in ref_gt_labels
    ]
    gt_labels = [[_1] + _2 for _1, _2 in zip(gt_labels, ref_gt_labels)]
    gt_labels_rolled = [
        gt_labels_frame for gt_labels_b in gt_labels for gt_labels_frame in gt_labels_b
    ]
    gt_labels_rolled = [_.long() for _ in gt_labels_rolled]

    return img_rolled, img_metas_rolled, gt_bboxes_rolled, gt_labels_rolled
