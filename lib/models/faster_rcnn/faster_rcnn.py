# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import os
import pickle

import torch
from mmdet.core import bbox2result

from lib.handlers.detection.train.faster_rcnn import get_gradients
from lib.models.delta_distillation.conv import Conv
from lib.models.faster_rcnn.base_detector import BaseVideoDetector
from lib.utils.nn import set_delta_distillation_schedule
from lib.utils.tensor import roll_time_data

from .builder import MODELS, build_detector


@MODELS.register_module()
class VideoFasterRCNN(BaseVideoDetector):
    """
    This class implements a FasterRCNN model supporting video snippets.
    Basically, it wraps a detector model with functionalities to support video.
    """

    def __init__(
        self,
        detector,
        pretrains=None,
        init_cfg=None,
        frozen_modules=None,
        train_cfg=None,
        test_cfg=None,
    ):
        """
        Class constructor.
        :param detector: A torch model performing image object detection.
        :param pretrains: Dictionary specifying pretrained modules.
        :param init_cfg: A configuration object for the model.
        :param frozen_modules: A list of models to be frozen.
        :param train_cfg: A configuration object with training information.
        :param test_cfg: A configuration object with inference information.
        """
        super().__init__(init_cfg)

        if isinstance(pretrains, dict):
            detector_pretrain = pretrains.get("detector", None)
            if detector_pretrain:
                detector.init_cfg = dict(type="Pretrained", checkpoint=detector_pretrain)
            else:
                detector.init_cfg = None

        self.detector = build_detector(detector)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

        self.macs = None

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        ref_img=None,
        ref_img_metas=None,
        ref_gt_bboxes=None,
        ref_gt_labels=None,
        gt_instance_ids=None,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        ref_gt_instance_ids=None,
        ref_gt_bboxes_ignore=None,
        ref_gt_masks=None,
        ref_proposals=None,
        **kwargs
    ):
        """
        Performs forward propagation (training mode)
        :param img: The batch of current images.
        :param img_metas: Metadata for the current images.
        :param gt_bboxes: Groundtruth bounding boxes for the current images.
        :param gt_labels: Groundtruth classification labels for the current images.
        :param ref_img: The batch of closest past keyframes.
        :param ref_img_metas: Metadata for closest past keyframes.
        :param ref_gt_bboxes: Groundtruth bounding boxes for closest past keyframes.
        :param ref_gt_labels: Groundtruth classification labels for closest past keyframes.
        :param gt_instance_ids: Groundtruth instance identifier for the current images.
        :param gt_bboxes_ignore: Bounding boxes to be ignored for the current images.
        :param gt_masks: Groundtruth segmentation masks for the current images.
        :param proposals: Object proposals for the current images.
        :param ref_gt_instance_ids: Groundtruth instance identifier for closest past keyframes.
        :param ref_gt_bboxes_ignore: Bounding boxes to be ignored for closest past keyframes.
        :param ref_gt_masks: Groundtruth segmentation masks for closest past keyframes.
        :param ref_proposals: Object proposals for closest past keyframes.
        :param kwargs: Keyword arguments.
        :return:
        A dictionary with unweighted loss functions.
        """

        set_delta_distillation_schedule(self, T=2)

        if self.macs is None:
            macs_file = self.train_cfg["checkpoint"].mac
            self.macs = (
                pickle.load(open(macs_file, "rb")).cuda() if os.path.isfile(macs_file) else 1.0
            )

        img, img_metas, gt_bboxes, gt_labels = roll_time_data(
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            ref_img,
            ref_img_metas,
            ref_gt_bboxes,
            ref_gt_labels,
        )

        alpha, beta = self.train_cfg["loss"].alpha, self.train_cfg["loss"].beta

        if alpha != 0:
            dz_ref = get_gradients(self.detector, img, img_metas, gt_bboxes, gt_labels)

        losses = self.detector.forward_train(
            img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=gt_bboxes_ignore
        )

        if alpha != 0.0:
            dz_pred = [m.dz_pred for m in self.modules() if isinstance(m, Conv)]
            loss_distill_ = torch.stack([((x - y) ** 2).mean() for x, y in zip(dz_pred, dz_ref)])
            loss_distill = loss_distill_.mean() * alpha
        else:
            loss_distill = torch.zeros(1).cuda()
        losses.update({"loss_distill": loss_distill})

        if beta != 0.0:
            gates = [m.g for m in self.modules() if isinstance(m, Conv)]
            loss_gate = torch.stack(gates).flatten() * self.macs
            loss_gate = loss_gate.sum() * beta
        else:
            loss_gate = torch.zeros(1).cuda()

        losses.update({"loss_gate": loss_gate})

        return losses

    def train_step(self, data, optimizer):
        """
        Computes optimization objectives over a batch of data.
        :param data: Tuple with inputs to the model.
        :param optimizer: Gradient descent optimizer. Unused but required by superclass.
        :return:
        A dictionary holding loss functions and other utility variables.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["img_metas"]))

        return outputs

    def simple_test(
        self, img, img_metas, ref_img=None, ref_img_metas=None, proposals=None, rescale=False
    ):
        """
        Performs forward propagation for inference.
        :param img: The batch of current images.
        :param img_metas: Metadata for the current images.
        :param ref_img: The batch of closest past keyframes.
        :param ref_img_metas: Metadata for closest past keyframes.
        :param proposals: Object proposals for the current images.
        :param rescale: Whether to rescale the results.
        :return:
        A dictionary with detected bounding boxes.
        """

        if isinstance(img_metas[0], list):
            img_metas = [_[0] for _ in img_metas]

        # Two stage detector
        if hasattr(self.detector, "roi_head"):
            outs = self.detector.simple_test(img, img_metas, rescale=rescale)

        # Single stage detector
        elif hasattr(self.detector, "bbox_head"):
            bbox_list = self.detector.simple_test(img, img_metas, rescale=rescale)
            # skip post-processing when exporting to ONNX
            if torch.onnx.is_in_onnx_export():
                return bbox_list

            outs = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
        else:
            raise TypeError("detector must has roi_head or bbox_head.")

        results = dict()
        results["bbox_results"] = outs
        return results
