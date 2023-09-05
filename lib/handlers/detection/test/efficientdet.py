# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import json
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval

from lib.models.efficientdet.effdet.evaluator import COCOEvaluator
from lib.utils.nn import set_delta_distillation_schedule


@torch.no_grad()
def validate_epoch(config, model, dataloader):
    """
    Performs quick validation after one epoch of training.
    :param config: Configuration object for the experiment.
    :param model: The model to be tested.
    :param dataloader: Dataloader for the test dataset.
    :return:
    A float containing mAP@0.5, computed with Coco API.
    """
    model.eval()
    set_delta_distillation_schedule(model, T=config.data.num_frames_test)

    evaluator = COCOEvaluator(dataloader.loader.dataset.coco)

    for i, (frames, targets) in enumerate(dataloader):
        output = model(frames, targets)
        evaluator.add_predictions(output["detections"], targets)
        if i % config.log.frequency.iteration == 0:
            print(f"Validation: [{i}/{len(dataloader)}] processed")

    return evaluator.evaluate()


@torch.no_grad()
def test(config, model, dataloader):
    """
    Performs validation on Imagenet VID and computes metrics with COCO API.
    :param config: Configuration object for the experiment.
    :param model: The model to be tested.
    :param dataloader: Dataloader for the test dataset.
    :return:
    The coco api object holding object detection metrics.
    """
    model.eval()
    set_delta_distillation_schedule(model, config.data.num_frames_test)

    img_ids, results = [], []
    for i, (frames, targets) in enumerate(dataloader):
        output = model(frames, targets["img_scale"], targets["img_size"])
        output = output.cpu()
        sample_ids = targets["img_id"].cpu()
        for index, sample in enumerate(output):
            image_id = int(sample_ids[index])
            for det in sample:
                score = float(det[4])
                if score < 0.001:
                    break

                coco_det = {
                    "image_id": image_id,
                    "bbox": det[0:4].tolist(),
                    "score": score,
                    "category_id": det[5].item(),
                }
                img_ids.append(image_id)
                results.append(coco_det)

        if i % config.log.frequency.iteration == 0:
            print(f"Test: [{i}/{len(dataloader)}] processed")

    res_file = Path(config.log.results_dir) / "results.json"
    res_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, res_file.open("w"), indent=4)
    coco_results = dataloader.loader.dataset.coco.loadRes(str(res_file))
    coco_eval = COCOeval(dataloader.loader.dataset.coco, coco_results, "bbox")
    coco_eval.params.imgIds = img_ids  # score only ids we've used
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
