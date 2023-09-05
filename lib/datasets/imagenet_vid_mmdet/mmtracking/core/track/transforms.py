# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------
import numpy as np


def restore_result(result, return_ids=False):
    """Restore the results (list of results of each category) into the results
    of the model forward.

    Args:
        result (list[ndarray]): shape (n, 5) or (n, 6)
        return_ids (bool, optional): Whether the input has tracking
            result. Default to False.

    Returns:
        tuple: tracking results of each class.
    """
    labels = []
    for i, bbox in enumerate(result):
        labels.extend([i] * bbox.shape[0])
    bboxes = np.concatenate(result, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    if return_ids:
        ids = bboxes[:, 0].astype(np.int64)
        bboxes = bboxes[:, 1:]
        return bboxes, labels, ids
    else:
        return bboxes, labels
