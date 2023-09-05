# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
#
# This code is from https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import numpy as np


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    # output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_pred = pred.argmax(dim=1).cpu().numpy()
    seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix
