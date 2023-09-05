# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.

import copy
import os
import random

import cv2
import numpy as np
from PIL import Image

from .cityscapes_sequence import CityscapesSequence


class CityscapesPairwise(CityscapesSequence):
    """
    Extends the cityscapes sequence dataset to support 2-frame clips.
    The distance between frames in a clip is sampled randomly during training and fixed during test.
    """

    def __init__(self, config, mode):
        """
        Class constructor.
        :param config: Configuration object.
        :param mode: `train` or `val`.
        """
        super(CityscapesSequence, self).__init__(config, mode)

        self.frame_distance = (
            config.data.TRAIN.keyframe_distance
            if mode == "train"
            else config.data.TEST.keyframe_distance
        )
        self.mode = mode
        self.files = self.extract_clips(clip_length=1)

    def __getitem__(self, index):
        """
        Returns the `index`-th clip from the dataset.
        :param index: The index of the example in the dataset.
        :return:
        A tuple like (frames, labels, img_shape, clip_name)
        """
        key_frame = self.files[index][0]
        distance = (
            self.frame_distance if self.mode == "test" else random.randint(1, self.frame_distance)
        )
        if distance == 0:
            clip = [key_frame]
        else:
            other_frame = self._get_frame_from_keyframe(key_frame, distance)
            clip = [other_frame, key_frame]

        frames = [
            cv2.imread(os.path.join(self.root, frame["img"]), cv2.IMREAD_COLOR) for frame in clip
        ]
        size, name = frames[0].shape, clip[0]["name"]

        if "test" in self.list_path:
            frames = [self.input_transform(frame) for frame in frames]
            frames = [frame.transpose((2, 0, 1)) for frame in frames]
            return frames.copy(), np.array(size), name

        labels = [
            np.array(Image.open(os.path.join(self.root, f["label"])))
            for f in clip
            if f["label"] is not None
        ]
        labels = [self.convert_label(l) for l in labels]

        frames, labels = self.gen_sample_clip(frames, labels, self.multi_scale, self.flip)

        return frames.copy(), labels.copy(), np.array(size), name
