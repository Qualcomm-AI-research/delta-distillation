# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.

import copy
import os
import random

import cv2
import numpy as np
from PIL import Image

from .cityscapes import Cityscapes


class CityscapesSequence(Cityscapes):
    """
    Extends the cityscapes dataset to support video clips rather than just frames.
    """

    def __init__(self, config, mode):
        """
        Class constructor.
        :param config: Configuration object.
        :param mode: `train` or `val`.
        """
        super().__init__(config, mode)

        if not hasattr(config.data.TRAIN, "num_frames") and not hasattr(
            config.data.TEST, "num_frames"
        ):
            num_frames = config.data.num_frames
        else:
            num_frames = (
                config.data.TRAIN.num_frames if mode == "train" else config.data.TEST.num_frames
            )
            num_frames = num_frames if num_frames is not None else config.data.num_frames

        self.num_frames = num_frames
        self.files = self.extract_clips(clip_length=num_frames)

    def extract_clips(self, clip_length):
        """
        Complements every original file with previous frames, and builds clips of `clip_length`.
        :param clip_length: The desired clip length
        :return:
        A list of clip information.
        """
        # rename file paths
        for file in self.files:
            file["img"] = file["img"].replace("leftImg8bit/", "leftImg8bit_sequence/")

        clips = []
        for last_frame in self.files:
            clip = [last_frame]
            for i in range(1, clip_length):
                frame = self._get_frame_from_keyframe(last_frame, distance=i, with_label=True)
                clip += [frame]

            clips += [list(reversed(clip))]

        return clips

    def rand_crop_clip(self, frames, labels):
        """
        Randomly crops frames and labels for data augmentation
        :param frames: A batch of frames.
        :param labels: A batch of groundtruth labels.
        :return:
        A tuple like (cropped_frames, cropped_labels).
        """
        h, w = frames[0].shape[:-1]
        frames = [self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0)) for image in frames]
        labels = [
            self.pad_image(label, h, w, self.crop_size, (self.ignore_label,)) for label in labels
        ]

        new_h, new_w = labels[0].shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        frames = [image[y : y + self.crop_size[0], x : x + self.crop_size[1]] for image in frames]
        labels = [label[y : y + self.crop_size[0], x : x + self.crop_size[1]] for label in labels]

        return frames, labels

    def multi_scale_aug_clip(self, frames, labels, rand_scale=1, rand_crop=True):
        """
        Randomly scales frames and labels for data augmentation
        :param frames: A batch of frames.
        :param labels: A batch of groundtruth labels.
        :param rand_scale: Factor of random scaling.
        :param rand_crop: Whether to perform random cropping.
        :return:
        A tuple like (augmented_frames, augmented_labels).
        """
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = frames[0].shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        frames = [
            cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for image in frames
        ]
        labels = [
            cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST) for label in labels
        ]

        if rand_crop:
            frames, labels = self.rand_crop_clip(frames, labels)

        return frames, labels

    def gen_sample_clip(self, frames, labels, multi_scale=True, is_flip=True):
        """
        Performs a series of data augmentation for a given batch of frames and labels.
        :param frames: A batch of frames.
        :param labels: A batch of groundtruth labels.
        :param multi_scale: whether to perform scale augmentation.
        :param is_flip: whether to perform random horizontal flip.
        :return:
        A tuple like (augmented_frames, augmented_labels).
        """
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            frames, labels = self.multi_scale_aug_clip(frames, labels, rand_scale=rand_scale)
        else:
            new_h, new_w = self.crop_size
            frames = [
                cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                for image in frames
            ]
            labels = [
                cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                for label in labels
            ]

        frames = [self.input_transform(frame) for frame in frames]
        labels = [self.label_transform(label) for label in labels]

        frames = [frame.transpose((2, 0, 1)) for frame in frames]

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            frames = [frame[:, :, ::flip] for frame in frames]
            labels = [label[:, ::flip] for label in labels]

        if self.downsample_rate != 1:
            labels = [
                cv2.resize(
                    label,
                    None,
                    fx=self.downsample_rate,
                    fy=self.downsample_rate,
                    interpolation=cv2.INTER_NEAREST,
                )
                for label in labels
            ]

        return np.stack(frames), np.stack(labels)

    def __getitem__(self, index):
        """
        Returns the `index`-th clip from the dataset.
        :param index: The index of the example in the dataset.
        :return:
        A tuple like (frames, labels, img_shape, clip_name)
        """
        clip = self.files[index]
        return self._load_clip(clip)

    def _load_clip(self, clip):
        """
        Loads a given clip from disk.
        :param clip: A list of metedata about the clip to be loaded.
        :return:
        A tuple like (frames, labels, img_shape, clip_name)
        """
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

    def _get_frame_from_keyframe(self, key_frame, distance, with_label=False):
        """
        Returns metadata of a frame relative to a given keyframe.
        :param key_frame: Metadata of the keyframe.
        :param distance: Temporal distance to the keyframe.
        :param with_label: Whether the frame has a label to be loaded.
        :return:
        Metadata for the new frame.
        """
        frame = copy.deepcopy(key_frame)
        last_frame_id = frame["img"].split("/")[-1].split("_")[2]
        frame_id = f"{int(last_frame_id) - distance:06d}"

        frame["name"] = frame["name"].replace(f"_{last_frame_id}_gtFine", f"_{frame_id}_gtFine")
        frame["img"] = frame["img"].replace(
            f"_{last_frame_id}_leftImg8bit", f"_{frame_id}_leftImg8bit"
        )
        frame["label"] = None

        if with_label:
            frame["label"] = frame["label"].replace(
                f"_{last_frame_id}_gtFine", f"_{frame_id}_gtFine"
            )

        return frame
