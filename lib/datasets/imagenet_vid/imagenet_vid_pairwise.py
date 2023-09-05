# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import copy
import os
import pickle
import random

import numpy as np
from PIL import Image

from .imagenet_vid_multi_frame import ImagenetVIDDatasetMultiFrame
from .transforms import RandomFlip


class ImagenetVIDDatasetPairwise(ImagenetVIDDatasetMultiFrame):
    """
    Extends the ImagenetVID dataset to support 2-frame clips.
    The distance between frames in a clip is sampled randomly during training and fixed during test.
    """

    def __init__(self, config, is_train):
        """
        Class constructor.
        :param config: A configuration object.
        :param is_train: Whether the dataset should be in training mode or not.
        """
        super().__init__(config, is_train)

        self.T = 2
        self.frame_range = (
            config.frame_range
        )  # range from keyframe used for pairwise frame sampling
        self.video_infos = None

    def get_video_id(self, img_info):
        """
        Extracts the video id from image metadata.
        :param img_info: Image metadata
        :return:
        The image id.
        """
        if img_info["id"] < 10000000:  # vid
            img_id = img_info["file_name"].split("/")[-2]
        else:  # det
            img_id = img_info["file_name"].split("/")[-1]
        return img_id

    def get_video_infos(self, num_samples=-1):
        """
        Gathers video metadata for every frame.
        :param num_samples: Specifies how many samples each video should be composed of.
        :return:
        A list of video information.
        """
        video_infos = {}
        for img_info in self.img_infos:
            if self.exclude_DET_images and img_info["id"] >= 10000000:
                continue

            video_id = self.get_video_id(img_info)
            if video_id not in video_infos.keys():
                video_infos[video_id] = []
            video_infos[video_id].append(img_info)

        self.video_infos = copy.deepcopy(video_infos)
        video_infos = self._complete_clip(video_infos, num_samples)
        return video_infos

    def __getitem__(self, index):
        """
        Loads a clip as specfied by index.
        :param index: Index of example.
        :return:
        A tuple like (image, annotations).
        """
        clip_infos_index = self.clip_infos[index]

        assert len(self.clip_infos[index]) in [1, 2]
        if self.is_train and self.T == 2:
            key_frame = clip_infos_index[0]
            all_frames = self.video_infos[self.get_video_id(key_frame)]
            key_frame_ind = list(map(lambda x: x["id"], all_frames)).index(key_frame["id"])
            ref_frame_ind = (
                key_frame_ind + random.sample(range(-self.frame_range, self.frame_range), 1)[0]
            )
            ref_frame_ind = min(max(ref_frame_ind, 0), len(all_frames) - 1)
            ref_frame = all_frames[ref_frame_ind]
            clip_infos_index = (
                [key_frame, ref_frame] if key_frame_ind <= ref_frame_ind else [ref_frame, key_frame]
            )

        anns = [self._parse_img_ann(img_info["id"], img_info) for img_info in clip_infos_index]
        paths = [img_info["file_name"] for img_info in clip_infos_index]
        paths_full = [
            os.path.join(
                self.root2["VID"] if ("VID" in path or not self.is_train) else self.root2["DET"],
                path.replace(".fake", ""),
            )
            for path in paths
        ]
        imgs = [Image.open(path).convert("RGB") for path in paths_full]

        tfs = self.transform.transforms
        for tf in tfs:
            if isinstance(tf, RandomFlip):
                do_horizontal = random.random() < tf.prob if tf.horizontal else False
                do_vertical = random.random() < tf.prob if tf.vertical else False
                tranformed = [
                    tf(img, ann, do_horizontal, do_vertical) for img, ann in zip(imgs, anns)
                ]
                imgs, anns = [_[0] for _ in tranformed], [_[1] for _ in tranformed]
            else:
                tranformed = [tf(img, ann) for img, ann in zip(imgs, anns)]
                imgs, anns = [_[0] for _ in tranformed], [_[1] for _ in tranformed]

        return np.asarray(imgs), pickle.dumps(anns)
