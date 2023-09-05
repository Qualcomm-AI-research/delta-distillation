# Copyright (c) 2023 Qualcomm Technologies, Inc.

# All Rights Reserved.
import copy
import os
import pickle
import random

import numpy as np
from PIL import Image

from .dataset import CocoDetection
from .transforms import RandomFlip


class ImagenetVIDDatasetMultiFrame(CocoDetection):
    """
    Implements the ImagenetVID dataset.
    """

    def __init__(self, config, is_train):
        """
        Class constructor.
        :param config: A configuration object.
        :param is_train: Whether the dataset should be in training mode or not.
        """
        path_label = config.label.train if is_train else config.label.val
        path_data = config.frames.vid
        super().__init__(root=path_data, ann_file=path_label, transform=None)

        self.root2 = {"VID": config.frames.vid, "DET": config.frames.det}
        self.is_train = is_train
        self.num_frames_per_video = config.num_frames_per_video if self.is_train else -1
        self.T = config.num_frames_test
        self.dataset_sampling_rate = config.dataset_sampling_rate
        self.exclude_DET_images = config.exclude_DET_images

        self._clip_infos = None  # lazy loading of clip infos

    def get_video_infos(self, num_samples=-1):
        """
        Gathers video metadata for every frame.
        :param num_samples: Specifies how many samples each video should be composed of.
        :return:
        A list of video information.
        """
        video_infos = {}
        for img_info in self.img_infos:
            if img_info["id"] < 10000000:  # vid
                img_id = img_info["file_name"].split("/")[-2]
            else:  # det
                img_id = img_info["file_name"].split("/")[-1]
            if not img_id in video_infos.keys():
                video_infos[img_id] = []
            video_infos[img_id].append(img_info)

        video_infos = self._complete_clip(video_infos, num_samples)
        return video_infos

    def extract_clips(self, interval):
        """
        Extracts clips of given length.
        :param interval: Desired clip length.
        :return:
        A list of clip metadata.
        """
        video_infos = self.get_video_infos(self.num_frames_per_video)
        if interval == -1:
            return video_infos

        n_DET, n_VID = 0, 0
        clip_infos = {}
        for video_name, video_info in video_infos.items():
            for i in range(0, len(video_info), interval):
                clip_i_info = (
                    video_info[i : i + interval]
                    if i + interval < len(video_info)
                    else video_info[i::]
                )
                for _ in range(len(clip_i_info), interval):
                    clip_i_info.append(copy.deepcopy(clip_i_info[-1]))
                    # mark the fake frames added at the end of short clips
                    clip_i_info[-1]["file_name"] += ".fake"
                clip_infos[f"{video_name}_{i // interval}"] = clip_i_info
            if len(video_info) == 1:
                n_DET = n_DET + 1
            else:
                n_VID = n_VID + len(range(0, len(video_info), interval))
        print(f"total #clips: {n_VID + n_DET} (n_VID={n_VID}, n_DET={n_DET})")

        return list(clip_infos.values())

    def __getitem__(self, index):
        """
        Loads a clip as specfied by index.
        :param index: Index of example.
        :return:
        A tuple like (image, annotations).
        """
        anns = [
            self._parse_img_ann(img_info["id"], img_info) for img_info in self.clip_infos[index]
        ]
        paths = [img_info["file_name"] for img_info in self.clip_infos[index]]
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

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.clip_infos)

    @property
    def clip_infos(self):
        """
        Lazy getter of clip infos.
        """
        if self._clip_infos is None:
            clip_infos = self.extract_clips(interval=self.T)
            # sub-sample the dataset
            if self.dataset_sampling_rate != -1:
                clip_infos = [
                    clip_infos[i] for i in range(0, len(clip_infos), self.dataset_sampling_rate)
                ]
            self._clip_infos = clip_infos
        return self._clip_infos

    @staticmethod
    def _complete_clip(video_infos, num_samples):
        """
        Utility function to update and complete video infos.
        :param video_infos: Original video metadata.
        :param num_samples: Specifies how many samples each video should be composed of.
        :return:
        Updated video metadata.
        """
        if num_samples > 0:
            for key, video_info in video_infos.items():
                video_length = len(video_info)
                if video_length == 1:
                    continue  # DET dataset
                idx = np.linspace(0, video_length - 1, num=num_samples + 1)
                indices = []
                for i in range(len(idx) - 1):
                    indices.append(
                        np.floor(0.5 * (idx[i] + idx[i + 1])).astype("int")
                    )  # VID format
                video_info_updated = np.array(video_info)[indices].tolist()
                video_infos[key] = video_info_updated
        return video_infos
