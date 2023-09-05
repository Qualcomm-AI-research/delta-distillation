# ------------------------------------------------------------------------------
# Copyright (c) OpenMMLab.
# Licensed under the Apache License 2.0.
# All rights reserved.
#
# This code is from https://github.com/open-mmlab/mmtracking
# ------------------------------------------------------------------------------

from mmdet.datasets import DATASETS
from mmdet.datasets.api_wrappers import COCO

from lib.datasets.imagenet_vid_mmdet.mmtracking.coco_video_dataset import CocoVideoDataset
from lib.datasets.imagenet_vid_mmdet.mmtracking.parsers import CocoVID


@DATASETS.register_module()
class ImagenetDataset(CocoVideoDataset):
    """ImageNet VID dataset for video object detection."""

    CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
               'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
               'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
               'zebra')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotations from COCO/COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO/COCOVID apis.
        """
        if self.load_as_video:
            data_infos = self.load_video_anns(ann_file)
        else:
            data_infos = self.load_image_anns(ann_file)
        return data_infos

    def load_image_anns(self, ann_file):
        """Load annotations from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO apis.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        all_img_ids = self.coco.get_img_ids()
        self.img_ids = []
        data_infos = []
        for img_id in all_img_ids:
            info = self.coco.load_imgs([img_id])[0]
            info['filename'] = info['file_name']
            if info['is_vid_train_frame']:
                self.img_ids.append(img_id)
                data_infos.append(info)
        return data_infos

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCOVID apis.
        """
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)

            if self.test_mode:
                img_ids = self.key_img_sampling(img_ids, **self.key_img_sampler)

            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                if self.test_mode:
                    assert not info['is_vid_train_frame'], \
                        'is_vid_train_frame must be False in testing'
                    self.img_ids.append(img_id)
                    data_infos.append(info)
                elif info['is_vid_train_frame']:
                    self.img_ids.append(img_id)
                    data_infos.append(info)
        return data_infos

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
