# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Josip Saric.
import copy
import logging
import numpy as np
from typing import Callable, List, Union
import torch
from pathlib import Path
from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import (
    BoxMode
)
from .target_generator import PanopticDeepLabTargetGenerator

__all__ = ["PanopticDeeplabDatasetMapper"]


class ExtendedAugInput(T.AugInput):
    def __init__(self, image, boxes=None, sem_seg=None, box_labels=None, baol_offset_weights=None):
        super(ExtendedAugInput, self).__init__(image, boxes=boxes, sem_seg=sem_seg)
        self.box_labels = box_labels
        self.baol_offset_weights = baol_offset_weights

    def transform(self, tfm):
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)
        if self.baol_offset_weights is not None:
            self.baol_offset_weights = tfm.apply_segmentation(self.baol_offset_weights)


class PanopticDeeplabDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        panoptic_target_generator: Callable,
        load_offset_weights=False,
        offset_weights_folder_path=""
    ):
        """
        NOTE: this interface is experimental.
        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
        """
        # fmt: off
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.panoptic_target_generator = panoptic_target_generator
        self.load_offset_weights = load_offset_weights
        self.offset_weights_folder_path = offset_weights_folder_path

    @classmethod
    def from_config(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
            "load_offset_weights": cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.BAOL.ENABLED,
            "offset_weights_folder_path": cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.BAOL.WEIGHTS_PATH
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load image.
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        # Panoptic label is encoded in RGB image.
        pan_seg_file_name = dataset_dict.pop("pan_seg_file_name")

        pan_seg_gt = utils.read_image(pan_seg_file_name, "RGB")
        baol_offset_weights = None
        if self.load_offset_weights:
            try:
                baol_offset_weights = utils.read_image(f"{self.offset_weights_folder_path}/{Path(pan_seg_file_name).stem}.png")
            except Exception:
                print("Could not load baol offset weights for image: ", pan_seg_file_name)
                baol_offset_weights = np.ones(pan_seg_gt.shape[:2])


        # Reuses semantic transform for panoptic labels.
        boxes, box_labels = [], []
        for obj in dataset_dict["segments_info"]:
            boxes.append(obj["bbox"])
            box_labels.append(obj["category_id"])
        if len(boxes) != 0:
            boxes = np.array(boxes)
            box_labels = np.array(box_labels)
            boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            boxes = torch.as_tensor(boxes)
            box_labels = torch.as_tensor(box_labels)
        else:
            boxes = None
            box_labels = None

        aug_input = ExtendedAugInput(
            image,
            sem_seg=pan_seg_gt,
            boxes=boxes,
            box_labels=box_labels,
            baol_offset_weights=baol_offset_weights
        )
        _ = self.augmentations(aug_input)
        image, pan_seg_gt = aug_input.image, aug_input.sem_seg
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # Generates training targets for Panoptic-DeepLab.

        targets = self.panoptic_target_generator(
            rgb2id(pan_seg_gt),
            dataset_dict["segments_info"],
            same_pallet_ids=dataset_dict.get("same-pallet-ids", None)
        )
        if self.load_offset_weights:
            baol_offset_weights = torch.from_numpy(np.array(aug_input.baol_offset_weights)).unsqueeze(0)
            targets["offset_weights"] = targets["offset_weights"] * baol_offset_weights

        dataset_dict.update(targets)

        return dataset_dict
