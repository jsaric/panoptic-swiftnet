import random
import numpy as np
import torch
from fvcore.transforms.transform import (
    CropTransform
)
from detectron2.data.transforms.augmentation import Augmentation


class ClassUniformCrop(Augmentation):
    """
    Favors crops containing rare classes. Use only with RepeatFactorTrainingSampler.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.
        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())
        self.cat_factors = None
        self.p_true_random_crop = 0.5

    def set_cat_repeat_factors(self, factors):
        self.cat_factors = factors
        self.rare_classes = torch.tensor([i for i in range(len(self.cat_factors)) if self.cat_factors[i] > 1.0])

    def get_transform(self, image, boxes, box_labels):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        target_box = self._choose_target_instance(boxes, box_labels)
        if target_box is None:
            h0, w0 = self._rand_location(h, w, croph, cropw)
        else:
            h0, w0 = self._choose_target_crop(h, w, croph, cropw, target_box)

        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        return CropTransform(w0, h0, cropw, croph)

    def _choose_target_instance(self, boxes, box_labels):
        if boxes is None or box_labels is None:
            return None
        combined = torch.cat((self.rare_classes, box_labels.unique()))
        uniques, counts = combined.unique(return_counts=True)
        rare_classes = uniques[counts > 1]
        if len(rare_classes) == 0:
            return None
        c = random.choices(rare_classes, weights=self.cat_factors[rare_classes])[0]
        box_idx = random.choice(torch.where(box_labels == c)[0])
        box = boxes[box_idx]
        return box

    def _choose_target_crop(self, h, w, croph, cropw, bbox):
        if bbox is not None:
            if random.uniform(0, 1) > self.p_true_random_crop:
                for _ in range(50):
                    h0, w0 = self._rand_location(h, w, croph, cropw)
                    h1, w1 = h0 + croph, w0 + cropw
                    if self.bb_intersection_over_union([w0, h0, w1, h1], [*bbox]) > 0.:
                        break
                return h0, w0
        return self._rand_location(h, w, croph, cropw)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width
        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))

    def _rand_location(self, h, w, croph, cropw):
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return h0, w0

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou