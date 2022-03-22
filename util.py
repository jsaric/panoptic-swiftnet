import json
import os
from collections import OrderedDict

import numpy as np
from PIL import Image as img
from detectron2.evaluation import (
    DatasetEvaluator
)
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from tabulate import tabulate

import torch
import itertools
from panopticapi.utils import rgb2id
import io
import contextlib
import tempfile


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def save_semantic_annotation(label,
                    save_dir,
                    filename,
                    colormap=None):
    """Saves the given label to image on disk.
    Args:
        label: The numpy array to be saved. The data will be converted
            to uint8 and saved as png image.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        add_colormap: Boolean, add color map to the label or not.
        normalize_to_unit_values: Boolean, normalize the input values to [0, 1].
        scale_values: Boolean, scale the input values to [0, 255] for visualization.
        colormap: A colormap for visualizing segmentation results.
        image: merge label with image if provided
    """
    # Add colormap for visualizing the prediction.
    colored_label = colormap[label]
    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image


def save_panoptic_annotation(label,
                             save_dir,
                             filename,
                             label_divisor,
                             colormap=None,
                             image=None):
    """Saves the given label to image on disk.
    Args:
        label: The numpy array to be saved. The data will be converted
            to uint8 and saved as png image.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        colormap: A colormap for visualizing segmentation results.
        image: merge label with image if provided
    """
    if colormap is None:
        raise ValueError('Expect a valid colormap.')

    # Add colormap to label.
    colored_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    taken_colors = set([0, 0, 0])

    def _random_color(base, max_dist=50):
        new_color = base + np.random.randint(low=-max_dist,
                                             high=max_dist + 1,
                                             size=3)
        return tuple(np.maximum(0, np.minimum(255, new_color)))

    for lab in np.unique(label):
        mask = label == lab
        base_color = colormap[lab // label_divisor]
        if tuple(base_color) not in taken_colors:
            taken_colors.add(tuple(base_color))
            color = base_color
        else:
            while True:
                color = _random_color(base_color)
                if color not in taken_colors:
                    taken_colors.add(color)
                    break
        colored_label[mask] = color

    if image is not None:
        colored_label = 0.5 * colored_label + 0.5 * image

    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def save_offset_image(offset,
                      save_dir,
                      filename):
    """Saves image with heatmap.
    Args:
        image: The offset to save.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
    """
    offset_image = flow_compute_color(offset[:, :, 1], offset[:, :, 0])
    pil_image = img.fromarray(offset_image.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def save_heatmap_image(image,
                       center_heatmap,
                       save_dir,
                       filename,
                       ratio=0.5):
    """Saves image with heatmap.
    Args:
        image: The image.
        center_heatmap: Ndarray, center heatmap.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        radio: Float, ratio to mix heatmap and image, out = ratio * heatmap + (1 - ratio) * image.
    """
    center_heatmap = center_heatmap[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
    center_heatmap = center_heatmap.clip(0, 255)
    image = ratio * center_heatmap + (1 - ratio) * image
    pil_image = img.fromarray(image.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")


        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = {**meta.stuff_dataset_id_to_contiguous_id,
                   **meta.thing_dataset_id_to_contiguous_id}
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None

        self._num_classes = len(self._contiguous_id_to_dataset_id.keys())
        self._class_names = meta.all_classes

        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            panoptic = rgb2id(np.array(img.open(input["pan_seg_file_name"])))
            gt = self.panoptic2semantic(panoptic, input["segments_info"])
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        print(results)
        self._logger.info(results)
        return results

    def panoptic2semantic(self, panoptic, segments_info):
        semantic = np.zeros_like(panoptic, dtype=np.int) + self._ignore_label
        for seg in segments_info:
            cat_id = seg["category_id"]
            if cat_id == self._ignore_label:
                continue
            semantic[panoptic == seg["id"]] = cat_id
        return semantic

