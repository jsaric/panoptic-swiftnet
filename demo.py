import argparse

import torch
import os
from train_net import setup, Trainer, DetectionCheckpointer
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Demo inference on a single image.')
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-image',
                        help='Path to the input image',
                        required=True)
    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def panoptic_labels_to_color(label, colormap, label_divisor):
    # https://github.com/bowenc0221/panoptic-deeplab/blob/cf8e20bbbf1cf540c7593434b965a93c4a889890/segmentation/utils/save_annotation.py#L210
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
    return colored_label


def main(args):
    cfg = setup(args)

    dataset_name = cfg.DATASETS.TEST[0]
    meta = MetadataCatalog.get(dataset_name)
    if "coco" in dataset_name or "cityscapes" in dataset_name:
        colormap = np.array(meta.stuff_colors + meta.thing_colors)
    else:
        colormap = np.array(meta.colors)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS
    )
    model.eval()

    transform = T.ResizeShortestEdge(
        cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
    )
    img_orig = np.array(Image.open(args.input_image))
    h, w, _ = img_orig.shape
    img = transform.get_transform(img_orig).apply_image(img_orig)
    model_input = [{
        "image": torch.from_numpy(img).permute(2, 0, 1),
        "width": w,
        "height": h
    }]
    out = model(model_input)[0]
    preds_color = panoptic_labels_to_color(
        out["panoptic_seg"][0].squeeze().cpu().numpy(),
        label_divisor=meta.label_divisor,
        colormap=colormap
    )
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img_orig)
    axarr[1].imshow(preds_color)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)
    main(args)
