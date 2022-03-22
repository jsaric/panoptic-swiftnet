import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.data import MetadataCatalog
from detectron2.modeling import (
    META_ARCH_REGISTRY,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances

from .backbone import resnet, densenet
from .postprocessing.postprocessing import get_panoptic_segmentation
from .psn_modules import PanopticSwiftNetDecoder, FastPanopticSwiftNetHead
import torchvision.transforms.functional as tvtf

__all__ = ["PanopticSwiftNet"]


@META_ARCH_REGISTRY.register()
class PanopticSwiftNet(nn.Module):
    """
    Main class for panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_psn_backbone(cfg)
        self.decoder = build_psn_decoder(cfg)
        self.sem_seg_head = FastPanopticSwiftNetHead(
            decoder_channels=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.DECODER_CHANNELS,
            num_classes=[cfg.MODEL.PANOPTIC_SWIFTNET.NUM_CLASSES],
            class_key=["sem_seg"]
        )
        self.ins_embed_head = FastPanopticSwiftNetHead(
            decoder_channels=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.INS_DECODER_CHANNELS,
            num_classes=[1, 2],
            class_key=["center", "offset"]
        )
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED
        self.semantic_loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.common_stride = cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.COMMON_STRIDE
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.num_classes = cfg.MODEL.PANOPTIC_SWIFTNET.NUM_CLASSES
        self.syncbn = cfg.MODEL.PANOPTIC_SWIFTNET.SYNCBN
        self.final_up_ac = cfg.MODEL.PANOPTIC_SWIFTNET.FINAL_UP_ALIGN_CORNERS
        self.trt_conversion = False

        if cfg.MODEL.PANOPTIC_SWIFTNET.SEMANTIC_LOSS.TYPE == "cross_entropy":
            self.semantic_criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
)
        elif cfg.MODEL.PANOPTIC_SWIFTNET.SEMANTIC_LOSS.TYPE == "hard_pixel_mining":
            self.semantic_criterion = DeepLabCE(ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE, top_k_percent_pixels=cfg.MODEL.PANOPTIC_SWIFTNET.SEMANTIC_LOSS.LOSS_TOP_K)
        else:
            raise ValueError("Unexpected loss type!")

        self.center_loss_weight = cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.CENTER_LOSS_WEIGHT
        self.offset_loss_weight = cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.OFFSET_LOSS_WEIGHT
        self.center_criterion = nn.MSELoss(reduction="none")
        self.offset_criterion = nn.L1Loss(reduction="none")
        if self.syncbn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        if not self.trt_conversion:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x / 255. - self.pixel_mean) / self.pixel_std for x in images]
            # To avoid error in ASPP layer when input has different size.
            size_divisibility = (
                self.size_divisibility
                if self.size_divisibility > 0
                else self.backbone.size_divisibility
            )
            images = ImageList.from_tensors(images, size_divisibility)
            features = self.backbone(images.tensor)
        else:
            features = self.backbone(batched_inputs)
        features = self.decoder(features)
        output = {**self.ins_embed_head(features["instance_features"]),
                  **self.sem_seg_head(features["semantic_features"])}

        if self.training:
            losses = {}
            if "sem_seg" in batched_inputs[0]:
                targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
                targets = ImageList.from_tensors(
                    targets, size_divisibility, self.ignore_value
                ).tensor
                if "sem_seg_weights" in batched_inputs[0]:
                    # The default D2 DatasetMapper may not contain "sem_seg_weights"
                    # Avoid error in testing when default DatasetMapper is used.
                    weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                    weights = ImageList.from_tensors(weights, size_divisibility).tensor
                else:
                    weights = None
            else:
                targets = None
                weights = None
            sem_seg_losses = self.semantic_losses(output["sem_seg"], targets, weights)
            losses.update(sem_seg_losses)

            if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
                center_targets = [x["center"].to(self.device) for x in batched_inputs]
                center_targets = ImageList.from_tensors(
                    center_targets, size_divisibility
                ).tensor.unsqueeze(1)
                center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
                center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor

                offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
                offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
                offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
                offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor
            else:
                center_targets = None
                center_weights = None

                offset_targets = None
                offset_weights = None
            center_losses = self.center_losses(output["center"], center_targets, center_weights)
            offset_losses = self.offset_losses(output["offset"], offset_targets, offset_weights)
            losses.update(center_losses)
            losses.update(offset_losses)
            return losses
        else:
            if self.benchmark_network_speed:
                return []

            for k, v in output.items():
                output[k] = F.interpolate(
                    v,
                    scale_factor=self.common_stride,
                    mode="bilinear",
                    align_corners=self.final_up_ac
                )
                if k == "offset":
                    output[k] *= self.common_stride
            if self.trt_conversion:
                return output
            sem_seg_results = output["sem_seg"]
            center_results = output["center"]
            offset_results = output["offset"]

            processed_results = []
            for sem_seg_result, center_result, offset_result, input_per_image, image_size in zip(
                sem_seg_results, center_results, offset_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height")
                width = input_per_image.get("width")

                h, w = image_size
                r = sem_seg_result[:, :h, :w]
                c = center_result[:, :h, :w]
                o = offset_result[:, :h, :w]

                with torch.no_grad():
                    panoptic_image, _ = get_panoptic_segmentation(
                        r.argmax(dim=0, keepdim=True),
                        c,
                        o,
                        thing_list=self.meta.thing_dataset_id_to_contiguous_id.values(),
                        label_divisor=self.meta.label_divisor,
                        stuff_area=self.stuff_area,
                        void_label=-1,
                        threshold=self.threshold,
                        nms_kernel=self.nms_kernel,
                        top_k=self.top_k,
                        num_classes=self.num_classes
                    )

                    panoptic_image = tvtf.resize(
                        panoptic_image.unsqueeze(0),
                        size=(height, width),
                        interpolation=tvtf.InterpolationMode.NEAREST).squeeze()

                # For semantic segmentation evaluation.
                r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                c = sem_seg_postprocess(center_result, image_size, height, width)
                o = sem_seg_postprocess(offset_result, image_size, height, width)
                processed_results.append({"sem_seg": r})
                panoptic_image = panoptic_image.squeeze(0)
                processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
                processed_results[-1]["centers"] = (c, None)
                processed_results[-1]["offsets"] = (o, None)

                # For instance segmentation evaluation.
                if self.predict_instances:
                    instances = []
                    panoptic_image_cpu = panoptic_image.cpu().numpy()
                    for panoptic_label in np.unique(panoptic_image_cpu):
                        if panoptic_label == -1:
                            continue
                        pred_class = panoptic_label // self.meta.label_divisor
                        isthing = pred_class in list(
                            self.meta.thing_dataset_id_to_contiguous_id.values()
                        )
                        # Get instance segmentation results.
                        if isthing:
                            instance = Instances((height, width))
                            # Evaluation code takes continuous id starting from 0
                            instance.pred_classes = torch.tensor(
                                [pred_class], device=panoptic_image.device
                            )
                            mask = panoptic_image == panoptic_label
                            instance.pred_masks = mask.unsqueeze(0)
                            # Average semantic probability
                            sem_scores = semantic_prob[pred_class, ...]
                            sem_scores = torch.mean(sem_scores[mask])
                            # Center point probability
                            mask_indices = torch.nonzero(mask).float()
                            center_y, center_x = (
                                torch.mean(mask_indices[:, 0]),
                                torch.mean(mask_indices[:, 1]),
                            )
                            center_scores = c[0, int(center_y.item()), int(center_x.item())]
                            # Confidence score is semantic prob * center prob.
                            instance.scores = torch.tensor(
                                [sem_scores * center_scores], device=panoptic_image.device
                            )
                            # Get bounding boxes
                            instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                            instances.append(instance)
                    if len(instances) > 0:
                        processed_results[-1]["instances"] = Instances.cat(instances)

            return processed_results

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.center_criterion(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.offset_criterion(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses

    def semantic_losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.semantic_criterion(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.semantic_loss_weight}
        return losses


def build_psn_backbone(cfg):
    if cfg.MODEL.BACKBONE.NAME == "resnet":
        # if cfg.MODEL.RESNETS.DEPTH == 18:
        return getattr(resnet, f"resnet{cfg.MODEL.RESNETS.DEPTH}")(
            pretrained=cfg.MODEL.PANOPTIC_SWIFTNET.PRETRAINED_BACKBONE,
            num_levels=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.NUM_LEVELS
        )
    elif "densenet" in cfg.MODEL.BACKBONE.NAME:
        return getattr(densenet, cfg.MODEL.BACKBONE.NAME)(
            pretrained=cfg.MODEL.PANOPTIC_SWIFTNET.PRETRAINED_BACKBONE,
            num_levels=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.NUM_LEVELS,
            memory_efficient=cfg.MODEL.PANOPTIC_SWIFTNET.MEMORY_EFFICIENT
        )


def build_psn_decoder(cfg):
    return PanopticSwiftNetDecoder(
        type=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.TYPE,
        num_levels=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.NUM_LEVELS,
        in_channels=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.IN_CHANNELS,
        feature_key=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.FEATURE_KEY,
        low_level_channels=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.LOW_LEVEL_CHANNELS,
        low_level_key=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.LOW_LEVEL_KEY,
        decoder_channels=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.DECODER_CHANNELS,
        ins_decoder_channesl=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.INS_DECODER_CHANNELS,
        use_bn=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.USE_BN,
        spp_grids=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.SPP_GRIDS,
        spp_square_grid=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.SPP_SQUARE_GRID,
        spp_drop_rate=cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.SPP_DROP_RATE
    )