# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN


def add_panoptic_swiftnet_config(cfg):
    """
    Add config for panoptic-swiftnet.
    """
    # Target generation parameters.
    cfg.INPUT.GAUSSIAN_SIGMA = 10
    cfg.INPUT.IGNORE_STUFF_IN_OFFSET = True
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC = False
    cfg.INPUT.TRT_CROP_H = 1152
    cfg.INPUT.TRT_CROP_W = 1920
    cfg.INPUT.CLASS_BALANCED_CROPS = False
    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "ADAM"
    # Panoptic-DeepLab semantic segmentation head.
    # We add an extra convolution before predictor.
    # cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K = 0.2
    # # Panoptic-DeepLab instance segmentation head.
    # cfg.MODEL.INS_EMBED_HEAD = CN()
    # cfg.MODEL.INS_EMBED_HEAD.NAME = "FastPanopticSwiftNetHead"
    # cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
    # cfg.MODEL.INS_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    # cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    #
    # # We add an extra convolution before predictor.
    # cfg.MODEL.INS_EMBED_HEAD.NORM = "SyncBN"
    # cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT = 200.0
    # cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT = 0.01
    # Panoptic-DeepLab post-processing setting.
    cfg.MODEL.PANOPTIC_SWIFTNET = CN()
    cfg.MODEL.PANOPTIC_SWIFTNET.PRETRAINED_BACKBONE = True
    cfg.MODEL.PANOPTIC_SWIFTNET.SYNCBN = False
    cfg.MODEL.PANOPTIC_SWIFTNET.MEMORY_EFFICIENT = False

    # Stuff area limit, ignore stuff region below this number.
    cfg.MODEL.PANOPTIC_SWIFTNET.STUFF_AREA = 2048
    cfg.MODEL.PANOPTIC_SWIFTNET.CENTER_THRESHOLD = 0.1
    cfg.MODEL.PANOPTIC_SWIFTNET.NMS_KERNEL = 7
    cfg.MODEL.PANOPTIC_SWIFTNET.TOP_K_INSTANCE = 200
    # If set to False, Panoptic-DeepLab will not evaluate instance segmentation.
    cfg.MODEL.PANOPTIC_SWIFTNET.PREDICT_INSTANCES = True
    # This is the padding parameter for images with various sizes. ASPP layers
    # requires input images to be divisible by the average pooling size and we
    # can use `MODEL.PANOPTIC_SWIFTNET.SIZE_DIVISIBILITY` to pad all images to
    # a fixed resolution (e.g. 640x640 for COCO) to avoid having a image size
    # that is not divisible by ASPP average pooling size.
    cfg.MODEL.PANOPTIC_SWIFTNET.SIZE_DIVISIBILITY = -1
    # Only evaluates network speed (ignores post-processing).
    cfg.MODEL.PANOPTIC_SWIFTNET.BENCHMARK_NETWORK_SPEED = False
    cfg.MODEL.PANOPTIC_SWIFTNET.FINAL_UP_ALIGN_CORNERS = False
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER = CN()
    # SHARED OR SEPARATE (FOR SEMSEG AND INSTANCE BRANCH)
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.TYPE = "SHARED"
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.NUM_LEVELS = 3
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.IN_CHANNELS = 512
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.FEATURE_KEY = "res5_2"
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.LOW_LEVEL_CHANNELS = [[512, 256],
                                                              [512, 256, 128],
                                                              [256, 128, 64],
                                                              [128, 64],
                                                              64]
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.LOW_LEVEL_KEY = [["res5o_1", "res4o_2"],
                                                         ["res5o_0", "res4o_1", "res3o_2"],
                                                         ["res4o_0", "res3o_1", "res2o_2"],
                                                         ["res3o_0", "res2o_1"],
                                                         "res2o_0"]
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.DECODER_CHANNELS = 256
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.INS_DECODER_CHANNELS = 256
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.USE_BN = True
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.SPP_GRIDS = (8, 4, 2, 1),
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.SPP_SQUARE_GRID = False,
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.SPP_DROP_RATE = 0.0
    cfg.MODEL.PANOPTIC_SWIFTNET.DECODER.COMMON_STRIDE = 4

    cfg.MODEL.PANOPTIC_SWIFTNET.NUM_CLASSES = 133

    cfg.MODEL.PANOPTIC_SWIFTNET.SEMANTIC_LOSS = CN()
    cfg.MODEL.PANOPTIC_SWIFTNET.SEMANTIC_LOSS.TYPE = "hard_pixel_mining"
    cfg.MODEL.PANOPTIC_SWIFTNET.SEMANTIC_LOSS.LOSS_TOP_K = 1.0

    cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS = CN()
    cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.CENTER_LOSS_WEIGHT = 200.0
    cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.OFFSET_LOSS_WEIGHT = 0.01
    cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.BAOL = CN()
    cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.BAOL.ENABLED = False
    cfg.MODEL.PANOPTIC_SWIFTNET.INSTANCE_LOSS.BAOL.WEIGHTS_PATH = ""