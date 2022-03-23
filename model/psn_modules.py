# ------------------------------------------------------------------------------
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from torch import nn
from .swiftnet_util import SpatialPyramidPooling as SPP, Upsample, Identity, _BNReluConv

__all__ = ["PanopticSwiftNetDecoder"]


class SinglePanopticSwiftNetDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key,
                 decoder_channels, use_bn=True, spp_grids=(8, 4, 2, 1), spp_square_grid=False,
                 spp_drop_rate=0.0, spp_class=SPP, num_levels=0, **kwargs):
        super(SinglePanopticSwiftNetDecoder, self).__init__()

        if num_levels > 0:
            spp_class = Identity
            bottleneck = _BNReluConv
        else:
            bottleneck = Identity

        num_levels = 3
        self.spp_size = kwargs.get('spp_size', decoder_channels)
        bt_size = self.spp_size
        level_size = self.spp_size // num_levels

        self.spp = spp_class(in_channels, num_levels, bt_size=bt_size, level_size=level_size, out_size=decoder_channels,
                             grids=spp_grids, square_grid=spp_square_grid, bn_momentum=0.01 / 2, use_bn=use_bn,
                             drop_rate=spp_drop_rate)
        self.bottleneck = bottleneck(in_channels, decoder_channels, k=1, batch_norm=False)

        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        self.low_level_key = low_level_key

        # Transform low-level feature
        upsamples = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            upsamples += [Upsample(decoder_channels, low_level_channels[i], decoder_channels, use_bn=use_bn)]
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, features):
        x = features[self.feature_key]
        x = self.bottleneck(x)
        x = self.spp(x)

        # build decoder
        for i in range(self.decoder_stage):
            k = self.low_level_key[i]
            iterable = isinstance(k, list) or isinstance(k, tuple)
            l = features[k] if not iterable else [features[ki] for ki in k]
            x = self.upsamples[i](x, l)

        return x


class FastPanopticSwiftNetHead(nn.Module):
    def __init__(self, decoder_channels, num_classes, class_key, *args, **kwargs):
        super().__init__()
        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels, num_classes[i], 1)
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x, semantic=None):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)
        return pred


class PanopticSwiftNetDecoder(nn.Module):
    def __init__(self, type, in_channels, feature_key, low_level_channels, low_level_key,
                 decoder_channels, num_levels, **kwargs):
        super(PanopticSwiftNetDecoder, self).__init__()
        # Build semantic decoder
        self.semantic_decoder = SinglePanopticSwiftNetDecoder(in_channels, feature_key, low_level_channels,
                                                              low_level_key, decoder_channels, num_levels=num_levels)

        # Build instance decoder
        self.instance_decoder = None
        if type == "separate":
            self.instance_decoder = SinglePanopticSwiftNetDecoder(in_channels, feature_key, low_level_channels,
                                                                  low_level_key, kwargs.get("INS_DECODER_CHANNELS"), num_levels=num_levels)

    def forward(self, features):
        ret = OrderedDict()
        # Semantic branch
        decoded_sem_features = decoded_inst_features = self.semantic_decoder(features)
        if self.instance_decoder is not None:
            decoded_inst_features = self.instance_decoder(features)
        ret["semantic_features"] = decoded_sem_features
        ret["instance_features"] = decoded_inst_features
        return ret
