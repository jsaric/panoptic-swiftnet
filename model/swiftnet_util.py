import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=True)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 groups=-1):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0, groups=1, separable=False, bn_class=nn.BatchNorm2d):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', bn_class(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d
        warnings.warn(f'Using conv type {k}x{k}: {conv_class}')
        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation, groups=groups))
        if drop_rate > 0:
            warnings.warn(f'Using dropout with p: {drop_rate}')
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))


class _ConvBNReLu(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_ConvBNReLu, self).__init__()
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                          dilation=dilation))
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_out, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))


class _ConvBNReLuConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_ConvBNReLuConv, self).__init__()
        padding = k // 2
        self.add_module('conv1', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation))
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_out, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_maps_out, num_maps_out, kernel_size=1, padding=0, bias=bias,
                                           dilation=dilation))


class Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True,
                 bn_class=nn.BatchNorm2d):
        super(Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        if isinstance(skip_maps_in, int):
            self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in,
                                          k=1,
                                          batch_norm=use_bn and bneck_starts_with_bn,
                                          bn_class=bn_class)
        else:
            self.bottleneck = nn.ModuleList([_BNReluConv(maps, num_maps_in,
                                                         k=1,
                                                         batch_norm=use_bn and bneck_starts_with_bn,
                                                         bn_class=bn_class)
                                             for maps in skip_maps_in])
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, separable=separable,
                                      bn_class=bn_class)
        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        warnings.warn(f'\tUsing skips: {self.use_skip} (only skips: {self.only_skip})', UserWarning)
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='bilinear', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)

    def forward(self, x, skip):
        if isinstance(skip, torch.Tensor):
            skip = self.bottleneck.forward(skip)
        else:
            skip = sum([self.bottleneck[i].forward(s) for i, s in enumerate(skip)])
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True, bn_class=nn.BatchNorm2d):
        super(SpatialPyramidPooling, self).__init__()
        self.fixed_size = fixed_size
        self.grids = grids
        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))
        self.square_grid = square_grid
        self.upsampling_method = upsample
        if self.fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='bilinear', size=fixed_size)
            self.square_grid = True
            warnings.warn(f'Fixed upsample size', UserWarning)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum,
                                                  batch_norm=use_bn and starts_with_bn, bn_class=bn_class))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                            drop_rate=drop_rate, bn_class=bn_class))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                        bn_class=bn_class))

    def forward(self, x):
        levels = []
        target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = self.upsampling_method(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class MultiLevelNorm(nn.Module):
    def __init__(self, norm_layer, num_levels, *args, **kwargs):
        super(MultiLevelNorm, self).__init__()
        self.norms = nn.ModuleList([norm_layer(*args, **kwargs) for _ in range(num_levels)])
        self.level = 1000

    def forward(self, x):
        return self.norms[self.level].forward(x)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(MultiLevelNorm, self)._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys,
                                                          unexpected_keys, error_msgs)
        for norm in self.norms:
            norm._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                       error_msgs)
