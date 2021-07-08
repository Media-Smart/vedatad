import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from vedacore.misc import registry
from vedacore.modules import ConvModule, constant_init, kaiming_init


@registry.register_module('neck')
class TDM(nn.Module):
    """Temporal Down-Sampling Module."""

    def __init__(self,
                 in_channels,
                 stage_layers=(1, 1, 1, 1),
                 kernel_sizes=3,
                 strides=2,
                 paddings=1,
                 dilations=1,
                 out_channels=256,
                 conv_cfg=dict(typename='Conv1d'),
                 norm_cfg=dict(typename='BN1d'),
                 act_cfg=dict(typename='ReLU'),
                 out_indices=(0, 1, 2, 3, 4)):
        super(TDM, self).__init__()

        self.in_channels = in_channels
        self.num_stages = len(stage_layers)
        self.stage_layers = stage_layers
        self.kernel_sizes = _ntuple(self.num_stages)(kernel_sizes)
        self.strides = _ntuple(self.num_stages)(strides)
        self.paddings = _ntuple(self.num_stages)(paddings)
        self.dilations = _ntuple(self.num_stages)(dilations)
        self.out_channels = _ntuple(self.num_stages)(out_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_indices = out_indices

        assert (len(self.stage_layers) == len(self.kernel_sizes) == len(
            self.strides) == len(self.paddings) == len(self.dilations) == len(
                self.out_channels))

        self.td_layers = []
        for i in range(self.num_stages):
            td_layer = self.make_td_layer(self.stage_layers[i], in_channels,
                                          self.out_channels[i],
                                          self.kernel_sizes[i],
                                          self.strides[i], self.paddings[i],
                                          self.dilations[i], self.conv_cfg,
                                          self.norm_cfg, self.act_cfg)
            in_channels = self.out_channels[i]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, td_layer)
            self.td_layers.append(layer_name)

    @staticmethod
    def make_td_layer(num_layer, in_channels, out_channels, kernel_size,
                      stride, padding, dilation, conv_cfg, norm_cfg, act_cfg):
        layers = []
        layers.append(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        for _ in range(1, num_layer):
            layers.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initiate the parameters."""
        for m in self.modules():
            if isinstance(m, _ConvNd):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)

        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.td_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if (i + 1) in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
