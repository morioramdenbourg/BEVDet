import torch
from mmcv.cnn import build_norm_layer
from torch import nn
from ..builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class BEVDetFuser(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_cfg=None):
        self.in_channels = sum(in_channels)
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, self.out_channels)[1],
            nn.ReLU(),
        )

    def forward(self, inputs):
        return super().forward(torch.cat(inputs, dim=1))