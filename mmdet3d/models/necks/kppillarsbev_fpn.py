import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from ..builder import NECKS
from .. import builder


@NECKS.register_module()
class KPPillarsBEVFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factors,
                 input_feature_index,
                 resnet_cfg,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.input_feature_index = input_feature_index

        # Upsample layers
        self.upsample_layers = nn.ModuleList()
        for scale_factor in scale_factors:
            up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.upsample_layers.append(up)

        # lateral layers
        self.lateral_layers = nn.ModuleList()
        for in_ch in in_channels:
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
            )
            self.lateral_layers.append(lateral_conv)

        # resnset layers between each pyramid layer
        self.resnet_layers = nn.ModuleList()
        for resnet_layer in resnet_cfg:
            self.resnet_layers.append(builder.build_backbone(resnet_layer))

        # Final Conv Layer (tailored to task)
        self.conv = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        fused_features = feats[self.input_feature_index[0]]
        for i, idx in enumerate(self.input_feature_index):
            feat = feats[idx]

            # lateral layer to reduce channel dimension
            lateral_feat = self.lateral_layers[i](feat)

            # upsample
            upsample = self.upsample_layers[i](lateral_feat)

            # element-wise addition (TODO: concat instead?)
            fused_features = fused_features + upsample

            # resnet
            fused_features = self.resnet_layers[i](fused_features)[0]

        # final conv layer
        fused_features = self.conv(fused_features)
        return fused_features