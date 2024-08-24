from torch import nn
from mmdet3d.registry import MODELS
from .resnet import ResNetBlock


@MODELS.register_module()
class KPPillarsBEVBackbone(nn.Module):

    def __init__(self, in_channels, out_channels, layers):
        super(KPPillarsBEVBackbone, self).__init__()
        self.blocks_s0 = self.create_blocks(in_channels[0], out_channels[0], layers[0])
        self.blocks_s1 = self.create_blocks(in_channels[1], out_channels[1], layers[1])
        self.blocks_s2 = self.create_blocks(in_channels[2], out_channels[2], layers[2])
        self.blocks_s3 = self.create_blocks(in_channels[3], out_channels[3], layers[3])
        # 1x1 conv
        self.conv = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1)

    @staticmethod
    def create_blocks(in_channels, out_channels, num_layers):
        layers = []
        for i in range(num_layers - 1):
            block = ResNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1
            )
            layers.append(block)
        downsample_block = ResNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )
        layers.append(downsample_block)
        return nn.Sequential(*layers)

    def forward(self, grids):
        grids_s0 = grids[0]
        grids_s1 = grids[1]
        grids_s2 = grids[2]
        grids_s3 = grids[3]

        stem = grids_s0.clone()
        stem += grids_s0
        s0_out = self.blocks_s0(stem)

        stem_s1 = s0_out + grids_s1
        s1_out = self.blocks_s1(stem_s1)

        stem_s2 = s1_out + grids_s2
        s2_out = self.blocks_s2(stem_s2)

        stem_s3 = s2_out + grids_s3
        s3_out = self.blocks_s3(stem_s3)

        output = self.conv(s3_out)
        return [s0_out, s1_out, s2_out, output]