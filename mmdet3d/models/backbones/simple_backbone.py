from .resnet import CustomResNet
from torch import nn
from ..builder import BACKBONES

@BACKBONES.register_module()
class SimpleBackbone(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimpleBackbone, self).__init__()

        self.block = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_map):
        forward = self.block(feature_map)
        return forward
