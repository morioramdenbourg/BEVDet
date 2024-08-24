from mmdet.models import FPN
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class KPPillarsBEVNeck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(KPPillarsBEVNeck, self).__init__()
        self.fpn = FPN(in_channels,
                       out_channels,
                       num_outs=len(in_channels),
                       add_extra_convs=False,
                       relu_before_extra_convs=False,
                       no_norm_on_lateral=False)

    def forward(self, grids):
        return self.fpn(grids)
