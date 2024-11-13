import random

import torch
from mmcv.ops import Voxelization
from torch import nn

from .centerpoint import CenterPoint
from ..builder import DETECTORS
from .. import builder
from ...ops.kpconv.models.blocks import SimpleBlock
from ...utils.file_logger import write_bboxes_to_files, write_points_to_file


@DETECTORS.register_module()
class KPPillarsBEV(CenterPoint):

    def __init__(self, pts_preprocessors, pts_voxel_layers, pts_voxel_encoders, pts_middle_encoders, pts_backbones, **kwargs):
        super(KPPillarsBEV, self).__init__(**kwargs)

        self.pts_voxel_layers = nn.ModuleList()
        for voxel_layer in pts_voxel_layers:
            voxelization = Voxelization(**voxel_layer)
            self.pts_voxel_layers.append(voxelization)

        self.pts_voxel_encoders = nn.ModuleList()
        for voxel_encoder in pts_voxel_encoders:
            encoder = builder.build_voxel_encoder(voxel_encoder)
            self.pts_voxel_encoders.append(encoder)

        self.pts_middle_encoders = nn.ModuleList()
        for middle_encoder in pts_middle_encoders:
            encoder = builder.build_middle_encoder(middle_encoder)
            self.pts_middle_encoders.append(encoder)

        self.pts_preprocessors = nn.ModuleList()
        for processor in pts_preprocessors:
            block = SimpleBlock(**processor)
            self.pts_preprocessors.append(block)

        self.pts_backbones = nn.ModuleList()
        for backbone in pts_backbones:
            self.pts_backbones.append(builder.build_backbone(backbone))

        self.num_layers = len(pts_voxel_layers)

    def kpbev(self, aug_pts, index):
        # voxelization
        self.pts_voxel_layer = self.pts_voxel_layers[index]
        voxels, num_points, coors = self.voxelize(aug_pts)

        # kpbev
        anchors, x = self.pts_voxel_encoders[index](voxels, coors, num_points, aug_pts)

        # scatter
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoders[index](x, coors, batch_size)

        # resnet blocks
        x = self.pts_backbones[index](x)

        return x

    def extract_pts_feat(self, pts, img_feats, img_metas):
        pts_coords = [p[:, :3] for p in pts]
        x = [p[:, 3:] for p in pts]
        q_batches = [p.shape[0] for p in pts_coords]

        pts_coords = torch.cat(pts_coords, dim=0)
        x = torch.cat(x, dim=0)

        aug_pts = torch.cat([pts_coords, x], dim=1)
        aug_pts = aug_pts.split(q_batches, dim=0)

        grids = []
        for index in range(self.num_layers):
            x = self.kpbev(aug_pts, index)
            grids.append(x[0])

        # FPN
        x = self.pts_neck(grids)

        return [x]
