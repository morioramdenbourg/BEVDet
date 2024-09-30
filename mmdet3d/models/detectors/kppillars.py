import random

import torch
from torch import nn

from .centerpoint import CenterPoint
from ..builder import DETECTORS
from ...ops.kpconv.models.blocks import SimpleBlock
from ...utils.file_logger import write_bboxes_to_files, write_points_to_file


@DETECTORS.register_module()
class KPPillars(CenterPoint):

    def __init__(self, pts_preprocessor, **kwargs):
        super(KPPillars, self).__init__(**kwargs)

        self.pts_preprocessors = nn.ModuleList()
        for processor in pts_preprocessor:
            block = SimpleBlock(**processor)
            self.pts_preprocessors.append(block)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        pts_coords = [p[:, :3] for p in pts]
        x = [p[:, 3:] for p in pts]

        for preprocessor in self.pts_preprocessors:
            x = preprocessor(pts_coords, x)

        aug_pts = [torch.cat([c, f], dim=1) for c, f in zip(pts_coords, x)]

        self.r = random.randint(1, 100)
        if self.r == 1:
            write_points_to_file('./vis/kppillarsbev_nus', aug_pts, img_metas)

        voxels, num_points, coors = self.voxelize(aug_pts)
        voxel_features = voxels[:, :, 3:]
        voxel_features = torch.mean(voxel_features, dim=1)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return [x]

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):

        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        if self.r == 1:
            write_bboxes_to_files('./vis/kppillarsbev_nus', gt_bboxes_3d, gt_labels_3d, img_metas, losses)

        return losses

