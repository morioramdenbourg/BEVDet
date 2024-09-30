import json
import torch

from .centerpoint import CenterPoint
from ..builder import DETECTORS


@DETECTORS.register_module()
class SimpleModel(CenterPoint):

    def __init__(self, results_dir, **kwargs):
        super(SimpleModel, self).__init__(**kwargs)
        self.results_dir = results_dir

    def extract_pts_feat(self, pts, img_feats, img_metas):
        if not self.with_pts_bbox:
            return None

        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = torch.mean(voxels, dim=1)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
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
        return losses

    def write_bboxes_to_files(self, gt_bboxes_3d, gt_labels_3d, img_metas):
        for index, meta in enumerate(img_metas):
            sample_token = meta['sample_idx']
            print(f'Writing bboxes to {self.results_dir} for sample {sample_token}')
            # write gt to file (in lidar)
            sample_gt_bboxes_3d = gt_bboxes_3d[index]
            sample_gt_labels_3d = gt_labels_3d[index]
            sample_gt_file_name = self.results_dir + '/' + sample_token + '_gt.pth'
            torch.save((sample_gt_bboxes_3d, sample_gt_labels_3d), sample_gt_file_name)

    def write_points_to_file(self, pts, img_metas):
        for index, meta in enumerate(img_metas):
            sample_token = meta['sample_idx']
            print(f'Writing pts to {self.results_dir} for sample {sample_token}')
            meta = {
                'use_lidar': True,
                'sample_token': sample_token,
            }
            # write meta to file
            sample_meta_file_name = self.results_dir + '/' + sample_token + '_meta.json'
            with open(sample_meta_file_name, 'w') as meta_file:
                json.dump(meta, meta_file)
            # write points to file (in lidar)
            sample_points = pts[index]
            sample_points_file_name = self.results_dir + '/' + sample_token + '_points.pth'
            torch.save(sample_points, sample_points_file_name)
