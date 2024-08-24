import json
import os
import time

import numpy as np
import torch
from mmcv.ops import Voxelization

from mmdet3d.models import CenterPoint
from mmdet3d.models.builder import DETECTORS, build_backbone, build_middle_encoder, build_head
from mmdet3d.models.detectors import Base3DDetector
from pyquaternion import Quaternion

# class OutputStore:
#
#     GRIDS = 'grids'
#     BACKBONES = 'backbones'
#     PREDICTIONS = 'predictions'
#
#     def __init__(self, enabled):
#         self.enabled = enabled
#         self.store = {}
#
#     def check_enabled(self):
#         if not self.enabled:
#             raise Exception('OutputStore is not enabled')
#
#     def put_output(self, name, output):
#         self.check_enabled()
#         self.store[name] = output
#
#     def get_output(self, name):
#         self.check_enabled()
#
#         if name in self.store:
#             return self.store[name]
#         else:
#             raise Exception(f'{name} is not in store')


@DETECTORS.register_module()
class SimpleModel(CenterPoint):

    def __init__(self, results_dir, **kwargs):
        super(SimpleModel, self).__init__(**kwargs)
        self.results_dir = results_dir

    def extract_feat(self, imgs, batch_points, batch_img_metas):
        batch_size = len(batch_img_metas)
        grids = []
        for index in range(batch_size):
            points = batch_points[index]
            voxels = self.voxelize(points)

            voxel_features = voxels[0]
            voxel_coords = voxels[1]
            sample_id_column = torch.full((voxel_coords.shape[0], 1), index, device=voxel_coords.device)
            voxel_coords_with_sample_id = torch.cat([sample_id_column, voxel_coords], dim=1)
            voxel_features_pooled = torch.mean(voxel_features, dim=1)
            grid = self.scatter(voxel_features_pooled, voxel_coords_with_sample_id)

            grids.append(grid)

        grids_stacked = torch.cat(grids, dim=0)
        feats = self.backbone(grids_stacked)
        return feats

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        feats = self.extract_feat(None, points, img_metas)
        forward = self.bbox_head([feats])
        predictions = self.bbox_head.get_bboxes(forward, img_metas)
        loss = self.bbox_head.loss(gt_bboxes_3d, gt_labels_3d, forward)

        if True and loss['task0.loss_heatmap'] < 500:
           for index, meta in enumerate(img_metas):
                sample_token = meta['sample_idx']

                print(f'Writing to {self.results_dir} for sample {sample_token}')

                # write meta to file
                meta = {
                    'use_lidar': True,
                    'sample_token': sample_token,
                }
                sample_meta_file_name = self.results_dir + '/' + sample_token + '_meta.json'
                with open(sample_meta_file_name, 'w') as meta_file:
                    json.dump(meta, meta_file)

                # write points to file (in lidar)
                sample_points = points[index]
                sample_points_file_name = self.results_dir + '/' + sample_token + '_points.pth'
                torch.save(sample_points, sample_points_file_name)

                # write gt to file (in lidar?)
                sample_gt_bboxes_3d = gt_bboxes_3d[index]
                sample_gt_labels_3d = gt_labels_3d[index]
                sample_gt_file_name = self.results_dir + '/' + sample_token + '_gt.pth'
                torch.save((sample_gt_bboxes_3d, sample_gt_labels_3d), sample_gt_file_name)

                # write feature map to file
                feature_map = feats[index]
                feature_map_file_name = self.results_dir + '/' + sample_token + '_feature_map.pth'
                torch.save(feature_map, feature_map_file_name)

                # write predictions to file (in lidar)
                pred = predictions[index]
                pred_file_name = self.results_dir + '/' + sample_token + '_predictions.pth'
                torch.save(pred, pred_file_name)

        return loss

    def simple_test(self, img, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    @torch.no_grad()
    def voxelize(self, points):
        voxels = self.voxelization(points)
        return voxels