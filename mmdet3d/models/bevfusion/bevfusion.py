from copy import deepcopy

import numpy as np
import torch
from mmengine.structures import InstanceData
from torch import nn

from torch.nn import functional as F

from .det_data_sample import Det3DDataSample
from ..detectors.centerpoint import CenterPoint
from ..builder import DETECTORS
from .. import builder
from mmcv.ops import Voxelization


@DETECTORS.register_module()
class BEVFusion(CenterPoint):

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 img_backbone=None,
                 img_neck=None,
                 view_transform=None,
                 fusion_layer=None,
                 pts_bbox_head=None,
                 init_cfg=None,
                 **kwargs):
        super(BEVFusion, self).__init__(**kwargs)

        self.voxelize_reduce = True
        self.pts_voxel_layer = Voxelization(**pts_voxel_layer) if pts_voxel_layer is not None else None
        self.img_backbone = builder.build_backbone(img_backbone) if img_backbone is not None else None
        self.img_neck = builder.build_neck(img_neck) if img_neck is not None else None
        self.view_transform = builder.build_fusion_layer(view_transform) if view_transform is not None else None
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder) if pts_middle_encoder is not None else None
        self.fusion_layer = builder.build_fusion_layer(fusion_layer) if fusion_layer is not None else None
        self.pts_backbone = builder.build_backbone(pts_backbone) if pts_backbone is not None else None
        self.pts_neck = builder.build_neck(pts_neck) if pts_neck is not None else None
        self.pts_bbox_head = builder.build_head(pts_bbox_head) if pts_bbox_head is not None else None
        self.adaptive_pool = nn.AdaptiveAvgPool2d((256, 704))

        self.init_weights()

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    def extract_img_feat(self,
                         x,
                         points,
                         lidar2image,
                         camera_intrinsics,
                         camera2lidar,
                         img_aug_matrix,
                         lidar_aug_matrix,
                         img_metas):
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        x = x[0]
        x = self.adaptive_pool(x)

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        # with torch.autocast(device_type='cuda', dtype=torch.float32):
        x = self.view_transform(
            x,
            points,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_pts_feat(self, pts, img_feats, img_metas) -> torch.Tensor:
        # with torch.autocast('cuda', enabled=False):
        # points = [point.float() for point in points]
        feats, coords, sizes = self.voxelize(pts)
        batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    def extract_feat(self, points, img, img_metas):
        features = []
        if img is not None:
            img = img.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(img_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.stack([np.eye(4) for _ in range(6)])))
                lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = img.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = img.new_tensor(np.array(camera_intrinsics))
            camera2lidar = img.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = img.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = img.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(img,
                                                deepcopy(points),
                                                lidar2image,
                                                camera_intrinsics,
                                                camera2lidar,
                                                img_aug_matrix,
                                                lidar_aug_matrix,
                                                img_metas)
            features.append(img_feature)

        # pts_feature = self.extract_pts_feat(points, None, img_metas)
        # features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            # Only points
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        # Img Features, Pts Features
        return None, x

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):

        data_samples = []
        for i in range(len(gt_bboxes_3d)):
            instance = InstanceData()
            instance.bboxes_3d = gt_bboxes_3d[i]
            instance.labels_3d = gt_labels_3d[i]
            data_sample = Det3DDataSample(instance.metainfo, instance)
            data_samples.append(data_sample)
        losses = self.pts_bbox_head.loss(pts_feats, data_samples)
        return losses

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes