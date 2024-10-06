from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from mmcv.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from ..detectors.centerpoint import CenterPoint
from ..builder import DETECTORS
from .. import builder
# from .ops import Voxelization
from mmcv.ops import Voxelization


@DETECTORS.register_module()
class BEVFusion(CenterPoint):

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 view_transform=None,
                 fusion_layer=None,
                 bbox_head=None,
                 init_cfg=None,
                 **kwargs):
        super(BEVFusion, self).__init__(**kwargs)

        self.voxelize_reduce = True
        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        # self.img_backbone = builder.build_backbone(img_backbone) if img_backbone is not None else None
        # self.img_neck = builder.build_neck(img_neck) if img_neck is not None else None
        # self.view_transform = builder.build_fusion_layer(view_transform) if view_transform is not None else None
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        # self.fusion_layer = builder.build_fusion_layer(fusion_layer) if fusion_layer is not None else None
        self.pts_backbone = builder.build_backbone(pts_backbone)
        self.pts_neck = builder.build_neck(pts_neck)
        self.bbox_head = builder.build_head(bbox_head)

        self.init_weights()

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
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

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return None, x


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