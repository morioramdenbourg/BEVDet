import torch
from mmcv.runner import ModuleList
from mmdet.models import DETECTORS
from torch import nn
from mmcv.ops import Voxelization
from .bevdet import BEVDet
from .centerpoint import CenterPoint
from .. import builder

@DETECTORS.register_module()
class BEVDetKPPillarsBEV(CenterPoint):

    def __init__(self,
                 pts_voxel_layers,
                 pts_voxel_encoders,
                 pts_middle_encoders,
                 pts_backbones,
                 img_backbone=None,
                 img_neck=None,
                 img_view_transformer=None,
                 fusion_norm=None,
                 fusion_layer=None,
                 fusion_encoder_backbone=None,
                 fusion_encoder_neck=None,
                 **kwargs):
        super(BEVDetKPPillarsBEV, self).__init__(**kwargs)
        self.img_backbone = builder.build_backbone(img_backbone) if img_backbone is not None else None
        self.img_neck = builder.build_neck(img_neck) if img_neck is not None else None

        self.fuser = builder.build_fusion_layer(fusion_layer)
        self.fusion_encoder_backbone = builder.build_backbone(fusion_encoder_backbone) if fusion_encoder_backbone is not None else None
        self.fusion_encoder_neck = builder.build_neck(fusion_encoder_neck) if fusion_encoder_neck is not None else None

        self.img_view_transformer = builder.build_neck(img_view_transformer) if img_view_transformer is not None else None

        fusion_norm_radar = fusion_norm['radar']
        fusion_norm_rgb = fusion_norm['rgb']
        self.radar_norm = nn.LayerNorm(fusion_norm_radar['normalized_shape'], elementwise_affine=fusion_norm_radar['elementwise_affine'])
        self.rgb_norm = nn.LayerNorm(fusion_norm_rgb['normalized_shape'], elementwise_affine=fusion_norm_rgb['elementwise_affine'])

        self.pts_voxel_layers = ModuleList()
        for voxel_layer in pts_voxel_layers:
            voxelization = Voxelization(**voxel_layer)
            self.pts_voxel_layers.append(voxelization)

        self.pts_voxel_encoders = ModuleList()
        for voxel_encoder in pts_voxel_encoders:
            encoder = builder.build_voxel_encoder(voxel_encoder)
            self.pts_voxel_encoders.append(encoder)

        self.pts_middle_encoders = ModuleList()
        for middle_encoder in pts_middle_encoders:
            encoder = builder.build_middle_encoder(middle_encoder)
            self.pts_middle_encoders.append(encoder)

        self.pts_backbones = ModuleList()
        for backbone in pts_backbones:
            self.pts_backbones.append(builder.build_backbone(backbone))

        self.num_layers = len(pts_voxel_layers)

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

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

    def prepare_inputs(self, inputs):
        # split the inputs into eac h frame
        assert len(inputs) == 6
        imgs, sensor2lidar, intrins, post_rots, post_trans, bda = inputs
        return [imgs, sensor2lidar, intrins, post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, **kwargs):
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])

        feats, depth = self.img_view_transformer([x] + img[1:6])
        return feats, depth

    def extract_pts_feat(self, pts, img_metas, **kwargs):
        grids = []
        for index in range(self.num_layers):
            x = self.kpbev(pts, index)
            grids.append(x[0])

        x = self.pts_neck(grids)

        return x

    def extract_feat(self, points, img, img_metas, **kwargs):
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points, img_metas, **kwargs)

        # Normalize the rgb and radar paths to a similar scale
        img_feats = self.rgb_norm(img_feats)
        pts_feats = self.radar_norm(pts_feats)

        feats = self.fuser([img_feats, pts_feats])
        feats = self.fusion_encoder_backbone(feats)
        feats = self.fusion_encoder_neck(feats)
        return feats

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        feats = self.extract_feat(points, img_inputs, img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train([feats],
                                            gt_bboxes_3d,
                                            gt_labels_3d,
                                            img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        feats = self.extract_feat(points, img, img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts([feats], img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
