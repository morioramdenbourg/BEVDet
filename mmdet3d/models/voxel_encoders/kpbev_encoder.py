import torch
from mmcv.cnn import build_norm_layer
from torch import nn
from ..builder import VOXEL_ENCODERS
from ...ops.kpconv.models.blocks import SimpleBlock


@VOXEL_ENCODERS.register_module()
class KPBEVEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 point_cloud_range,
                 voxel_size,
                 norm_cfg,
                 kpconv_args):
        super(KPBEVEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # z y x format
        self.coors_range_min = point_cloud_range[:3]
        self.voxel_size = voxel_size

        linear_1_norm = build_norm_layer(norm_cfg, self.out_channels)
        linear_2_norm = build_norm_layer(norm_cfg, self.out_channels)

        self.kpconv = SimpleBlock(**kpconv_args)
        self.linear_1 = nn.Sequential(nn.Linear(self.in_channels + 7, self.out_channels),
                                      linear_1_norm[1],
                                      nn.ReLU())
        self.linear_2 = nn.Sequential(nn.Linear(self.out_channels, self.out_channels),
                                      linear_2_norm[1],
                                      nn.ReLU())

    def forward(self, voxels, coors, num_points_per_voxel, pts):
        coors_range_min_zyx = torch.tensor([self.coors_range_min[2], self.coors_range_min[1], self.coors_range_min[0]],
                                                dtype=torch.float32,
                                                device=voxels.device)
        voxel_size_zyx = torch.tensor([self.voxel_size[2], self.voxel_size[1], self.voxel_size[0]],
                                      dtype=torch.float32,
                                      device=voxels.device)

        # voxel coords are in z y x format with sample id [sample_id, z, y, x]
        anchors_zyx = coors[:, 1:] * voxel_size_zyx + (voxel_size_zyx / 2) + coors_range_min_zyx # [n, 3]
        # points in voxel features are in x y z format
        anchors = anchors_zyx[:, [2, 1, 0]]
        # account for the padded empty points in the voxel, and set them to the anchor to negate any difference
        empty_pts_mask = torch.all(voxels[:, :, :3] == 0, dim=2) # [n, 30]
        empty_pts_mask_expanded = empty_pts_mask.unsqueeze(-1).expand(-1, -1, 3) # [n, 30, 3]
        anchors_expanded = anchors.unsqueeze(1).expand(-1, voxels.size(1), -1) # [n, 1, 3]

        # set the xyz of the padded points to the anchor
        # TODO: should it be something else?
        voxels[:, :, :3] = torch.where(empty_pts_mask_expanded, anchors_expanded, voxels[:, :, :3])

        # Find the L1 difference between the points in the voxel and the anchor, only 2D
        anchors_diff = voxels[:, :, :3] - anchors.unsqueeze(1)
        voxels = torch.cat((voxels, anchors_diff[:, :, :2]), dim=2)

        # Find the L1 difference between the points in the voxel and the centroid, only 2D
        centroids = torch.mean(voxels[:, :, :3], dim=1)
        centroids_diff = voxels[:, :, :3] - centroids.unsqueeze(1)
        voxels = torch.cat((voxels, centroids_diff[:, :, :2]), dim=2)

        # add the centoid points itself as a feature
        centroids_expanded = centroids.unsqueeze(1).expand(-1, voxels.size(1), -1)
        voxels = torch.cat((voxels, centroids_expanded[:, :, :2]), dim=2)

        # Number of points per voxel
        num_points_expanded = num_points_per_voxel.unsqueeze(1).unsqueeze(2).expand(-1, voxels.size(1), 1) # [n, 30, 1]
        voxels = torch.cat((voxels, num_points_expanded), dim=2)

        # group the voxels by sample ID to use as queries and supports
        q_batches = []
        s_batches = []
        unique_sample_ids = coors[:, 0].unique()
        for sample_id in unique_sample_ids:
            sample_id_mask = coors[:, 0] == sample_id
            valid_voxels = voxels[sample_id_mask]

            q_batches.append(valid_voxels.shape[0])
            s_batches.append(valid_voxels.view(-1, valid_voxels.size(2)).shape[0])

        supports = voxels.view(-1, voxels.size(2))[:, :3]
        x = voxels.view(-1, voxels.size(2))[:, 3:]

        # first linear layer, [N_in, F_in + 7 -> F_out]
        x = self.linear_1(x)

        # kpconv layer, stacked features
        x = self.kpconv(anchors, supports, q_batches, s_batches, x)

        # second linear layer, [N_an, F_out]
        x = self.linear_2(x)

        return anchors, x

