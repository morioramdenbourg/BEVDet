import torch

from mmcv.ops import Voxelization
from mmdet3d.registry import MODELS
from torch import nn
from .kpconv.kpconv_block import KPConvBlock

@MODELS.register_module()
class KPBEVEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 spatial_scale,
                 point_cloud_range,
                 grid_size,
                 max_num_points_per_cell,
                 kpconv_args):
        super(KPBEVEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_num_points_per_cell = max_num_points_per_cell

        # s0 -> s3
        height = point_cloud_range[5] - point_cloud_range[2]
        grid_sizes = [grid_size,
                      (grid_size[0] // 2, grid_size[1] // 2),
                      (grid_size[0] // 4, grid_size[1] // 4),
                      (grid_size[0] // 8, grid_size[1] // 8)]
        voxel_sizes = [[spatial_scale, spatial_scale, height],
                       [spatial_scale * 2, spatial_scale * 2, height],
                       [spatial_scale * 4, spatial_scale * 4, height],
                       [spatial_scale * 8, spatial_scale * 8, height]]
        self.voxelizations = []
        for i in range(len(grid_sizes)):
            voxelization = Voxelization(voxel_size=voxel_sizes[i],
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=self.max_num_points_per_cell,
                                        max_voxels=grid_sizes[i][0] * grid_sizes[i][1])
            self.voxelizations.append((voxelization, voxel_sizes[i], grid_sizes[i]))

        num_added_dimensions = 10
        self.kpconv = KPConvBlock(**kpconv_args)
        self.pre_linear_block = nn.Sequential(nn.Linear(self.in_channels + num_added_dimensions, self.out_channels),
                                              nn.BatchNorm1d(self.out_channels),
                                              nn.ReLU())
        self.post_linear_block = nn.Sequential(nn.Linear(self.out_channels, self.out_channels),
                                               nn.BatchNorm1d(self.out_channels),
                                               nn.ReLU())

    def forward(self, inputs):
        points = inputs['points']
        dim_count = points.shape[2]
        coords_dim = 3
        feature_dim = 2
        batch_dim = 0
        # batch_size = points.size(batch_dim)
        # batched_grids = []
        # for i in range(batch_size):
            # print(f'Running KPBEV for batch={i}')
        grids = []

        for voxelization, voxel_size, grid_size in self.voxelizations:
            # print(f'Running KPBEV with voxel_size={voxel_size}')
            points_squeezed = points.view(-1, dim_count)
            # voxels, voxel_coords, num_points_per_voxel = self.voxelize(voxelization, points[i])
            voxels, voxel_coords, num_points_per_voxel = self.voxelize(voxelization, points_squeezed)

            anchors = self.get_anchors(voxel_coords, voxel_size)
            expanded_anchors = anchors.unsqueeze(1).expand(-1, voxels.size(1), -1)
            anchors_diff = voxels[:, :, :coords_dim] - expanded_anchors
            voxels = torch.cat((voxels, anchors_diff), dim=feature_dim)

            centroids = self.get_centroids(voxels, num_points_per_voxel)
            expanded_centroids = centroids.unsqueeze(1).expand(-1, voxels.size(1), -1)
            centroid_diff = voxels[:, :, :coords_dim] - expanded_centroids
            voxels = torch.cat((voxels, centroid_diff), dim=feature_dim)
            voxels = torch.cat((voxels, expanded_centroids), dim=feature_dim)

            num_points_expanded = num_points_per_voxel.unsqueeze(1).unsqueeze(2).expand(-1, voxels.size(1), 1)
            voxels = torch.cat((voxels, num_points_expanded), dim=feature_dim)

            points_enhanced = self.get_enhanced_points(voxels, num_points_per_voxel)
            coords = points_enhanced[:, :coords_dim]
            features = points_enhanced[:, coords_dim:]
            features = self.apply_kpconv(anchors, coords, features)

            # print(f'Creating grid with grid_size={grid_size}')
            grid = self.create_grid(features, grid_size)
            grids.append(grid)
        # batched_grids.append(grids)
        #
        # combined_batched_grids = []
        # for i in range(len(self.voxelizations)):
        #     batched_grid = torch.cat([grid[i] for grid in batched_grids], dim=0)
        #     combined_batched_grids.append(batched_grid)
        # return combined_batched_grids
        return grids

    @torch.no_grad()
    def voxelize(self, voxelization, points):
        voxels, voxel_coords, num_points_per_voxel = voxelization(points)
        return voxels, voxel_coords, num_points_per_voxel

    def apply_kpconv(self, anchors, coords, features):
        features = self.pre_linear_block(features)

        batch_anchors = anchors.unsqueeze(0)
        batch_coords = coords.unsqueeze(0)
        batch_features = features.unsqueeze(0)
        neighbors = self.kpconv.find_neighbors(batch_anchors, batch_coords)
        batch_features = self.kpconv(batch_anchors, batch_coords, batch_features, neighbors)

        features = batch_features.squeeze(0)
        features = self.post_linear_block(features)
        return features

    @staticmethod
    def create_grid(features, grid_size):
        size, feature_size = features.shape
        total_size = grid_size[0] * grid_size[1]

        # pad to fit the length and height
        if size < total_size:
            pad_size = total_size - size
            pad = torch.zeros((pad_size, feature_size), dtype=features.dtype, device=features.device)
            features = torch.cat([features, pad], dim=0)

        indices = torch.arange(total_size, device=features.device)
        output = torch.zeros((grid_size[0], grid_size[0], feature_size), dtype=features.dtype, device=features.device)
        flat_indices = indices % (grid_size[0] * grid_size[1])
        output.view(-1, feature_size)[flat_indices] = features
        output = output.permute(2, 0, 1)
        output = output.unsqueeze(0)
        return output

    @staticmethod
    def get_anchors(voxel_coords, voxel_size):
        anchors = voxel_coords + (torch.Tensor(voxel_size).to(voxel_coords.device) / 2)
        return anchors

    def get_centroids(self, voxels, num_points_per_voxel):
        device = voxels.device
        num_voxels = voxels.size(0)
        spatial_coords = voxels[:, :, :3]
        max_points_mask = torch.arange(self.max_num_points_per_cell).unsqueeze(0).expand(num_voxels, -1).to(device)
        mask = max_points_mask < num_points_per_voxel.unsqueeze(1)
        masked_coords = spatial_coords * mask.unsqueeze(2)
        centroids = masked_coords.sum(dim=1) / num_points_per_voxel.unsqueeze(1)

        # centroids = []
        # for i in range(voxels.size(0)):
        #     num_points = num_points_per_voxel[i]
        #     if num_points > 0:
        #         voxel_points = voxels[i, :num_points, :3]
        #         centroid = voxel_points.mean(dim=0)
        #         centroids.append(centroid)
        #     else:
        #         centroids.append(torch.zeros(voxels.size(2)))
        # centroids = torch.stack(centroids)

        return centroids

    @staticmethod
    def get_enhanced_points(voxels, num_points_per_voxel):
        points = []
        for i in range(voxels.size(0)):
            num_points = num_points_per_voxel[i]
            voxel_points = voxels[i, :num_points, :]
            points.append(voxel_points)
        points = torch.cat(points, dim=0)
        return points
