import torch

import time

from mmdet3d.registry import MODELS
from mmengine.model import BaseDataPreprocessor
from .kpconv.kpconv_block import KPConvBlock

@MODELS.register_module()
class KPConvPreprocessor(BaseDataPreprocessor):

    def __init__(self, kpconv_args):
        super(KPConvPreprocessor, self).__init__()
        self.kpconv = KPConvBlock(**kpconv_args)

    def forward(self, data, training):
        start_time = time.time()
        print("IN PREPROCESSOR")
        # move to device
        data = super().forward(data, training)

        coord_dim = 3
        points = data['inputs']['points']
        points = self.pad_point_clouds(points)
        coordinates = points[:, :, :coord_dim]
        features = points[:, :, coord_dim:]
        neighbors = self.kpconv.find_neighbors(coordinates, coordinates)
        features = self.apply_dense_kpconvs(coordinates, features, neighbors)

        feature_dim = 2
        new_points = torch.cat([coordinates, features], dim=feature_dim)
        # Calculate the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"OUT PREPROCESSOR: {elapsed_time}")
        return {
            'inputs': {
                'points': new_points,
            },
            'data_samples': data['data_samples']
        }

    def apply_dense_kpconvs(self, coordinates, features, neighbors):
        # Dense convolutions - query and support points are same
        features = self.kpconv(coordinates, coordinates, features, neighbors)
        features = self.kpconv(coordinates, coordinates, features, neighbors)
        features = self.kpconv(coordinates, coordinates, features, neighbors)
        return features

    @staticmethod
    def pad_point_clouds(points):
        max_points = max(pc.size(0) for pc in points)
        points = [KPConvPreprocessor.pad_point_cloud(pc, max_points) for pc in points]
        points = torch.stack(points, dim=0)
        return points

    @staticmethod
    def pad_point_cloud(pc, max_points):
        num_points = pc.size(0)
        if num_points < max_points:
            padding = (0, 0, 0, max_points - num_points)
            pc = torch.nn.functional.pad(pc, padding, mode='constant', value=0)
        return pc