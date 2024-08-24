import torch
import torch.nn as nn

from .neighbors import find_neighbors
from .kpconv import KPConv
from ..common import convert_to_feature_order, convert_to_length_order


class KPConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_kernels,
                 kernel_size,
                 influence_radius,
                 neighborhood_limit,
                 groups,
                 bias,
                 dimension):
        super().__init__()
        self.kpconv = KPConv(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=num_kernels,
                             radius=kernel_size,
                             sigma=influence_radius,
                             groups=groups,
                             bias=bias,
                             dimension=dimension)
        self.search_radius = influence_radius
        self.neighborhood_limit = neighborhood_limit
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, query, support, features, neighbors):
        features = self.apply_kpconv(query, support, features, neighbors)
        features = convert_to_feature_order(features)
        features = self.norm(features)
        features = self.relu(features)
        features = convert_to_length_order(features)
        return features

    def apply_kpconv(self, query, support, features, neighbors):
        kpconvs = []
        batch_dim = 0
        batch_size = query.size(batch_dim)
        for i in range(batch_size):
            batch_query = query[i, :, :]
            batch_support = support[i, :, :]
            batch_features = features[i, :, :]
            batch_neighbors = neighbors[i]
            batch_kpconv = self.kpconv(batch_query, batch_support, batch_features, batch_neighbors)
            kpconvs.append(batch_kpconv)
        features = torch.stack(kpconvs, dim=0)
        return features

    def find_neighbors(self, query, support):
        batch_dim = 0
        batch_size = query.size(batch_dim)
        neighbors = []
        for i in range(batch_size):
            batch_query = query[i, :, :]
            batch_support = support[i, :, :]
            batch_query_length = batch_query.size(0)
            batch_support_length = batch_support.size(0)
            batch_neighbors = find_neighbors(
                batch_query,
                batch_support,
                batch_query_length,
                batch_support_length,
                search_radius=self.search_radius,
                neighbor_limit=self.neighborhood_limit)
            neighbors.append(batch_neighbors)
        return neighbors
