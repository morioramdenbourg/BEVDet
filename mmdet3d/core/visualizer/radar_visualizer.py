import math
import numpy as np
import cv2
import matplotlib.colors as mcolors
import torch

from PIL import Image
from matplotlib import patches, pyplot as plt
from matplotlib.collections import PolyCollection
from mmdet3d.registry import VISUALIZERS
from mmengine.visualization import Visualizer
from nuscenes import NuScenes, NuScenesExplorer
from pyquaternion import Quaternion

class_names = [
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier'
]


@VISUALIZERS.register_module()
class RadarVisualizer(Visualizer):

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        # self.nuscenes = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes/')
        # self.explorer = NuScenesExplorer(self.nuscenes)

    @staticmethod
    def draw_points_with_map_and_bounding_boxes(sample, points):
        fig, ax = plt.subplots()
        axes_limits = RadarVisualizer.find_axes_limits(points)

        metainfo = sample.metainfo
        map_rec = RadarVisualizer.find_map_rec(metainfo)
        ego2global = RadarVisualizer.find_ego2global(metainfo)

        RadarVisualizer.draw_ego_map(ax, axes_limits, map_rec, ego2global)
        RadarVisualizer.draw_ego_vehicle(ax)

        lidar2ego = RadarVisualizer.find_lidar2ego(metainfo)
        RadarVisualizer.draw_points(ax, points, lidar2ego)

        bboxes = sample.gt_instances_3d.bboxes_3d
        RadarVisualizer.draw_bounding_boxes(ax, bboxes, lidar2ego)
        ax.title.set_text('Raw Radar Points')
        plt.show()

    @staticmethod
    def draw_points_with_map_and_features_and_bounding_boxes(sample, points):
        fig, ax = plt.subplots()
        axes_limits = RadarVisualizer.find_axes_limits(points)

        metainfo = sample.metainfo
        map_rec = RadarVisualizer.find_map_rec(metainfo)
        ego2global = RadarVisualizer.find_ego2global(metainfo)

        RadarVisualizer.draw_ego_map(ax, axes_limits, map_rec, ego2global)
        RadarVisualizer.draw_ego_vehicle(ax)

        lidar2ego = RadarVisualizer.find_lidar2ego(metainfo)
        RadarVisualizer.draw_points_with_features(ax, points, lidar2ego)

        lidar2ego = RadarVisualizer.find_lidar2ego(metainfo)
        bboxes = sample.gt_instances_3d.bboxes_3d
        RadarVisualizer.draw_bounding_boxes(ax, bboxes, lidar2ego)
        ax.title.set_text('Radar Features after 3xKPConv Preprocessing Layers')
        plt.show()

    @staticmethod
    def draw_points(axis, points, lidar2ego):
        point_scale = 1.0
        np_points = RadarVisualizer.to_numpy(points)

        lidar2ego_rotation, lidar2ego_translation = lidar2ego
        lidar2ego_rotation = np.array(lidar2ego_rotation)
        ypr_rad = Quaternion(lidar2ego_rotation).yaw_pitch_roll
        cos_yaw = -np.cos(ypr_rad[0])
        sin_yaw = -np.sin(ypr_rad[0])
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        np_points = np.dot(np_points[:, :2], rotation_matrix)
        np_points[:, :2] += lidar2ego_translation[:2]

        axis.scatter(np_points[:, 0], np_points[:, 1], s=point_scale, color='black')

    @staticmethod
    def draw_points_with_features(axis, points, lidar2ego):
        point_scale = 1.0
        np_points = RadarVisualizer.to_numpy(points)

        lidar2ego_rotation, lidar2ego_translation = lidar2ego
        lidar2ego_rotation = np.array(lidar2ego_rotation)
        ypr_rad = Quaternion(lidar2ego_rotation).yaw_pitch_roll
        cos_yaw = -np.cos(ypr_rad[0])
        sin_yaw = -np.sin(ypr_rad[0])
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        np_points = np.dot(np_points[:, :2], rotation_matrix)
        np_points[:, :2] += lidar2ego_translation[:2]

        features = np_points[:, 3:]
        summed_features = features.sum(axis=1)
        scatter = axis.scatter(np_points[:, 0], np_points[:, 1], c=summed_features, s=point_scale, cmap='coolwarm')
        color_bar = plt.colorbar(scatter)
        color_bar.set_label('Summed Features')

    @staticmethod
    def draw_grids_on_map(sample, grids, points, layer_name):
        axes_limits = RadarVisualizer.find_axes_limits(points)
        x_limit, y_limit = axes_limits

        metainfo = sample.metainfo
        map_rec = RadarVisualizer.find_map_rec(metainfo)
        ego2global = RadarVisualizer.find_ego2global(metainfo)

        for index, grid in enumerate(grids):
            fig, axis = plt.subplots()
            RadarVisualizer.draw_ego_map(axis, axes_limits, map_rec, ego2global)

            x_size = int(x_limit * 2)
            y_size = int(y_limit * 2)
            grid = RadarVisualizer.to_numpy(grid)
            reduced_grid = grid.sum(axis=0)
            rescaled_grid = cv2.resize(reduced_grid, (x_size, y_size), interpolation=cv2.INTER_LINEAR)

            grid_overlay = axis.imshow(rescaled_grid,
                        extent=[-x_limit, x_limit, -y_limit, y_limit],
                        cmap='coolwarm',
                        alpha=0.5,
                        interpolation='nearest')
            cbar = plt.colorbar(grid_overlay, ax=axis, orientation='vertical')
            cbar.set_label('Feature Intensity (Summed)')

            RadarVisualizer.draw_ego_vehicle(axis)
            lidar2ego = RadarVisualizer.find_lidar2ego(metainfo)
            bboxes = sample.gt_instances_3d.bboxes_3d
            RadarVisualizer.draw_bounding_boxes(axis, bboxes, lidar2ego)

            plt.title(f'Feature Map Intensity after {layer_name} for Shape {grid.shape}')
            plt.show()

    @staticmethod
    def draw_ego_map(axis, axes_limits, map_rec, ego2global):
        ego2global_rotation, ego2global_translation = ego2global
        x_limit, y_limit = axes_limits
        map_mask = map_rec['mask']
        ego2global_rotation = np.array(ego2global_rotation)
        ego2global_translation = np.array(ego2global_translation)
        pixel_coords = map_mask.to_pixel_coords(ego2global_translation[0], ego2global_translation[1])
        scaled_limit_px = int(x_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()
        cropped = RadarVisualizer.crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        global2ego = RadarVisualizer.quaternion_inverse(ego2global_rotation)
        ypr_rad = Quaternion(global2ego).yaw_pitch_roll
        yaw_deg = math.degrees(ypr_rad[0])
        image = Image.fromarray(cropped).rotate(yaw_deg)
        rotated_cropped = np.array(image)

        ego_centric_map = RadarVisualizer.crop_image(rotated_cropped,
                                                     int(rotated_cropped.shape[1] / 2),
                                                     int(rotated_cropped.shape[0] / 2),
                                                     scaled_limit_px)

        ego_centric_map[ego_centric_map == map_mask.foreground] = 125
        ego_centric_map[ego_centric_map == map_mask.background] = 255
        axis.imshow(ego_centric_map,
                    extent=[-x_limit, y_limit, -x_limit, y_limit],
                    cmap='gray',
                    vmin=0,
                    vmax=255,
                    interpolation='nearest')
        axis.set_xlim(-x_limit, x_limit)
        axis.set_ylim(-y_limit, y_limit)
        axis.set_yticks(np.arange(-x_limit, x_limit + 1, 10))
        axis.set_xticks(np.arange(-y_limit, y_limit + 1, 10))
        axis.set_aspect('equal')

    @staticmethod
    def draw_ego_vehicle(axis):
        length = 4.084 # vehicle dimensions (wikipedia)
        width = 1.73 # vehicle dimensions (wikipedia)
        vehicle_rect = patches.Rectangle((-length / 2, -width / 2), length, width, linewidth=1, edgecolor='r',
                                         facecolor='none')
        axis.add_patch(vehicle_rect)

        arrow_length = length / 2
        arrow = patches.FancyArrow(
            x=0,
            y=0,
            dx=arrow_length,
            dy=0,
            width=0.3,
            color='r'
        )
        axis.add_patch(arrow)

    @staticmethod
    def draw_bounding_boxes(axis, bboxes, lidar2ego):
        lidar2ego_rotation, lidar2ego_translation = lidar2ego
        bev_bboxes = RadarVisualizer.to_numpy(bboxes.bev)
        ctr, w, h, theta = np.split(bev_bboxes, [2, 3, 4], axis=-1)
        cos_value, sin_value = np.cos(theta), np.sin(theta)
        w *= 1.0 # for visualization
        h *= 1.0 # for visalization
        vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
        vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        polygons = np.stack([pt1, pt2, pt3, pt4], axis=-2)

        lidar2ego_rotation = np.array(lidar2ego_rotation)
        ypr_rad = Quaternion(lidar2ego_rotation).yaw_pitch_roll
        cos_yaw = -np.cos(ypr_rad[0])
        sin_yaw = -np.sin(ypr_rad[0])
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        polygons[:, :, :2] = np.dot(polygons[:, :, :2], rotation_matrix)
        polygons[:, :, :2] += lidar2ego_translation[:2]

        polygon_collection = PolyCollection(
            polygons,
            alpha=0.5,
            linewidths=1,
            facecolors='red',
            edgecolors='#FF4500')
        axis.add_collection(polygon_collection)

    @staticmethod
    def draw_predicted_bounding_boxes(sample, predictions, points):
        fig, axis = plt.subplots()
        axes_limits = RadarVisualizer.find_axes_limits(points)
        metainfo = sample.metainfo
        map_rec = RadarVisualizer.find_map_rec(metainfo)
        ego2global = RadarVisualizer.find_ego2global(metainfo)

        RadarVisualizer.draw_ego_map(axis, axes_limits, map_rec, ego2global)
        RadarVisualizer.draw_ego_vehicle(axis)

        lidar2ego = RadarVisualizer.find_lidar2ego(metainfo)
        bboxes = predictions.bboxes_3d.detach()
        RadarVisualizer.draw_bounding_boxes(axis, bboxes, lidar2ego)
        # RadarVisualizer.draw_bounding_boxes_2(axis, bboxes)
        plt.title('Predicted Bounding Boxes')
        plt.show()

    @staticmethod
    def crop_image(image, x_px, y_px, axes_limit_px):
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px)
        y_max = int(y_px + axes_limit_px)

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    @staticmethod
    def quaternion_conjugate(quaternion):
        w, x, y, z = quaternion
        return np.array([w, -x, -y, -z])

    @staticmethod
    def quaternion_inverse(quaternion):
        w, x, y, z = quaternion
        norm_sq = w * w + x * x + y * y + z * z
        conjugate = RadarVisualizer.quaternion_conjugate(quaternion)
        return conjugate / norm_sq

    @staticmethod
    def find_ego2global(metainfo):
        ego2global_rotation = metainfo['ego2global_rotation']
        ego2global_translation = metainfo['ego2global_translation']
        return ego2global_rotation, ego2global_translation

    @staticmethod
    def find_lidar2ego(metainfo):
        lidar2ego_rotation = metainfo['lidar2ego_rotation']
        lidar2ego_translation = metainfo['lidar2ego_translation']
        return lidar2ego_rotation, lidar2ego_translation

    @staticmethod
    def find_map_rec(metainfo):
        return metainfo['map']

    @staticmethod
    def to_numpy(tensor):
        return tensor.cpu().numpy()

    @staticmethod
    def find_axes_limits(points):
        x_limit = points[:, 0].max().round().item()
        y_limit = points[:, 1].max().round().item()
        return x_limit, y_limit

