import argparse
import json
import os
import pickle
import cv2
import numpy as np
import torch
from pyquaternion.quaternion import Quaternion

color_map = {0: (255, 255, 0), 1: (0, 255, 255)}

CAMERA_VIEWS = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT'
]

bounding_box_line_indices_bev = [
    (0, 3),
    (3, 7),
    (7, 4),
    (0, 4)
]

bounding_box_line_indices = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7)
]

bounding_box_top_left = 4

# label_map = {
#     -1: 'background',
#     0: 'bicycle',
#     1: 'motorcycle',
#     2: 'pedestrian',
#     3: 'traffic_cone',
#     4: 'barrier',
#     5: 'car',
#     6: 'truck',
#     7: 'trailer',
#     8: 'bus',
#     9: 'construction_vehicle'
# }

label_map = {
    -1: 'background',
    0: 'car',
    1: 'truck',
    2: 'trailer',
    3: 'bus',
    4: 'construction_vehicle'
}

def depths_to_colors(depths):
    depths = np.sqrt(np.sum(depths ** 2, axis=1))
    depth_normalized = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
    depth_colormap = np.uint8(depth_normalized * 255)
    colors = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
    return colors

def get_lidar_to_camera(image_record):
    camera_to_lidar = np.eye(4, dtype=np.float32)
    camera_to_lidar[:3, :3] = image_record['sensor2lidar_rotation']
    camera_to_lidar[:3, 3] = image_record['sensor2lidar_translation']
    lidar_to_camera = np.linalg.inv(camera_to_lidar)
    return lidar_to_camera


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predictions directory')
    parser.add_argument(
        '--show-range',
        type=int,
        default=60,
        help='Range of visualization in BEV')
    parser.add_argument(
        '--canvas-size', type=int, default=1000, help='Size of canvas in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of the visualized canvas')
    parser.add_argument(
        '--vis-thresh',
        type=float,
        default=0.3,
        help='Threshold the predicted results')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--version',
        type=str,
        default='train',
        help='Version of nuScenes dataset')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes/',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='image',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=20, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load predicted results
    results_dir = '/home/ge95fav/IDP-SPADES/BEVDet/work_dirs/simple_nus/visualization/'

    # load dataset information
    info_path = args.root_path + '/small/nuscenes_infos_train.pkl'
    dataset = pickle.load(open(info_path, 'rb'))

    scale_factor = args.scale_factor
    canvas_size = args.canvas_size
    show_range = args.show_range

    # if args.format == 'video':
    #     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #     vout = cv2.VideoWriter(
    #         os.path.join(vis_dir, '%s.mp4' % args.video_prefix),
    #         fourcc,
    #         args.fps,
    #         (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canvas_size)))

    meta_file_names = [f for f in os.listdir(results_dir) if "meta" in f]
    num_samples = len(meta_file_names)
    sample_tokens = [file.split('_')[0] for file in meta_file_names]
    nuscenes_samples = { sample['token']: sample for sample in dataset['infos'] }

    commands = []
    for index, sample_token in enumerate(sample_tokens):
        try:
            print(f'{index}/{num_samples}')
            print(f'Sample token:', sample_token)
            nuscenes_sample = nuscenes_samples[sample_token]

            points_file_name = results_dir + '/' + sample_token + "_points.pth"
            if not os.path.exists(points_file_name):
                print(f'{points_file_name} does not exist')
                continue

            lidar_points = torch.load(points_file_name).detach().cpu().numpy()
            colors = depths_to_colors(lidar_points[:, :3])

            gt_file_name = results_dir + '/' + sample_token + "_gt.pth"
            if not os.path.exists(gt_file_name):
                print(f'{gt_file_name} does not exist')
                continue

            gt = torch.load(gt_file_name)
            gt_bboxes_3d, gt_labels_3d = gt
            gt_bboxes_3d_corners = gt_bboxes_3d.corners
            num_gt_bboxes = gt_bboxes_3d_corners.shape[0]

            feature_map_file_name = results_dir + '/' + sample_token + "_feature_map.pth"
            if not os.path.exists(feature_map_file_name):
                print(f'{feature_map_file_name} does not exist')
                continue
            feature_map = torch.load(feature_map_file_name)

            pred_file_name = results_dir + '/' + sample_token + "_predictions.pth"
            if not os.path.exists(pred_file_name):
                print(f'{pred_file_name} does not exist')
                continue
            predictions = torch.load(pred_file_name)

            images = []
            for view in CAMERA_VIEWS:
                image_record = nuscenes_sample['cams'][view]
                image_path = image_record['data_path']
                image = cv2.imread(image_path)

                width = image.shape[1]
                height = image.shape[0]

                lidar_to_camera = get_lidar_to_camera(image_record)
                camera_to_image = image_record['cam_intrinsic']

                # lidar -> camera
                ones = np.ones((lidar_points.shape[0], 1), dtype=lidar_points.dtype)
                lidar_points_h = np.concatenate([lidar_points[:, :3], ones], axis=1)
                camera_points_h = lidar_points_h @ lidar_to_camera.T

                # filter out points that are behind the camera's view
                camera_z_coords = camera_points_h[:, 2]
                valid_view = camera_z_coords >= 0

                valid = np.ones((lidar_points.shape[0]), dtype=bool)
                # filter out points that hit the ground
                valid = np.logical_and(lidar_points[:, 2] >= -5, valid_view, valid)

                # camera -> image
                camera_points = camera_points_h[:, :3]
                image_points_h = camera_points @ camera_to_image.T
                image_points = image_points_h[:, :2] / image_points_h[:, 2:3]

                print(f'Visualizing {valid.sum()} points on image {view}')
                for image_point_index, image_point in enumerate(image_points):
                    if valid[image_point_index]:
                        x, y = image_point[:2]
                        color = colors[image_point_index][0]
                        # cv2.circle(image,
                        #            (int(x), int(y)),
                        #            radius=5,
                        #            color=color.tolist(),
                        #            thickness=-1)

                # lidar -> camera
                lidar_gt_corners = gt_bboxes_3d_corners.detach().cpu().numpy()
                lidar_gt_corners_flat = lidar_gt_corners.reshape(-1, 3)
                ones_corners = np.ones((lidar_gt_corners_flat.shape[0], 1), dtype=lidar_gt_corners_flat.dtype)
                lidar_gt_corners_flat_h = np.concatenate([lidar_gt_corners_flat, ones_corners], axis=1)
                camera_gt_corners_flat_h = lidar_gt_corners_flat_h @ lidar_to_camera.T

                # filter out bboxes outside the camera's depth (i.e. behind the camera)
                camera_gt_corners = camera_gt_corners_flat_h[:, :3].reshape(-1, 8, 3)
                camera_gt_z_coords = camera_gt_corners[:, :, 2]
                valid_corner_view_points = np.all(camera_gt_z_coords >= 0, axis=1)

                # camera -> image
                image_gt_corners_flat_h = camera_gt_corners_flat_h[:, :3] @ camera_to_image.T
                image_gt_corners_flat = image_gt_corners_flat_h[:, :2] / image_gt_corners_flat_h[:, 2:3]
                image_gt_corners = image_gt_corners_flat.reshape(-1, 8, 2)

                # filter out bboxes beyond the image's boundaries
                x_coords = image_gt_corners[:, :, 0]
                y_coords = image_gt_corners[:, :, 1]
                x_within_bounds = np.all((x_coords >= 0) & (x_coords <= width), axis=1)
                y_within_bounds = np.all((y_coords >= 0) & (y_coords <= height), axis=1)
                valid_bounding_boxes = x_within_bounds & y_within_bounds & valid_corner_view_points

                # draw ground truth boxes on image
                print(f'Visualizing {valid_bounding_boxes.sum()} ground-truth bounding boxes on image {view}')

                for gt_bbox_index in range(num_gt_bboxes):
                    if valid_bounding_boxes[gt_bbox_index]:
                        for line_indices in bounding_box_line_indices:
                            p1 = image_gt_corners[gt_bbox_index, line_indices[0]]
                            p2 = image_gt_corners[gt_bbox_index, line_indices[1]]
                            cv2.line(image,
                                     (int(p1[0]), int(p1[1])),
                                     (int(p2[0]), int(p2[1])),
                                     color=(255, 255, 255),
                                     thickness=5)

                images.append(image)

            # BEV plane
            canvas = np.zeros((int(canvas_size), int(canvas_size), 3), dtype=np.uint8)

            # draw feature map on canvas
            # summed_feature_map = feature_map.sum(axis=0).detach().cpu().numpy() # Sum across the 64 channels
            # resized_summed_feature_map = cv2.resize(summed_feature_map, (canvas_size, canvas_size))
            # normalized_summed_feature_map = cv2.normalize(resized_summed_feature_map, None, 0, 255, cv2.NORM_MINMAX)
            # normalized_summed_feature_map = normalized_summed_feature_map.astype(np.uint8)
            # summed_feature_map_3ch = cv2.cvtColor(normalized_summed_feature_map, cv2.COLOR_GRAY2BGR)

            # draw lidar points on canvas
            x_scale = canvas_size / (show_range * 2)
            y_scale = canvas_size / (show_range * 2)

            print(f'Visualizing {lidar_points.shape[0]} points on canvas')
            for lidar_point_index, lidar_point in enumerate(lidar_points):
                x, y = lidar_point[:2]
                u = int(canvas_size / 2 + x_scale * x)
                v = int(canvas_size / 2 - y_scale * y)
                color = colors[lidar_point_index][0]
                cv2.circle(canvas,
                           (u, v),
                           radius=1,
                           color=color.tolist(),
                           thickness=-1)

            cv2.circle(canvas,
                       (canvas_size // 2, canvas_size // 2),
                       radius=5,
                       color=(255, 255, 255),
                       thickness=-1)

            print(f'Visualizing {num_gt_bboxes} ground-truth bounding boxes on canvas')
            for gt_bbox_index in range(num_gt_bboxes):
                for line_indices in bounding_box_line_indices_bev:
                    p1 = gt_bboxes_3d_corners[gt_bbox_index, line_indices[0]]
                    p2 = gt_bboxes_3d_corners[gt_bbox_index, line_indices[1]]
                    u1 = int(canvas_size / 2 + x_scale * p1[0])
                    v1 = int(canvas_size / 2 - y_scale * p1[1])
                    u2 = int(canvas_size / 2 + x_scale * p2[0])
                    v2 = int(canvas_size / 2 - y_scale * p2[1])
                    cv2.line(
                        canvas,
                        (u1, v1),
                        (u2, v2),
                        color=(255, 255, 255),
                        thickness=1)


                top_left_p = gt_bboxes_3d_corners[gt_bbox_index, bounding_box_top_left]
                top_left_u = int(canvas_size / 2 + x_scale * top_left_p[0])
                top_left_v = int(canvas_size / 2 - y_scale * top_left_p[1]) - 2
                label_number = gt_labels_3d[gt_bbox_index].item()
                label = label_map.get(label_number, "Unknown")
                cv2.putText(canvas,
                            text=label,
                            org=(top_left_u, top_left_v),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=0.5,
                            color=(255, 255, 255),
                            thickness=1)

            # draw predictions on canvas
            pred_bboxes = predictions[0]
            pred_bboxes_corners = pred_bboxes.corners.detach().cpu().numpy()
            print(f'Visualizing {pred_bboxes_corners.shape[0]} predictions on canvas')
            for pred_bbox_index in range(pred_bboxes_corners.shape[0]):
                for line_indices in bounding_box_line_indices_bev:
                    p1 = pred_bboxes_corners[pred_bbox_index, line_indices[0]]
                    p2 = pred_bboxes_corners[pred_bbox_index, line_indices[1]]
                    u1 = int(canvas_size / 2 + x_scale * p1[0])
                    v1 = int(canvas_size / 2 - y_scale * p1[1])
                    u2 = int(canvas_size / 2 + x_scale * p2[0])
                    v2 = int(canvas_size / 2 - y_scale * p2[1])
                    cv2.line(
                        canvas,
                        (u1, v1),
                        (u2, v2),
                        color=(203, 192, 255),
                        thickness=2)



            # fuse image-view and bev
            image = np.zeros((900 * 2 + canvas_size * scale_factor, 1600 * 3, 3), dtype=np.uint8)
            image[:900, :, :] = np.concatenate(images[:3], axis=1)
            img_back = np.concatenate([images[3][:, ::-1, :], images[4][:, ::-1, :], images[5][:, ::-1, :]], axis=1)
            image[900 + canvas_size * scale_factor:, :, :] = img_back
            image = cv2.resize(image, (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canvas_size)))
            w_begin = int((1600 * 3 / scale_factor - canvas_size) // 2)
            image[int(900 / scale_factor):int(900 / scale_factor) + canvas_size, w_begin:w_begin + canvas_size, :] = canvas

            if args.format == 'image':
                path = os.path.join(results_dir, '%s.png' % sample_token)
                cv2.imwrite(path, image)
                commands.append(f'rm -rf ~/Downloads/{sample_token}.png && scp ge95fav@129.187.227.223:{path} ~/Downloads')

        except Exception as error:
            print(error)
            continue

    if args.format == 'video':
        vout.release()

    for command in commands:
        print(command)


if __name__ == '__main__':
    main()
