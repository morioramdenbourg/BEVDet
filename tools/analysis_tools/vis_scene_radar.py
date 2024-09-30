import argparse
import json
import os
import pickle
import cv2
import numpy as np
import torch
import traceback
from pyquaternion.quaternion import Quaternion

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

def features_to_colors(features):
    features = np.sum(features ** 2, axis=1)
    features_normalized = (features - np.min(features)) / (np.max(features) - np.min(features))
    features_colormap = np.uint8(features_normalized * 255)
    colors = cv2.applyColorMap(features_colormap, cv2.COLORMAP_JET)
    return colors

def get_ego_to_camera(image_record):
    camera_to_ego_rot = Quaternion(image_record['sensor2ego_rotation'])
    camera_to_ego_trans = np.array(image_record['sensor2ego_translation'])
    ego_to_camera_h = np.eye(4, dtype=np.float32)
    ego_to_camera_h[:3, :3] = camera_to_ego_rot.rotation_matrix
    ego_to_camera_h[:3, 3] = camera_to_ego_trans
    ego_to_camera_h = np.linalg.inv(ego_to_camera_h)
    return ego_to_camera_h

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
        type=float,
        default=51.2,
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
    results_dir = '/home/ge95fav/IDP-SPADES/BEVDet/vis/kppillarsbev_nus'

    # load dataset information
    dataset = pickle.load(open(args.root_path + '/full/nuscenes_infos_train.pkl', 'rb'))

    meta_file_names = [f for f in os.listdir(results_dir) if "meta" in f]
    sample_tokens = [file.split('_')[0] for file in meta_file_names]
    nuscenes_samples = { sample['token']: sample for sample in dataset['infos'] }

    commands = []
    for index, sample_token in enumerate(sample_tokens):
        try:
            print(f'{index}/{len(meta_file_names)}')
            print(f'Sample token:', sample_token)

            if not sample_token in nuscenes_samples:
                print('Skipping sample token', sample_token)
                continue

            nuscenes_sample = nuscenes_samples[sample_token]

            meta_file_name = results_dir + '/' + sample_token + "_meta.json"
            if not os.path.exists(meta_file_name):
                print(f'{meta_file_name} does not exist')
                continue

            with open(meta_file_name, 'r') as meta_file:
                meta = json.load(meta_file)
                if meta['loss_heatmap'] > 2:
                    print('Skipping sample token', sample_token, 'due to high heatmap loss')
                    continue

            points_file_name = results_dir + '/' + sample_token + "_pointspts_before_voxelize.pth"
            if not os.path.exists(points_file_name):
                print(f'{points_file_name} does not exist')
                continue

            anchor_points_file_name = results_dir + '/' + sample_token + "_pointspts_after_middle_encoder.pth"
            if not os.path.exists(anchor_points_file_name):
                print(f'{anchor_points_file_name} does not exist')
                continue

            radar_points = torch.load(points_file_name).detach().cpu().numpy()
            anchor_points = torch.load(anchor_points_file_name).detach().cpu().numpy()
            colors = features_to_colors(radar_points[:, 3:])
            anchor_colors = features_to_colors(anchor_points[:, 3:])

            gt_file_name = results_dir + '/' + sample_token + "_gt.pth"
            if not os.path.exists(gt_file_name):
                print(f'{gt_file_name} does not exist')
                continue

            gt = torch.load(gt_file_name)
            gt_bboxes_3d, gt_labels_3d = gt

            images = []
            for view in CAMERA_VIEWS:
                image_record = nuscenes_sample['cams'][view]
                image = cv2.imread(image_record['data_path'])

                width = image.shape[1]
                height = image.shape[0]

                lidar_to_camera = get_lidar_to_camera(image_record)
                camera_to_image = image_record['cam_intrinsic']

                # lidar -> camera
                ones = np.ones((radar_points.shape[0], 1), dtype=radar_points.dtype)
                radar_points_h = np.concatenate([radar_points[:, :3], ones], axis=1)
                camera_points_h = radar_points_h @ lidar_to_camera.T

                # filter out points that are behind the camera's view
                camera_z_coords = camera_points_h[:, 2]
                valid_view = camera_z_coords >= 0

                valid = np.ones((radar_points.shape[0]), dtype=bool)
                # filter out points that hit the ground
                valid = np.logical_and(valid_view, valid)

                # camera -> image
                camera_points = camera_points_h[:, :3]
                image_points_h = camera_points @ camera_to_image.T
                image_points = image_points_h[:, :2] / image_points_h[:, 2:3]

                print(f'Visualizing {valid.sum()} points on image {view}')
                for image_point_index, image_point in enumerate(image_points):
                    if valid[image_point_index]:
                        x, y = image_point[:2]
                        color = colors[image_point_index][0]
                        cv2.circle(image,
                                   (int(x), int(y)),
                                   radius=5,
                                   color=color.tolist(),
                                   thickness=-1)

                # lidar -> camera
                gt_bboxes_3d_corners = gt_bboxes_3d.corners
                lidar_gt_corners = gt_bboxes_3d_corners.detach().cpu().numpy()
                lidar_gt_corners_flat = lidar_gt_corners.reshape(-1, 3)
                ones_corners = np.ones((lidar_gt_corners_flat.shape[0], 1), dtype=lidar_gt_corners_flat.dtype)
                lidar_gt_corners_flat_h = np.concatenate([lidar_gt_corners_flat, ones_corners], axis=1)
                camera_gt_corners_flat_h = lidar_gt_corners_flat_h @ lidar_to_camera.T

                # filter out bboxes outside the camera's depth (i.e. behind the camera)
                camera_gt_corners = camera_gt_corners_flat_h[:, :3].reshape(-1, 8, 3)
                camera_gt_z_coords = camera_gt_corners[:, :, 2]
                valid_corner_view_points = camera_gt_z_coords > 0

                # camera -> image
                image_gt_corners_flat_h = camera_gt_corners_flat_h[:, :3] @ camera_to_image.T
                image_gt_corners_flat = image_gt_corners_flat_h[:, :2] / image_gt_corners_flat_h[:, 2:3]
                image_gt_corners = image_gt_corners_flat.reshape(-1, 8, 2)

                # filter out bboxes beyond the image's boundaries
                x_coords = image_gt_corners[:, :, 0]
                y_coords = image_gt_corners[:, :, 1]
                x_within_bounds = (x_coords >= 0) & (x_coords <= width)
                y_within_bounds = (y_coords >= 0) & (y_coords <= height)
                valid_bounding_boxes = x_within_bounds & y_within_bounds & valid_corner_view_points

                for gt_bbox_index in range(gt_bboxes_3d.tensor.shape[0]):
                    for line_indices in bounding_box_line_indices:
                        if not valid_bounding_boxes[gt_bbox_index, line_indices[0]] and not valid_bounding_boxes[gt_bbox_index, line_indices[1]]:
                            continue

                        p1 = image_gt_corners[gt_bbox_index, line_indices[0]]
                        p2 = image_gt_corners[gt_bbox_index, line_indices[1]]
                        cv2.line(image,
                                 (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])),
                                 color=(255, 255, 255),
                                 thickness=5)

                images.append(image)

            # BEV plane
            canvas = np.zeros((int(args.canvas_size), int(args.canvas_size), 3), dtype=np.uint8)

            # draw lidar points on canvas
            x_scale = args.canvas_size / (args.show_range * 2)
            y_scale = args.canvas_size / (args.show_range * 2)

            # print(f'Visualizing {radar_points.shape[0]} points on canvas')
            # for radar_point_index, points in enumerate(radar_points):
            #     x, y = points[:2]
            #     color = colors[radar_point_index][0]
            #     cv2.circle(canvas,
            #                (int(args.canvas_size / 2 + x_scale * x), int(args.canvas_size / 2 - y_scale * y)),
            #                radius=1,
            #                color=color.tolist(),
            #                thickness=-1)

            print(f'Visualizing {anchor_points.shape[0]} anchor points on canvas')
            for anchor_point_index, points in enumerate(anchor_points):
                x, y = points[:2]
                color = anchor_colors[anchor_point_index][0]
                cv2.circle(canvas,
                           (int(args.canvas_size / 2 + x_scale * x), int(args.canvas_size / 2 - y_scale * y)),
                           radius=2,
                           color=color.tolist(),
                           thickness=-1)

            cv2.circle(canvas,
                       (args.canvas_size // 2, args.canvas_size // 2),
                       radius=5,
                       color=(255, 255, 255),
                       thickness=-1)

            # Reload to transform again
            gt = torch.load(gt_file_name)
            gt_bboxes_3d, gt_labels_3d = gt

            print(f'Visualizing {gt_bboxes_3d.tensor.shape[0]} ground-truth bounding boxes on canvas')
            for gt_bbox_index in range(gt_bboxes_3d.tensor.shape[0]):
                for line_indices in bounding_box_line_indices_bev:
                    p1 = gt_bboxes_3d.corners[gt_bbox_index, line_indices[0]]
                    p2 = gt_bboxes_3d.corners[gt_bbox_index, line_indices[1]]
                    cv2.line(canvas,
                             (int(args.canvas_size // 2 + x_scale * p1[0].item()), int(args.canvas_size // 2 - y_scale * p1[1].item())),
                             (int(args.canvas_size // 2 + x_scale * p2[0].item()), int(args.canvas_size // 2 - y_scale * p2[1].item())),
                             color=(255, 255, 255),
                             thickness=1)

                # top_left_p = gt_bboxes_3d_corners[gt_bbox_index, bounding_box_top_left]
                # # top_left_p = np.dot(top_left_p, R_z.T)
                # top_left_u = int(canvas_size / 2 + x_scale * top_left_p[0])
                # top_left_v = int(canvas_size / 2 - y_scale * top_left_p[1]) - 2
                # label_number = gt_labels_3d[gt_bbox_index].item()
                # label = label_map.get(label_number, "Unknown")
                # cv2.putText(canvas,
                #             text=label,
                #             org=(top_left_u, top_left_v),
                #             fontFace=cv2.FONT_HERSHEY_PLAIN,
                #             fontScale=0.5,
                #             color=(255, 255, 255),
                #             thickness=1)

            # fuse image-view and bev
            image = np.zeros((900 * 2 + args.canvas_size * args.scale_factor, 1600 * 3, 3), dtype=np.uint8)
            image[:900, :, :] = np.concatenate(images[:3], axis=1)
            img_back = np.concatenate([images[3][:, ::-1, :], images[4][:, ::-1, :], images[5][:, ::-1, :]], axis=1)
            image[900 + args.canvas_size * args.scale_factor:, :, :] = img_back
            image = cv2.resize(image, (int(1600 / args.scale_factor * 3), int(900 / args.scale_factor * 2 + args.canvas_size)))
            w_begin = int((1600 * 3 / args.scale_factor - args.canvas_size) // 2)
            image[int(900 / args.scale_factor):int(900 / args.scale_factor) + args.canvas_size, w_begin:w_begin + args.canvas_size, :] = canvas

            if args.format == 'image':
                path = os.path.join(results_dir, '%s.png' % sample_token)
                cv2.imwrite(path, image)
                commands.append(
                    f'rm -rf ~/Downloads/pictures/{sample_token}.png && scp ge95fav@129.187.227.223:{path} ~/Downloads/pictures')

        except Exception as e:
            traceback.print_exc()
            print(f'Error processing sample {sample_token}: {e}')
            continue


    for command in commands:
        print(command)


if __name__ == '__main__':
    main()
