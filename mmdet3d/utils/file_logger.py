import json

import torch


def write_bboxes_to_files(results_dir, gt_bboxes_3d, gt_labels_3d, img_metas, losses):
    for index, meta in enumerate(img_metas):
        sample_token = meta['sample_idx']
        print(f'Writing bboxes to {results_dir} for sample {sample_token}')
        meta = {
            'use_lidar': True,
            'sample_token': sample_token,
            'loss_heatmap': losses['task0.loss_heatmap'].item(),
            'loss_xy': losses['task0.loss_xy'].item()
        }
        # write meta to file
        sample_meta_file_name = results_dir + '/' + sample_token + '_meta.json'
        with open(sample_meta_file_name, 'w') as meta_file:
            json.dump(meta, meta_file)
        # write gt to file (in lidar)
        sample_gt_bboxes_3d = gt_bboxes_3d[index]
        sample_gt_labels_3d = gt_labels_3d[index]
        sample_gt_file_name = results_dir + '/' + sample_token + '_gt.pth'
        torch.save((sample_gt_bboxes_3d, sample_gt_labels_3d), sample_gt_file_name)


def write_points_to_file(results_dir, pts, img_metas, file_name):
    for index, meta in enumerate(img_metas):
        sample_token = meta['sample_idx']
        sample_points_file_name = results_dir + '/' + sample_token + '_points' + file_name + '.pth'
        print(f'Writing pts to {sample_points_file_name} for sample {sample_token}')
        # meta = {
        #     'use_lidar': True,
        #     'sample_token': sample_token,
        # }
        # # write meta to file
        # sample_meta_file_name = results_dir + '/' + sample_token + '_meta.json'
        # with open(sample_meta_file_name, 'w') as meta_file:
        #     json.dump(meta, meta_file)
        # write points to file (in lidar)
        sample_points = pts[index]
        torch.save(sample_points, sample_points_file_name)