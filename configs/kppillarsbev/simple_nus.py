"""
mAP: 0.1513
mATE: 0.5916
mASE: 0.3076
mAOE: 0.7253
mAVE: 0.8964
mAAE: 0.2375
NDS: 0.2998
Eval time: 80.4s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.556	0.336	0.186	0.467	0.713	0.222
truck	0.136	0.571	0.264	0.498	0.957	0.297
bus	0.199	0.672	0.238	0.450	1.777	0.374
trailer	0.141	0.834	0.250	0.711	0.379	0.129
construction_vehicle	0.005	0.798	0.519	1.542	0.081	0.302
pedestrian	0.134	0.365	0.301	1.521	1.056	0.438
motorcycle	0.023	0.607	0.273	0.649	1.612	0.062
bicycle	0.012	0.443	0.292	0.565	0.597	0.076
traffic_cone	0.051	0.609	0.411	nan	nan	nan
barrier	0.256	0.680	0.343	0.125	nan	nan
{'pts_bbox_NuScenes/car_AP_dist_0.5': 0.3652, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.5579, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.6306, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.6686, 'pts_bbox_NuScenes/car_trans_err': 0.3364, 'pts_bbox_NuScenes/car_scale_err': 0.1856, 'pts_bbox_NuScenes/car_orient_err': 0.4672, 'pts_bbox_NuScenes/car_vel_err': 0.7126, 'pts_bbox_NuScenes/car_attr_err': 0.222, 'pts_bbox_NuScenes/mATE': 0.5916, 'pts_bbox_NuScenes/mASE': 0.3076, 'pts_bbox_NuScenes/mAOE': 0.7253, 'pts_bbox_NuScenes/mAVE': 0.8964, 'pts_bbox_NuScenes/mAAE': 0.2375, 'pts_bbox_NuScenes/truck_AP_dist_0.5': 0.0246, 'pts_bbox_NuScenes/truck_AP_dist_1.0': 0.1185, 'pts_bbox_NuScenes/truck_AP_dist_2.0': 0.1835, 'pts_bbox_NuScenes/truck_AP_dist_4.0': 0.2175, 'pts_bbox_NuScenes/truck_trans_err': 0.571, 'pts_bbox_NuScenes/truck_scale_err': 0.2639, 'pts_bbox_NuScenes/truck_orient_err': 0.4979, 'pts_bbox_NuScenes/truck_vel_err': 0.9569, 'pts_bbox_NuScenes/truck_attr_err': 0.2971, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.0049, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.0139, 'pts_bbox_NuScenes/construction_vehicle_trans_err': 0.7982, 'pts_bbox_NuScenes/construction_vehicle_scale_err': 0.5188, 'pts_bbox_NuScenes/construction_vehicle_orient_err': 1.5421, 'pts_bbox_NuScenes/construction_vehicle_vel_err': 0.081, 'pts_bbox_NuScenes/construction_vehicle_attr_err': 0.3019, 'pts_bbox_NuScenes/bus_AP_dist_0.5': 0.0227, 'pts_bbox_NuScenes/bus_AP_dist_1.0': 0.1425, 'pts_bbox_NuScenes/bus_AP_dist_2.0': 0.2943, 'pts_bbox_NuScenes/bus_AP_dist_4.0': 0.3379, 'pts_bbox_NuScenes/bus_trans_err': 0.672, 'pts_bbox_NuScenes/bus_scale_err': 0.2379, 'pts_bbox_NuScenes/bus_orient_err': 0.4499, 'pts_bbox_NuScenes/bus_vel_err': 1.7768, 'pts_bbox_NuScenes/bus_attr_err': 0.3738, 'pts_bbox_NuScenes/trailer_AP_dist_0.5': 0.0039, 'pts_bbox_NuScenes/trailer_AP_dist_1.0': 0.0694, 'pts_bbox_NuScenes/trailer_AP_dist_2.0': 0.2174, 'pts_bbox_NuScenes/trailer_AP_dist_4.0': 0.2719, 'pts_bbox_NuScenes/trailer_trans_err': 0.8342, 'pts_bbox_NuScenes/trailer_scale_err': 0.2497, 'pts_bbox_NuScenes/trailer_orient_err': 0.7106, 'pts_bbox_NuScenes/trailer_vel_err': 0.3787, 'pts_bbox_NuScenes/trailer_attr_err': 0.1292, 'pts_bbox_NuScenes/barrier_AP_dist_0.5': 0.0777, 'pts_bbox_NuScenes/barrier_AP_dist_1.0': 0.243, 'pts_bbox_NuScenes/barrier_AP_dist_2.0': 0.3221, 'pts_bbox_NuScenes/barrier_AP_dist_4.0': 0.3828, 'pts_bbox_NuScenes/barrier_trans_err': 0.6799, 'pts_bbox_NuScenes/barrier_scale_err': 0.3434, 'pts_bbox_NuScenes/barrier_orient_err': 0.1253, 'pts_bbox_NuScenes/barrier_vel_err': nan, 'pts_bbox_NuScenes/barrier_attr_err': nan, 'pts_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0007, 'pts_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.0158, 'pts_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.0329, 'pts_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.0421, 'pts_bbox_NuScenes/motorcycle_trans_err': 0.6068, 'pts_bbox_NuScenes/motorcycle_scale_err': 0.2729, 'pts_bbox_NuScenes/motorcycle_orient_err': 0.6489, 'pts_bbox_NuScenes/motorcycle_vel_err': 1.6122, 'pts_bbox_NuScenes/motorcycle_attr_err': 0.062, 'pts_bbox_NuScenes/bicycle_AP_dist_0.5': 0.0019, 'pts_bbox_NuScenes/bicycle_AP_dist_1.0': 0.0105, 'pts_bbox_NuScenes/bicycle_AP_dist_2.0': 0.0135, 'pts_bbox_NuScenes/bicycle_AP_dist_4.0': 0.0231, 'pts_bbox_NuScenes/bicycle_trans_err': 0.4428, 'pts_bbox_NuScenes/bicycle_scale_err': 0.292, 'pts_bbox_NuScenes/bicycle_orient_err': 0.5647, 'pts_bbox_NuScenes/bicycle_vel_err': 0.5974, 'pts_bbox_NuScenes/bicycle_attr_err': 0.0764, 'pts_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.0937, 'pts_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.1209, 'pts_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.1402, 'pts_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.181, 'pts_bbox_NuScenes/pedestrian_trans_err': 0.3652, 'pts_bbox_NuScenes/pedestrian_scale_err': 0.3012, 'pts_bbox_NuScenes/pedestrian_orient_err': 1.5208, 'pts_bbox_NuScenes/pedestrian_vel_err': 1.0555, 'pts_bbox_NuScenes/pedestrian_attr_err': 0.438, 'pts_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.0119, 'pts_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.0315, 'pts_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.0559, 'pts_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.1057, 'pts_bbox_NuScenes/traffic_cone_trans_err': 0.6095, 'pts_bbox_NuScenes/traffic_cone_scale_err': 0.411, 'pts_bbox_NuScenes/traffic_cone_orient_err': nan, 'pts_bbox_NuScenes/traffic_cone_vel_err': nan, 'pts_bbox_NuScenes/traffic_cone_attr_err': nan, 'pts_bbox_NuScenes/NDS': 0.2998102882135, 'pts_bbox_NuScenes/mAP': 0.15130378906367253}
"""

_base_ = [
    '../../../configs/_base_/schedules/schedule_2x.py',
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py'
]

class_names = [
    'car'
]
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesDataset'
config_file_prefix = 'full'
workflow = [('train', 1), ('val', 1)]

numC_Trans = 80
multi_adj_frame_id_cfg = (1, 1+1, 1)

# point_cloud_range = [-60, -60, -3, 60, 60, 3]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.4, 0.4, 0.2]
voxel_size_head = [0.1, 0.1, 0.2]

model = dict(
    type='SimpleModel',
    pts_voxel_layer=dict(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=30,
        max_voxels=20000
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=5,
        output_shape=(256, 256),
    ),
    pts_backbone=dict(
        type='CustomResNet',
        with_cp=False,
        numC_input=5,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]
    ),
    pts_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=64
    ),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=64,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size_head[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True
    ),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size_head,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size_head[:2],
            pre_max_size=1000,
            post_max_size=83,

            # Scale-NMS
            nms_thr=0.125,
            nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
            nms_rescale_factor=[0.7, [0.4, 0.6], [0.3, 0.4], 0.9, [1.0, 1.0], [1.5, 2.5]],
        )
    ),
    results_dir='./work_dirs/simple_nus/visualization',
    init_cfg=None
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        use_dim=[0, 1, 2, 3, 4],
        sweeps_num=10,
        remove_close=False,
        file_client_args=dict(backend='disk')
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0]),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        use_dim=[0, 1, 2, 3, 4],
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        use_dim=[0, 1, 2, 3, 4],
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D',class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points'])
]
evaluation = dict(interval=1, pipeline=eval_pipeline)

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/' + config_file_prefix + '/nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        filter_empty_gt=True,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        modality=input_modality
    ),
    val=dict(
        ann_file=data_root + '/' + config_file_prefix + '/nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes = class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality
    ),
    test=dict(
        ann_file=data_root + '/' + config_file_prefix + '/nuscenes_infos_val.pkl',
        pipeline = test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality
    )
)