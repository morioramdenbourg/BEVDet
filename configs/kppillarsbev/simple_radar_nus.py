"""
mAP: 0.0299
mATE: 0.8847
mASE: 0.7774
mAOE: 0.8270
mAVE: 1.0152
mAAE: 0.6956
NDS: 0.0964
Eval time: 54.5s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.245	0.438	0.191	0.380	0.718	0.213
truck	0.017	0.663	0.340	0.657	0.908	0.221
bus	0.036	0.746	0.244	0.405	1.496	0.130
trailer	0.000	1.000	1.000	1.000	1.000	1.000
construction_vehicle	0.000	1.000	1.000	1.000	1.000	1.000
pedestrian	0.000	1.000	1.000	1.000	1.000	1.000
motorcycle	0.000	1.000	1.000	1.000	1.000	1.000
bicycle	0.000	1.000	1.000	1.000	1.000	1.000
traffic_cone	0.000	1.000	1.000	nan	nan	nan
barrier	0.000	1.000	1.000	1.000	nan	nan
{'pts_bbox_NuScenes/car_AP_dist_0.5': 0.1206, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.2334, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.2878, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.3385, 'pts_bbox_NuScenes/car_trans_err': 0.4381, 'pts_bbox_NuScenes/car_scale_err': 0.1906, 'pts_bbox_NuScenes/car_orient_err': 0.3804, 'pts_bbox_NuScenes/car_vel_err': 0.7182, 'pts_bbox_NuScenes/car_attr_err': 0.2134, 'pts_bbox_NuScenes/mATE': 0.8847, 'pts_bbox_NuScenes/mASE': 0.7774, 'pts_bbox_NuScenes/mAOE': 0.827, 'pts_bbox_NuScenes/mAVE': 1.0152, 'pts_bbox_NuScenes/mAAE': 0.6956, 'pts_bbox_NuScenes/truck_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/truck_AP_dist_1.0': 0.0061, 'pts_bbox_NuScenes/truck_AP_dist_2.0': 0.0186, 'pts_bbox_NuScenes/truck_AP_dist_4.0': 0.0453, 'pts_bbox_NuScenes/truck_trans_err': 0.6629, 'pts_bbox_NuScenes/truck_scale_err': 0.3402, 'pts_bbox_NuScenes/truck_orient_err': 0.6573, 'pts_bbox_NuScenes/truck_vel_err': 0.9076, 'pts_bbox_NuScenes/truck_attr_err': 0.221, 'pts_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/trailer_trans_err': 1.0, 'pts_bbox_NuScenes/trailer_scale_err': 1.0, 'pts_bbox_NuScenes/trailer_orient_err': 1.0, 'pts_bbox_NuScenes/trailer_vel_err': 1.0, 'pts_bbox_NuScenes/trailer_attr_err': 1.0, 'pts_bbox_NuScenes/bus_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/bus_AP_dist_1.0': 0.0208, 'pts_bbox_NuScenes/bus_AP_dist_2.0': 0.052, 'pts_bbox_NuScenes/bus_AP_dist_4.0': 0.071, 'pts_bbox_NuScenes/bus_trans_err': 0.7464, 'pts_bbox_NuScenes/bus_scale_err': 0.2436, 'pts_bbox_NuScenes/bus_orient_err': 0.4052, 'pts_bbox_NuScenes/bus_vel_err': 1.4961, 'pts_bbox_NuScenes/bus_attr_err': 0.1304, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/construction_vehicle_trans_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_scale_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_orient_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_vel_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_attr_err': 1.0, 'pts_bbox_NuScenes/NDS': 0.09644958241452398, 'pts_bbox_NuScenes/mAP': 0.029851322292049093}
"""

_base_ = [
    '../../../configs/_base_/schedules/schedule_2x.py',
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py'
]

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle'
]
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
# class_names = [
#     'car'
# ]
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesRadarDataset'
config_file_prefix = 'full'
workflow = [('train', 1), ('val', 1)]

# log_config = dict(
#     interval=10,
#     hooks=[
#         dict(type="TextLoggerHook"),
#         dict(type="TensorboardLoggerHook"),
#         dict(
#             type="WandbLoggerHook",
#             init_kwargs=dict(project="ProjectName"),
#             log_artifact=False,
#         ),
#     ],
# )

numC_Trans = 80
multi_adj_frame_id_cfg = (1, 1+1, 1)

# point_cloud_range = [-60, -60, -3, 60, 60, 3]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.4, 0.4, 0.2]
voxel_size_head = [0.1, 0.1, 0.2]

# radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17, 18]
radar_use_dim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
sweeps_num = 5
load_dim = 18

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
        in_channels=18,
        output_shape=(256, 256),
    ),
    pts_backbone=dict(
        type='CustomResNet',
        with_cp=False,
        numC_input=18,
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
    results_dir='./vis/simple_radar_nus',
    init_cfg=None
)

train_pipeline = [
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=load_dim,
        sweeps_num=sweeps_num,
        remove_close=None,
        use_dim=radar_use_dim,
        test_mode=False,
        file_client_args=dict(backend='disk')
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0]),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=load_dim,
        sweeps_num=sweeps_num,
        remove_close=None,
        use_dim=radar_use_dim,
        test_mode=True,
        file_client_args=dict(backend='disk')
    ),
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
            dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D',
                 class_names=class_names,
                 with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

eval_pipeline = [
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=load_dim,
        sweeps_num=sweeps_num,
        remove_close=None,
        use_dim=radar_use_dim,
        test_mode=False,
        file_client_args=dict(backend='disk')
    ),
    dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D',class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points'])
]
evaluation = dict(interval=1, pipeline=eval_pipeline)

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=True,
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
        type=dataset_type,
        ann_file=data_root + '/' + config_file_prefix + '/nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes = class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/' + config_file_prefix + '/nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality
    )
)