"""
Evaluating bboxes of pts_bbox
mAP: 0.2960
mATE: 0.7271
mASE: 0.3133
mAOE: 0.8737
mAVE: 0.7794
mAAE: 0.2319
NDS: 0.3554
Eval time: 106.4s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.607	0.395	0.170	0.190	0.739	0.228
truck	0.233	0.661	0.247	0.254	0.657	0.215
bus	0.277	0.825	0.263	0.316	1.513	0.210
trailer	0.090	1.104	0.277	1.040	0.460	0.095
construction_vehicle	0.049	1.066	0.514	1.671	0.118	0.368
pedestrian	0.334	0.733	0.320	1.431	0.869	0.485
motorcycle	0.258	0.639	0.315	1.301	1.385	0.212
bicycle	0.188	0.708	0.346	1.458	0.493	0.042
traffic_cone	0.493	0.512	0.379	nan	nan	nan
barrier	0.431	0.627	0.302	0.201	nan	nan
{'pts_bbox_NuScenes/car_AP_dist_0.5': 0.3108, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.6001, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.7271, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.7892, 'pts_bbox_NuScenes/car_trans_err': 0.3945, 'pts_bbox_NuScenes/car_scale_err': 0.1703, 'pts_bbox_NuScenes/car_orient_err': 0.1902, 'pts_bbox_NuScenes/car_vel_err': 0.7395, 'pts_bbox_NuScenes/car_attr_err': 0.2277, 'pts_bbox_NuScenes/mATE': 0.7271, 'pts_bbox_NuScenes/mASE': 0.3133, 'pts_bbox_NuScenes/mAOE': 0.8737, 'pts_bbox_NuScenes/mAVE': 0.7794, 'pts_bbox_NuScenes/mAAE': 0.2319, 'pts_bbox_NuScenes/truck_AP_dist_0.5': 0.0225, 'pts_bbox_NuScenes/truck_AP_dist_1.0': 0.1637, 'pts_bbox_NuScenes/truck_AP_dist_2.0': 0.3178, 'pts_bbox_NuScenes/truck_AP_dist_4.0': 0.4294, 'pts_bbox_NuScenes/truck_trans_err': 0.661, 'pts_bbox_NuScenes/truck_scale_err': 0.2469, 'pts_bbox_NuScenes/truck_orient_err': 0.254, 'pts_bbox_NuScenes/truck_vel_err': 0.6574, 'pts_bbox_NuScenes/truck_attr_err': 0.215, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0029, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.0628, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.1301, 'pts_bbox_NuScenes/construction_vehicle_trans_err': 1.0658, 'pts_bbox_NuScenes/construction_vehicle_scale_err': 0.5141, 'pts_bbox_NuScenes/construction_vehicle_orient_err': 1.6714, 'pts_bbox_NuScenes/construction_vehicle_vel_err': 0.1179, 'pts_bbox_NuScenes/construction_vehicle_attr_err': 0.3681, 'pts_bbox_NuScenes/bus_AP_dist_0.5': 0.0091, 'pts_bbox_NuScenes/bus_AP_dist_1.0': 0.1644, 'pts_bbox_NuScenes/bus_AP_dist_2.0': 0.4145, 'pts_bbox_NuScenes/bus_AP_dist_4.0': 0.5209, 'pts_bbox_NuScenes/bus_trans_err': 0.8254, 'pts_bbox_NuScenes/bus_scale_err': 0.2631, 'pts_bbox_NuScenes/bus_orient_err': 0.3163, 'pts_bbox_NuScenes/bus_vel_err': 1.5129, 'pts_bbox_NuScenes/bus_attr_err': 0.2098, 'pts_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_1.0': 0.0055, 'pts_bbox_NuScenes/trailer_AP_dist_2.0': 0.0884, 'pts_bbox_NuScenes/trailer_AP_dist_4.0': 0.2668, 'pts_bbox_NuScenes/trailer_trans_err': 1.1042, 'pts_bbox_NuScenes/trailer_scale_err': 0.2767, 'pts_bbox_NuScenes/trailer_orient_err': 1.04, 'pts_bbox_NuScenes/trailer_vel_err': 0.46, 'pts_bbox_NuScenes/trailer_attr_err': 0.0949, 'pts_bbox_NuScenes/barrier_AP_dist_0.5': 0.1256, 'pts_bbox_NuScenes/barrier_AP_dist_1.0': 0.4049, 'pts_bbox_NuScenes/barrier_AP_dist_2.0': 0.5581, 'pts_bbox_NuScenes/barrier_AP_dist_4.0': 0.6366, 'pts_bbox_NuScenes/barrier_trans_err': 0.6272, 'pts_bbox_NuScenes/barrier_scale_err': 0.3024, 'pts_bbox_NuScenes/barrier_orient_err': 0.2013, 'pts_bbox_NuScenes/barrier_vel_err': nan, 'pts_bbox_NuScenes/barrier_attr_err': nan, 'pts_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0631, 'pts_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.2516, 'pts_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.3352, 'pts_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.3816, 'pts_bbox_NuScenes/motorcycle_trans_err': 0.6394, 'pts_bbox_NuScenes/motorcycle_scale_err': 0.3145, 'pts_bbox_NuScenes/motorcycle_orient_err': 1.3007, 'pts_bbox_NuScenes/motorcycle_vel_err': 1.3854, 'pts_bbox_NuScenes/motorcycle_attr_err': 0.2121, 'pts_bbox_NuScenes/bicycle_AP_dist_0.5': 0.0337, 'pts_bbox_NuScenes/bicycle_AP_dist_1.0': 0.1529, 'pts_bbox_NuScenes/bicycle_AP_dist_2.0': 0.2651, 'pts_bbox_NuScenes/bicycle_AP_dist_4.0': 0.2993, 'pts_bbox_NuScenes/bicycle_trans_err': 0.7079, 'pts_bbox_NuScenes/bicycle_scale_err': 0.3457, 'pts_bbox_NuScenes/bicycle_orient_err': 1.4581, 'pts_bbox_NuScenes/bicycle_vel_err': 0.4933, 'pts_bbox_NuScenes/bicycle_attr_err': 0.0421, 'pts_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.0986, 'pts_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.2773, 'pts_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.4245, 'pts_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.5339, 'pts_bbox_NuScenes/pedestrian_trans_err': 0.7329, 'pts_bbox_NuScenes/pedestrian_scale_err': 0.3198, 'pts_bbox_NuScenes/pedestrian_orient_err': 1.4315, 'pts_bbox_NuScenes/pedestrian_vel_err': 0.869, 'pts_bbox_NuScenes/pedestrian_attr_err': 0.4853, 'pts_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.2155, 'pts_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.4757, 'pts_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.6, 'pts_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.6792, 'pts_bbox_NuScenes/traffic_cone_trans_err': 0.5124, 'pts_bbox_NuScenes/traffic_cone_scale_err': 0.3791, 'pts_bbox_NuScenes/traffic_cone_orient_err': nan, 'pts_bbox_NuScenes/traffic_cone_vel_err': nan, 'pts_bbox_NuScenes/traffic_cone_attr_err': nan, 'pts_bbox_NuScenes/NDS': 0.35544544758242236, 'pts_bbox_NuScenes/mAP': 0.2959594075402994}
"""

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_2x.py']

## Dataset
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesRadarDataset'
config_file_prefix = 'full'
# config_file_prefix = 'small'
workflow = [('train', 1), ('val', 1)]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

batch_size = 14
# optimizer = dict(type='AdamW', lr=5e-5 * batch_size, weight_decay=0.01) # 5e-5 * batch_size for radar, same weight decay
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(
    grad_clip=dict(
        max_norm=5,
        norm_type=2
    )
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,]
)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
file_client_args = dict(backend='disk')
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False
)
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)
radar_use_dim = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17, 18]
sweeps_num = 5
load_dim = 18

## Image Path
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

## Head
voxel_size_head = [0.1, 0.1, 0.2]

## kpconv
kp_influence_fn = 'gaussian' # influence decreases exponentially with distance
aggregation_mode = 'closest' # for sparse where nearest neighbor is most important
in_points_dim = 3
neighborhood_limit = 30
num_kernel_points = 15 # number of kernel points around each anchor
fixed_kernel_points = 'center' # kernel points are fixed to the center of the kernel
use_batch_norm = False
batch_norm_momentum = 0.98
modulated = False
use_group_norm = True
num_groups = 16

# view transformer
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

scale_config = [
    dict(
        # kpbev
        radius=0.3,
        KP_extent=0.12,
        conv_radius=0.3,
        output_shape=(128, 128),
        voxel_size=[0.8, 0.8, 8],
        # resnet backbone
        num_layer=[2],
        stride=[1],
        num_channels=[64]
    ),
    dict(
        # kpbev
        radius=0.6,
        KP_extent=0.24,
        conv_radius=0.6,
        output_shape=(64, 64),
        voxel_size=[1.6, 1.6, 8],
        # resnet backbone
        num_layer=[2],
        stride=[1],
        num_channels=[64 * 2]
    ),
    dict(
        # kpbev
        radius=1.2,
        KP_extent=0.48,
        conv_radius=1.2,
        output_shape=(32, 32),
        voxel_size=[3.2, 3.2, 8],
        # resnet backbone
        num_layer=[2],
        stride=[1],
        num_channels=[64 * 4]
    ),
    dict(
        # kpbev
        radius=2.4,
        KP_extent=0.96,
        conv_radius=2.4,
        output_shape=(16, 16),
        voxel_size=[6.4, 6.4, 8],
        # resnet backbone
        num_layer=[2],
        stride=[1],
        num_channels=[64 * 8]
    )
]

## voxels
max_num_points=30
max_voxels=20000

model = dict(
    type='BEVDetKPPillarsBEV',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/models/bevdet-r50-cbgs.pth",
            prefix="img_backbone."
        )
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/models/bevdet-r50-cbgs.pth",
            prefix="img_neck."
        ),
        freeze_lateral=True,
        freeze_fpn_convs=True
    ),
    img_view_transformer=dict(
        type='CustomLSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=64,
        downsample=16
    ),
    pts_voxel_layers=[
        dict(
            voxel_size=config['voxel_size'],
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels
        ) for config in scale_config
    ],
    pts_voxel_encoders=[
        dict(
            type='KPBEVEncoder',
            in_channels=len(radar_use_dim) - 3,
            out_channels=64,
            point_cloud_range=point_cloud_range,
            voxel_size=config['voxel_size'],
            norm_cfg=dict(type='GN', num_groups=num_groups),
            kpconv_args=dict(
                block_name=f'KPBEVEncoder_KPConv_S{index}',
                in_dim=64,
                out_dim=64,
                radius=config['radius'],
                norm_cfg=dict(type='GN', num_groups=num_groups),
                config=dict(
                    KP_extent=config['KP_extent'],
                    conv_radius=config['conv_radius'],
                    in_points_dim=in_points_dim,
                    modulated=modulated,
                    num_kernel_points=num_kernel_points,
                    fixed_kernel_points=fixed_kernel_points,
                    KP_influence=kp_influence_fn,
                    aggregation_mode=aggregation_mode,
                    neighborhood_limit=neighborhood_limit
                )
            ),
            init_cfg=dict(
                type="Pretrained",
                checkpoint="work_dirs/models/bevdet_kppillarsbev__multi_scale__frozen_rgb.pth",
                prefix=f"pts_voxel_encoders.{index}"
            ),
        ) for index, config in enumerate(scale_config)
    ],
    pts_middle_encoders=[
        dict(
            type='PointPillarsScatter',
            in_channels=64,
            output_shape=config['output_shape']
        ) for config in scale_config
    ],
    pts_backbones=[
        dict(
            type='CustomResNet',
            with_cp=False,
            norm_cfg=dict(type='GN', num_groups=num_groups),
            numC_input=64,
            num_layer=config['num_layer'],
            stride=config['stride'],
            num_channels=config['num_channels'],
            init_cfg=dict(
                type="Pretrained",
                checkpoint="work_dirs/models/bevdet_kppillarsbev__multi_scale__frozen_rgb.pth",
                prefix=f"pts_backbones.{index}"
            )
        ) for index, config in enumerate(scale_config)
    ],
    pts_neck=dict(
        type='KPPillarsBEVFPN',
        in_channels=[64, 64 * 2, 64 * 4, 64 * 8],
        out_channels=64,
        input_feature_index=(0, 1, 2, 3),
        scale_factors=[1, 2, 4, 8],
        norm_cfg=dict(type='GN', num_groups=num_groups),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/models/bevdet_kppillarsbev__multi_scale__frozen_rgb.pth",
            prefix="pts_neck."
        )
    ),
    fusion_norm=dict(
        radar=dict(
            normalized_shape=[64, 128, 128],
            elementwise_affine=False,
        ),
        rgb=dict(
            normalized_shape=[64, 128, 128],
            elementwise_affine=False,
        )
    ),
    fusion_layer=dict(
        type='BEVDetFuser',
        in_channels=[64, 64],
        out_channels=64,
        norm_cfg=dict(type='GN', num_groups=num_groups),
    ),
    fusion_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=64,
        num_channels=[64 * 2, 64 * 4, 64 * 8],
        norm_cfg=dict(type='GN', num_groups=num_groups),
    ),
    fusion_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=64 * 8 + 64 * 2,
        out_channels=256,
        norm_cfg=dict(type='GN', num_groups=num_groups),
    ),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=class_names),
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
            out_size_factor=8,
            voxel_size=voxel_size_head[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        norm_cfg=dict(type='GN', num_groups=num_groups)
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size_head,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size_head[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

train_pipeline = [
    dict(
        type='CustomPrepareImageInput',
        is_train=True,
        data_config=data_config
    ),
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
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5, sync_2d=False),
    dict(type='LoadImageAnnotations', classes=class_names),
    dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
    # dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='CustomPrepareImageInput', data_config=data_config),
    dict(type='LoadImageAnnotations', classes=class_names),
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
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]
            ),
            dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False
            ),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ]
    )
]

eval_pipeline = [
    dict(type='CustomPrepareImageInput', data_config=data_config),
    dict(type='LoadImageAnnotations', classes=class_names),
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
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False
    ),
    dict(type='Collect3D', keys=['points', 'img_inputs'])
]

data = dict(
    samples_per_gpu=batch_size,
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
        modality=input_modality,
        img_info_prototype='bevdet',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/' + config_file_prefix + '/nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes = class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality,
        img_info_prototype='bevdet',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/' + config_file_prefix + '/nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality,
        img_info_prototype='bevdet',
    )
)

evaluation = dict(interval=1, pipeline=eval_pipeline)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(project="bevdet_kppillarsbev"),
            interval=50,
            log_artifact=False,
        ),
    ],
)