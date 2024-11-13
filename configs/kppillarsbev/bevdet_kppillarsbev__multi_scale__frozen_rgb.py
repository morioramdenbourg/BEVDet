"""
Evaluating bboxes of pts_bbox
mAP: 0.0574
mATE: 0.9403
mASE: 0.9175
mAOE: 0.9131
mAVE: 0.9557
mAAE: 0.9027
NDS: 0.0658
Eval time: 50.1s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.574	0.403	0.175	0.218	0.645	0.221
truck	0.000	1.000	1.000	1.000	1.000	1.000
bus	0.000	1.000	1.000	1.000	1.000	1.000
trailer	0.000	1.000	1.000	1.000	1.000	1.000
construction_vehicle	0.000	1.000	1.000	1.000	1.000	1.000
pedestrian	0.000	1.000	1.000	1.000	1.000	1.000
motorcycle	0.000	1.000	1.000	1.000	1.000	1.000
bicycle	0.000	1.000	1.000	1.000	1.000	1.000
traffic_cone	0.000	1.000	1.000	nan	nan	nan
barrier	0.000	1.000	1.000	1.000	nan	nan
{'pts_bbox_NuScenes/car_AP_dist_0.5': 0.2996, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.5682, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.6843, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.7444, 'pts_bbox_NuScenes/car_trans_err': 0.4027, 'pts_bbox_NuScenes/car_scale_err': 0.1753, 'pts_bbox_NuScenes/car_orient_err': 0.2179, 'pts_bbox_NuScenes/car_vel_err': 0.6452, 'pts_bbox_NuScenes/car_attr_err': 0.2213, 'pts_bbox_NuScenes/mATE': 0.9403, 'pts_bbox_NuScenes/mASE': 0.9175, 'pts_bbox_NuScenes/mAOE': 0.9131, 'pts_bbox_NuScenes/mAVE': 0.9557, 'pts_bbox_NuScenes/mAAE': 0.9027, 'pts_bbox_NuScenes/NDS': 0.06578576951717732, 'pts_bbox_NuScenes/mAP': 0.05741418799266138}
"""

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_2x.py']

## Dataset
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesRadarDataset'
config_file_prefix = 'full'
# config_file_prefix = 'small'
workflow = [('train', 1), ('val', 1)]
# For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
class_names = [
    'car'
]
batch_size = 14
optimizer = dict(type='AdamW', lr=5e-5 * batch_size, weight_decay=0.01) # 5e-5 * batch_size for radar, same weight decay
# # Optimizer
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-60, -60, -5.0, 630, 60, 3.0] # set by kppillarsbev
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
    flip_dy_ratio=0.5)
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

scale_config = [
    dict(
        radius=0.3,
        KP_extent=0.12,
        conv_radius=0.3,
        output_shape=(128, 128),
        voxel_size=[0.8, 0.8, 8],
        # resnet backbone
        num_layer=[2],
        stride=[1],
        num_channels=[64],
        # view transformer
        grid_config={
            'x': [-51.2, 51.2, 0.8],
            'y': [-51.2, 51.2, 0.8],
            'z': [-5, 3, 8],
            'depth': [1.0, 60.0, 1.0],
        },
        # fusion
        fusion_in_channels=[64, 64],
        fusion_out_channels=64,
    ),
    dict(
        radius=0.6,
        KP_extent=0.24,
        conv_radius=0.6,
        output_shape=(64, 64),
        voxel_size=[1.6, 1.6, 8],
        # resnet backbone
        num_layer=[2],
        stride=[1],
        num_channels=[64 * 2],
        # view transformer
        grid_config={
            'x': [-51.2, 51.2, 1.6],
            'y': [-51.2, 51.2, 1.6],
            'z': [-5, 3, 8],
            'depth': [1.0, 60.0, 1.0],
        },
        # fusion
        fusion_in_channels=[64, 64 * 2],
        fusion_out_channels=64 * 2,
    ),
    dict(
        radius=1.2,
        KP_extent=0.48,
        conv_radius=1.2,
        output_shape=(32, 32),
        voxel_size=[3.2, 3.2, 8],
        # resnet backbone
        num_layer=[2],
        stride=[1],
        num_channels=[64 * 4],
        # view transformer
        grid_config={
            'x': [-51.2, 51.2, 3.2],
            'y': [-51.2, 51.2, 3.2],
            'z': [-5, 3, 8],
            'depth': [1.0, 60.0, 1.0],
        },
        # fusion
        fusion_in_channels=[64, 64 * 4],
        fusion_out_channels=64 * 4,
    ),
    dict(
        radius=2.4,
        KP_extent=0.96,
        conv_radius=2.4,
        output_shape=(16, 16),
        voxel_size=[6.4, 6.4, 8],
        # resnet backbone
        num_layer=[2],
        stride=[1],
        num_channels=[64 * 8],
        # view transformer
        grid_config={
            'x': [-51.2, 51.2, 6.4],
            'y': [-51.2, 51.2, 6.4],
            'z': [-5, 3, 8],
            'depth': [1.0, 60.0, 1.0],
        },
        # fusion
        fusion_in_channels=[64, 64 * 8],
        fusion_out_channels=64 * 8
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
        # norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/models/bevdet_kppillarsbev__single_scale__frozen_rgb.pth", # from bevdet config
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
        norm_cfg=dict(type='BN', requires_grad=False),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/models/bevdet_kppillarsbev__single_scale__frozen_rgb.pth",
            prefix="img_neck."
        ),
        # freeze_lateral=True,
        # freeze_fpn_convs=True
    ),
    img_view_transformers=[
        dict(
            type='CustomLSSViewTransformer', # may not be necssary, could change depth net per scale, # LSSViewTransformerBEVDepth
            grid_config=config['grid_config'],
            input_size=data_config['input_size'],
            in_channels=256,
            out_channels=64,
            downsample=16 # downsample for other scales
        ) for config in scale_config
    ],
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
            # freeze=False,
            init_cfg=dict(
                type="Pretrained",
                checkpoint="work_dirs/models/bevdet_kppillarsbev__single_scale__frozen_rgb.pth",
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
            # freeze=False,
            init_cfg=dict(
                type="Pretrained",
                checkpoint="work_dirs/models/bevdet_kppillarsbev__single_scale__frozen_rgb.pth",
                prefix=f"pts_backbones.{index}"
            )
        ) for index, config in enumerate(scale_config)
    ],
    fusion_layers=[
        dict(
            type='BEVDetFuser',
            in_channels=config['fusion_in_channels'],
            out_channels=config['fusion_out_channels'],
            norm_cfg=dict(type='GN', num_groups=num_groups),
        ) for config in scale_config
    ],
    pts_neck=dict(
        type='KPPillarsBEVFPN',
        in_channels=[64, 64 * 2, 64 * 4, 64 * 8],
        out_channels=64,
        input_feature_index=(0, 1, 2, 3),
        scale_factors=[1, 2, 4, 8],
        norm_cfg=dict(type='GN', num_groups=num_groups),
        # freeze=False,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/models/bevdet_kppillarsbev__single_scale__frozen_rgb.pth",
            prefix="pts_neck."
        )
    ),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=64,
        tasks=[
            dict(num_class=1, class_names=['car']),
        ],
        # tasks=[
        #     dict(num_class=10, class_names=['car',
        #                                     'truck',
        #                                     'construction_vehicle',
        #                                     'bus',
        #                                     'trailer',
        #                                     'barrier',
        #                                     'motorcycle',
        #                                     'bicycle',
        #                                     'pedestrian',
        #                                     'traffic_cone']
        #          ),
        # ],
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