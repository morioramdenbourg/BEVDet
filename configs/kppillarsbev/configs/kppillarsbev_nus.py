_base_ = [
    '../../../configs/_base_/schedules/schedule-2x.py',
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=[
        'projects.KPPillarsBEV.kppillarsbev.radar_visualizer',
        'projects.KPPillarsBEV.kppillarsbev.nuscenes_radar_dataset',
        'projects.KPPillarsBEV.kppillarsbev.radar_visualization_hook',
        'projects.KPPillarsBEV.kppillarsbev.kppillarsbev',
        'projects.KPPillarsBEV.kppillarsbev.kpconv_preprocessor',
        'projects.KPPillarsBEV.kppillarsbev.kpbev_encoder',
        'projects.KPPillarsBEV.kppillarsbev.kppillarsbev_backbone',
        'projects.KPPillarsBEV.kppillarsbev.kppillarsbev_neck'
    ],
    allow_failed_imports=False
)

val_interval = 1
log_level = 'DEBUG'

# nuscenes guidelines
class_names = ['car']
data_root = 'data/nuscenes/'
config_file_prefix = 'small-2'
filter_empty_gt = False
shuffle = False

# radar point cloud guidelines
# x y z rcs vx_comp vy_comp x_rms y_rms vx_rms vya_rms timestamp
radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17, 18]
radar_use_features = radar_use_dims[3:]
load_dims = 18
remove_close = None
num_sweeps = 6

# kppillarsbev guidelines
point_cloud_range = [-60, -60, -3, 60, 60, 3]
spatial_scale = 0.5 # initial
features_out = 64
batch_size = 24
epochs = 50

# kppillarsbev backbone guidelines
backbone_in_channels = [features_out, features_out, features_out, features_out]
backbone_out_channels = [features_out, features_out, features_out, features_out]
backbone_layers = [3, 6, 6, 3]

# kppillarsbev neck guidelines
neck_in_channels = [features_out, features_out, features_out, features_out]

# resnet guidelines
grid_size = (224, 224) # common for ResNet

# kpconv guidelines
num_kernels = 8
kernel_size = 5.0
influence_radius = 1.0
neighborhood_limit = 20  # TODO: experiment or calculate
groups = 1
dim_3d = 3
bias = False

model = dict(
    type='KPPillarsBEV',
    data_preprocessor=dict(
        type='KPConvPreprocessor',
        kpconv_args=dict(
            in_channels=len(radar_use_features),
            out_channels=len(radar_use_features),
            num_kernels=num_kernels,
            kernel_size=kernel_size,
            influence_radius=influence_radius,
            neighborhood_limit=neighborhood_limit,
            groups=groups,
            bias=bias,
            dimension=dim_3d
        )
    ),
    encoder=dict(
        type='KPBEVEncoder',
        in_channels=len(radar_use_features),
        out_channels=features_out,
        spatial_scale=spatial_scale,
        point_cloud_range=point_cloud_range,
        grid_size=grid_size,
        max_num_points_per_cell=30,
        kpconv_args=dict(
            in_channels=features_out,
            out_channels=features_out,
            num_kernels=num_kernels,
            kernel_size=kernel_size,
            influence_radius=influence_radius,
            neighborhood_limit=neighborhood_limit,
            groups=groups,
            bias=bias,
            dimension=dim_3d
        )
    ),
    backbone=dict(
        type='KPPillarsBEVBackbone',
        in_channels=backbone_in_channels,
        out_channels=backbone_out_channels,
        layers=backbone_layers,
    ),
    neck=dict(
        type='KPPillarsBEVNeck',
        in_channels=neck_in_channels,
        out_channels=features_out
    ),
    bbox_head=dict(
        type='CenterHead',
        in_channels=features_out,
        tasks=[dict(num_classes=len(class_names), class_names=class_names)],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=point_cloud_range,
            out_size_factor=4,
            voxel_size=[spatial_scale, spatial_scale, 6],
            max_num=500
        ),
        separate_head=dict(type='SeparateHead'),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        train_cfg=dict(
            max_objs=500,
            dense_reg=1,
            grid_size=grid_size,
            out_size_factor=4,
            point_cloud_range=point_cloud_range,
            gaussian_overlap=0.1,
            voxel_size=[spatial_scale, spatial_scale, 6],
            code_weights=[1.0],
            min_radius=2,
        ),
        norm_bbox=True,
        test_cfg=dict(
            nms_type='rotate',
            post_center_limit_range=point_cloud_range,
            score_threshold=0.1,
            nms_thr=0.2,
            pre_max_size=1000,
            post_max_size=500,
        ),
    ),
    store_output=True,
    init_cfg=None
)

train_pipeline = [
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=load_dims,
        sweeps_num=num_sweeps,
        use_dim=radar_use_dims,
        remove_close=remove_close,
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
    dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=val_interval)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=shuffle),
    dataset=dict(type='NuScenesRadarDataset',
                 data_root=data_root,
                 ann_file=f'{config_file_prefix}/nuscenes_infos_train.pkl',
                 pipeline=train_pipeline,
                 metainfo=dict(classes=class_names),
                 filter_empty_gt=filter_empty_gt,
    )
)


val_pipeline = [
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=load_dims,
        sweeps_num=0,
        use_dim=radar_use_dims,
        remove_close=remove_close,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]
val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(type='NuScenesRadarDataset',
                 data_root=data_root,
                 ann_file=f'{config_file_prefix}/nuscenes_infos_val.pkl',
                 test_mode=True,
                 pipeline=val_pipeline,
                 metainfo=dict(classes=class_names),
                 filter_empty_gt=filter_empty_gt,
    )
)
val_evaluator = dict(ann_file=data_root + f'{config_file_prefix}/nuscenes_infos_val.pkl')

test_pipeline = [
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=load_dims,
        sweeps_num=0,
        use_dim=radar_use_dims,
        remove_close=None
    ),
    dict(type='Pack3DDetInputs', keys=['points'])
]
test_dataloader = dict(
    batch_size=1,
    dataset=dict(type='NuScenesRadarDataset',
                 data_root=data_root,
                 ann_file=f'{config_file_prefix}/nuscenes_infos_test.pkl',
                 pipeline=test_pipeline,
                 test_mode=True,
                 metainfo=dict(classes=class_names)
    )
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='RadarVisualizer', name='RadarVisualizer')
default_hooks = dict(visualization=dict(type='RadarVisualizationHook'))