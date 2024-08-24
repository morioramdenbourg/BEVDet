_base_ = [
    '../../configs/_base_/schedules/schedule_2x.py',
    '../../configs/_base_/datasets/nus-3d.py',
    '../../configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.KPPillarsBEV.kppillarsbev.radar_visualizer',
                               'projects.KPPillarsBEV.kppillarsbev.radar_visualization_hook',
                               'projects.KPPillarsBEV.kppillarsbev.nuscenes_radar_dataset',
                               'projects.KPPillarsBEV.baseline.simple_model',
                               'projects.KPPillarsBEV.baseline.simple_scatter',
                               'projects.KPPillarsBEV.baseline.simple_backbone',
                               'projects.KPPillarsBEV.baseline.simple_preprocessor'],
                      allow_failed_imports=False)

class_names = ['car']
data_root = 'data/nuscenes/'
config_file_prefix = 'full'
log_level = 'INFO'
val_interval = 1
remove_close = None
num_sweeps = 6

filter_empty_gt = False

# Radar Encoding
# radar_use_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] # 18 is a timestamp
# x y z rcs vx_comp vy_comp x_rms y_rms vx_rms vy_rms timestamp
radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17, 18]
voxel_size = [0.5, 0.5, 6]
load_dims = 18
point_cloud_range = [-60, -60, -3, 60, 60, 3]
batch_size = 24
epochs = 50

model = dict(
    type='SimpleModel',
    data_preprocessor=dict(
        type='SimplePreProcessor',
        point_cloud_range=point_cloud_range,
        max_num_points=30,
        voxel_size=voxel_size,
        max_voxels=20000,
        deterministic=True
    ),
    scatter=dict(
        type='SimpleScatter',
        in_channels=len(radar_use_dims),
        output_shape=(128, 128),
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size
    ),
    backbone=dict(
        type='SimpleBackbone',
        in_channels=len(radar_use_dims),
        out_channels=64
    ),
    bbox_head=dict(
        type='CenterHead',
        in_channels=64,
        tasks=[dict(num_classes=len(class_names), class_names=class_names)],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=point_cloud_range,
            out_size_factor=1, # 8
            voxel_size=voxel_size,
            max_num=500
        ),
        separate_head=dict(type='SeparateHead'),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        train_cfg=dict(
            max_objs=500,
            dense_reg=1,
            grid_size=(128, 128),
            out_size_factor=1, # 8
            point_cloud_range=point_cloud_range,
            gaussian_overlap=0.1,
            voxel_size=voxel_size,
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
    sampler=dict(type='DefaultSampler', shuffle=False),
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