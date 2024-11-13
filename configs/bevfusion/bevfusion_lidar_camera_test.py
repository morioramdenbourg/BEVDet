_base_ = [
    '../_base_/schedules/schedule_2x.py',
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py'
]

# dataset & training
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesRadarDataset'
# config_file_prefix = 'full'
config_file_prefix = 'small'
workflow = [('train', 1), ('val', 1)]
batch_size = 1
optimizer = dict(type='AdamW', lr=5e-5 * batch_size, weight_decay=0.01)

# voxels
# point_cloud_range = [-60, -60, -5.0, 630, 60, 3.0] # set by kppillarsbev
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
max_num_points = 30
max_voxels = 20000
voxel_size_head = [0.1, 0.1, 0.2]
# voxel_sizes = [
#     [0.4, 0.4, 8],
#     [0.8, 0.8, 8],
#     [1.6, 1.6, 8],
#     [3.2, 3.2, 8]
# ]
voxel_size = [0.075, 0.075, 0.2]

# radar
radar_use_dim = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17, 18]
sweeps_num = 5
load_dim = 18

# kpconv
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

# image
backend_args = None
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
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

model = dict(
    type='BEVFusion',
    # pts_voxel_layer=dict(
    #     max_num_points=10,
    #     point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    #     # voxel_size=[0.075, 0.075, 0.2],
    #     voxel_size=[0.6, 0.6, 0.2],
    #     max_voxels=[120000, 160000],
    # ),
    # pts_middle_encoder=dict(
    #     type='PointPillarsScatter',
    #     in_channels=5,
    #     output_shape=(180, 180),
    # ),
    # pts_backbone = dict(
    #     type='SECOND',
    #     # in_channels=256,
    #     in_channels=5,
    #     out_channels=[128, 256],
    #     layer_nums=[5, 5],
    #     layer_strides=[1, 2],
    #     norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
    #     conv_cfg=dict(type='Conv2d', bias=False)
    # ),
    # pts_neck = dict(
    #     type='SECONDFPN',
    #     in_channels=[128, 256],
    #     out_channels=[256, 256],
    #     upsample_strides=[1, 2],
    #     norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
    #     upsample_cfg=dict(type='deconv', bias=False),
    #     use_conv_for_no_stride=True
    # ),
    # img_backbone=dict(
    #     type='mmdet.SwinTransformer',
    #     embed_dims=96,
    #     depths=[2, 2, 6, 2],
    #     num_heads=[3, 6, 12, 24],
    #     window_size=7,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     qk_scale=None,
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     drop_path_rate=0.2,
    #     patch_norm=True,
    #     out_indices=[1, 2, 3],
    #     with_cp=False,
    #     convert_weights=True,
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint=  # noqa: E251
    #         'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
    #         # noqa: E501
    #     )),
    img_backbone=dict(
        type='mmdet.ResNet',
        pretrained='torchvision://resnet18',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'
    ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[128, 256, 512],
        out_channels=64,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=64,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3], # 0.3 -> 0.4 pc range of bevdet
        ybound=[-54.0, 54.0, 0.3], # 0.3 -> 0.4 pc range of bevdet
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5], # 0.5 -> 1.0 depth bin
        downsample=2 # 0.4 for downsample
    ),
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256
    ),
    pts_bbox_head = dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)
        ),
        train_cfg=dict(
            dataset='nuScenes',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            grid_size=[1440, 1440, 41],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15
                ),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            )
        ),
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 41],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-54.0, -54.0],
            nms_type=None
        ),
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=10
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)
    )
)

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=None,
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
    # dict(
    #     type='ImageAug3D',
    #     final_dim=[256, 704],
    #     resize_lim=[0.48, 0.48],
    #     bot_pct_lim=[0.0, 0.0],
    #     rot_lim=[0.0, 0.0],
    #     rand_flip=False,
    #     is_train=False
    # ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True
    ),
    # dict(
    #     type='BEVFusionGlobalRotScaleTrans',
    #     scale_ratio_range=[0.9, 1.1],
    #     rot_range=[-0.78539816, 0.78539816],
    #     translation_std=0.5
    # ),
    # dict(type='BEVFusionRandomFlip3D'),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     scale_ratio_range=[0.9, 1.1],
    #     rot_range=[-0.78539816, 0.78539816],
    #     translation_std=0.5
    # ),
    # dict(type='BEVFusionRandomFlip3D'),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
    ),
    # dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ]
    )
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args
    ),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'num_views'
        ]
    )
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=1,
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