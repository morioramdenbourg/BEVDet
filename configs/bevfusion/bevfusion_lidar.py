_base_ = [
    '../_base_/schedules/schedule_2x.py',
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py'
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesRadarDataset'
# config_file_prefix = 'full'
config_file_prefix = 'small'
workflow = [('train', 1), ('val', 1)]
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')

batch_size = 4
optimizer = dict(type='AdamW', lr=5e-5 * batch_size, weight_decay=0.01) # 5e-5 * batch_size for radar, same weight decay

# voxels
# point_cloud_range = [-60, -60, -5.0, 630, 60, 3.0] # set by kppillarsbev
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
max_num_points=30
max_voxels=20000
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

# bevfusion
backend_args = None

model = dict(
    type='BEVFusion',
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        voxel_size=[0.075, 0.075, 0.2],
        max_voxels=[120000, 160000],
    ),
    pts_middle_encoder = dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1440, 1440, 41],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'
    ),
    pts_backbone = dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    pts_neck = dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    bbox_head = dict(
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
                type='TransformerDecoderLayer',
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

# model = dict(
#     type='KPPillarsBEV',
#     pts_preprocessors=[],
#     pts_voxel_layers=[
#         dict(
#             voxel_size=voxel_sizes[0],
#             point_cloud_range=point_cloud_range,
#             max_num_points=max_num_points,
#             max_voxels=max_voxels
#         ),
#         dict(
#             voxel_size=voxel_sizes[1],
#             point_cloud_range=point_cloud_range,
#             max_num_points=max_num_points,
#             max_voxels=max_voxels
#         ),
#         dict(
#             voxel_size=voxel_sizes[2],
#             point_cloud_range=point_cloud_range,
#             max_num_points=max_num_points,
#             max_voxels=max_voxels
#         ),
#         dict(
#             voxel_size=voxel_sizes[3],
#             point_cloud_range=point_cloud_range,
#             max_num_points=max_num_points,
#             max_voxels=max_voxels
#         )
#     ],
#     pts_voxel_encoders=[
#         dict(
#             type='KPBEVEncoder',
#             in_channels=len(radar_use_dim) - 3,
#             out_channels=64,
#             point_cloud_range=point_cloud_range,
#             voxel_size=voxel_sizes[0],
#             norm_cfg=dict(type='GN', num_groups=num_groups),
#             kpconv_args=dict(
#                 block_name='KPBEVEncoder_KPConv_S0',
#                 in_dim=64,
#                 out_dim=64,
#                 radius=0.6,
#                 config=dict(
#                     KP_extent=0.24,
#                     conv_radius=0.6,
#                     in_points_dim=in_points_dim,
#                     modulated=modulated,
#                     batch_norm_momentum=batch_norm_momentum,
#                     use_batch_norm=use_batch_norm,
#                     use_group_norm=use_group_norm,
#                     num_groups=num_groups,
#                     num_kernel_points=num_kernel_points,
#                     fixed_kernel_points=fixed_kernel_points,
#                     KP_influence=kp_influence_fn,
#                     aggregation_mode=aggregation_mode,
#                     neighborhood_limit=neighborhood_limit
#                 )
#             )
#         ),
#         dict(
#             type='KPBEVEncoder',
#             in_channels=len(radar_use_dim) - 3,
#             out_channels=64,
#             point_cloud_range=point_cloud_range,
#             voxel_size=voxel_sizes[1],
#             norm_cfg=dict(type='GN', num_groups=num_groups),
#             kpconv_args=dict(
#                 block_name='KPBEVEncoder_KPConv_S1',
#                 in_dim=64,
#                 out_dim=64,
#                 radius=1.2,
#                 config=dict(
#                     KP_extent=0.48,
#                     conv_radius=1.2,
#                     in_points_dim=in_points_dim,
#                     modulated=modulated,
#                     batch_norm_momentum=batch_norm_momentum,
#                     use_batch_norm=use_batch_norm,
#                     use_group_norm=use_group_norm,
#                     num_groups=num_groups,
#                     num_kernel_points=num_kernel_points,
#                     fixed_kernel_points=fixed_kernel_points,
#                     KP_influence=kp_influence_fn,
#                     aggregation_mode=aggregation_mode,
#                     neighborhood_limit=neighborhood_limit
#                 )
#             )
#         ),
#         dict(
#             type='KPBEVEncoder',
#             in_channels=len(radar_use_dim) - 3,
#             out_channels=64,
#             point_cloud_range=point_cloud_range,
#             voxel_size=voxel_sizes[2],
#             norm_cfg=dict(type='GN', num_groups=num_groups),
#             kpconv_args=dict(
#                 block_name='KPBEVEncoder_KPConv_S2',
#                 in_dim=64,
#                 out_dim=64,
#                 radius=2.4,
#                 config=dict(
#                     KP_extent=0.96,
#                     conv_radius=2.4,
#                     in_points_dim=in_points_dim,
#                     modulated=modulated,
#                     batch_norm_momentum=batch_norm_momentum,
#                     use_batch_norm=use_batch_norm,
#                     use_group_norm=use_group_norm,
#                     num_groups=num_groups,
#                     num_kernel_points=num_kernel_points,
#                     fixed_kernel_points=fixed_kernel_points,
#                     KP_influence=kp_influence_fn,
#                     aggregation_mode=aggregation_mode,
#                     neighborhood_limit=neighborhood_limit
#                 )
#             )
#         ),
#         dict(
#             type='KPBEVEncoder',
#             in_channels=len(radar_use_dim) - 3,
#             out_channels=64,
#             point_cloud_range=point_cloud_range,
#             voxel_size=voxel_sizes[3],
#             norm_cfg=dict(type='GN', num_groups=num_groups),
#             kpconv_args=dict(
#                 block_name='KPBEVEncoder_KPConv_S3',
#                 in_dim=64,
#                 out_dim=64,
#                 radius=4.8,
#                 config=dict(
#                     KP_extent=1.92,
#                     conv_radius=4.8,
#                     in_points_dim=in_points_dim,
#                     modulated=modulated,
#                     batch_norm_momentum=batch_norm_momentum,
#                     use_batch_norm=use_batch_norm,
#                     use_group_norm=use_group_norm,
#                     num_groups=num_groups,
#                     num_kernel_points=num_kernel_points,
#                     fixed_kernel_points=fixed_kernel_points,
#                     KP_influence=kp_influence_fn,
#                     aggregation_mode=aggregation_mode,
#                     neighborhood_limit=neighborhood_limit
#                 )
#             )
#         )
#     ],
#     pts_middle_encoders=[
#         dict(
#             type='PointPillarsScatter',
#             in_channels=64,
#             output_shape=(256, 256), # (256, 256)
#         ),
#         dict(
#             type='PointPillarsScatter',
#             in_channels=64,
#             output_shape=(128, 128), # (128, 128)
#         ),
#         dict(
#             type='PointPillarsScatter',
#             in_channels=64,
#             output_shape=(64, 64),  # (64, 64)
#         ),
#         dict(
#             type='PointPillarsScatter',
#             in_channels=64,
#             output_shape=(32, 32),  # (32, 32)
#         ),
#     ],
#     pts_backbones=[
#         dict(
#             type='CustomResNet',
#             with_cp=False,
#             norm_cfg=dict(type='GN', num_groups=num_groups),
#             numC_input=64,
#             num_layer=[2],
#             stride=[1],
#             num_channels=[64]
#         ),
#         dict(
#             type='CustomResNet',
#             with_cp=False,
#             norm_cfg=dict(type='GN', num_groups=num_groups),
#             numC_input=64,
#             num_layer=[2],
#             stride=[1],
#             num_channels=[64 * 2]
#         ),
#         dict(
#             type='CustomResNet',
#             with_cp=False,
#             norm_cfg=dict(type='GN', num_groups=num_groups),
#             numC_input=64,
#             num_layer=[2],
#             stride=[1],
#             num_channels=[64 * 4]
#         ),
#         dict(
#             type='CustomResNet',
#             with_cp=False,
#             norm_cfg=dict(type='GN', num_groups=num_groups),
#             numC_input=64,
#             num_layer=[2],
#             stride=[1],
#             num_channels=[64 * 8]
#         )
#     ],
#     pts_neck=dict(
#         type='KPPillarsBEVFPN',
#         in_channels=[64, 64 * 2, 64 * 4, 64 * 8],
#         out_channels=64,
#         input_feature_index=(0, 1, 2, 3),
#         scale_factors=[1, 2, 4, 8],
#         resnet_cfg=[
#             dict(
#                 type='CustomResNet',
#                 with_cp=False,
#                 norm_cfg=dict(type='GN', num_groups=num_groups),
#                 numC_input=64,
#                 num_layer=[2],
#                 stride=[1],
#                 num_channels=[64]
#             ),
#             dict(
#                 type='CustomResNet',
#                 with_cp=False,
#                 norm_cfg=dict(type='GN', num_groups=num_groups),
#                 numC_input=64,
#                 num_layer=[1],
#                 stride=[1],
#                 num_channels=[64]
#             ),
#             dict(
#                 type='CustomResNet',
#                 with_cp=False,
#                 norm_cfg=dict(type='GN', num_groups=num_groups),
#                 numC_input=64,
#                 num_layer=[1],
#                 stride=[1],
#                 num_channels=[64]
#             ),
#             dict(
#                 type='CustomResNet',
#                 with_cp=False,
#                 norm_cfg=dict(type='GN', num_groups=num_groups),
#                 numC_input=64,
#                 num_layer=[1],
#                 stride=[1],
#                 num_channels=[64]
#             )
#         ],
#         norm_cfg=dict(type='GN', num_groups=num_groups),
#     ),
#     pts_bbox_head=dict(
#         type='CenterHead',
#         in_channels=64,
#         tasks=[
#             dict(num_class=1, class_names=['car']),
#         ],
#         common_heads=dict(
#             reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
#         share_conv_channel=64,
#         bbox_coder=dict(
#             type='CenterPointBBoxCoder',
#             pc_range=point_cloud_range[:2],
#             post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#             max_num=500,
#             score_threshold=0.1,
#             out_size_factor=4,
#             voxel_size=voxel_size_head[:2],
#             code_size=9),
#         separate_head=dict(
#             type='SeparateHead', init_bias=-2.19, final_kernel=3, norm_cfg=dict(type='GN', num_groups=num_groups)),
#         loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
#         loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
#         norm_bbox=True,
#         norm_cfg=dict(type='GN', num_groups=num_groups)
#     ),
#     train_cfg=dict(
#         pts=dict(
#             point_cloud_range=point_cloud_range,
#             grid_size=[1024, 1024, 40],
#             voxel_size=voxel_size_head,
#             out_size_factor=4,
#             dense_reg=1,
#             gaussian_overlap=0.1,
#             max_objs=500,
#             min_radius=2,
#             code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#     ),
#     test_cfg=dict(
#         pts=dict(
#             pc_range=point_cloud_range[:2],
#             post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#             max_per_img=500,
#             max_pool_nms=False,
#             min_radius=[4, 12, 10, 1, 0.85, 0.175],
#             score_threshold=0.1,
#             out_size_factor=4,
#             voxel_size=voxel_size_head[:2],
#             pre_max_size=1000,
#             post_max_size=83,
#             # Scale-NMS
#             nms_thr=0.125,
#             nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
#             nms_rescale_factor=[0.7, [0.4, 0.6], [0.3, 0.4], 0.9, [1.0, 1.0], [1.5, 2.5]],
#         )
#     ),
#     init_cfg=None
# )

train_pipeline = [
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
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5
    ),
    # dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
    ),
    # dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix'
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

# train_pipeline = [
#     dict(
#         type='LoadRadarPointsMultiSweeps',
#         load_dim=load_dim,
#         sweeps_num=sweeps_num,
#         remove_close=None,
#         use_dim=radar_use_dim,
#         test_mode=False,
#         file_client_args=dict(backend='disk')
#     ),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-0.3925, 0.3925],
#         scale_ratio_range=[0.95, 1.05],
#         translation_std=[0, 0, 0]),
#     dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5, sync_2d=False),
#     dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectNameFilter', classes=class_names),
#     dict(type='PointShuffle'),
#     dict(type='DefaultFormatBundle3D', class_names=class_names),
#     dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]

# test_pipeline = [
#     dict(
#         type='LoadRadarPointsMultiSweeps',
#         load_dim=load_dim,
#         sweeps_num=sweeps_num,
#         remove_close=None,
#         use_dim=radar_use_dim,
#         test_mode=True,
#         file_client_args=dict(backend='disk')
#     ),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D'),
#             dict(type='RadarPointsRangeFilter', point_cloud_range=point_cloud_range),
#             dict(type='DefaultFormatBundle3D',
#                  class_names=class_names,
#                  with_label=False),
#             dict(type='Collect3D', keys=['points'])
#         ])
# ]

# eval_pipeline = [
#     dict(
#         type='LoadRadarPointsMultiSweeps',
#         load_dim=load_dim,
#         sweeps_num=sweeps_num,
#         remove_close=None,
#         use_dim=radar_use_dim,
#         test_mode=False,
#         file_client_args=dict(backend='disk')
#     ),
#     dict(type='DefaultFormatBundle3D',class_names=class_names, with_label=False),
#     dict(type='Collect3D', keys=['points'])
# ]
# evaluation = dict(interval=1, pipeline=eval_pipeline)

# input_modality = dict(
#     use_lidar=True,
#     use_camera=False,
#     use_radar=True,
#     use_map=False,
#     use_external=False
# )
input_modality = dict(
    use_lidar=True,
    use_camera=False,
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