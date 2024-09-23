# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms.processing import TestTimeAug
from mmengine.dataset.sampler import DefaultSampler
from mmengine.visualization.vis_backend import LocalVisBackend

from mmdet3d.datasets.semantickitti_dataset import SemanticKittiDataset
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import (LoadAnnotations3D,
                                                 LoadPointsFromFile,
                                                 PointSegClassMapping)
from mmdet3d.datasets.transforms.transforms_3d import (GlobalRotScaleTrans,
                                                       RandomFlip3D)
from mmdet3d.evaluation.metrics.seg_metric import SegMetric
from mmdet3d.evaluation.metrics.save_seg import SemantickInferMertric
from mmdet3d.models.segmentors.seg3d_tta import Seg3DTTAModel
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer

# For SemanticKitti we usually do 19-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'SemanticKittiDataset'
data_root = '/home/zjx/semantickitti/'
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = {
    0: 19,  # "unlabeled"
    1: 19,  # "outlier" mapped to "unlabeled" --------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: 19,  # "other-structure" mapped to "unlabeled" ------mapped
    60: 8,  # "lane-marking" to "road" ---------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: 19,  # "other-object" to "unlabeled" ----------------mapped
    252: 0,  # "moving-car" to "car" ------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------mapped
    254: 5,  # "moving-person" to "person" ------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehic------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------mapped
    258: 3,  # "moving-truck" to "truck" --------------------mapped
    259: 4  # "moving-other"-vehicle to "other-vehicle"-----mapped
}
# labels_map = {
#     0 : 0,     # "unlabeled"
#     1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
#     10: 1,     # "car"
#     11: 2,     # "bicycle"
#     13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
#     15: 3,     # "motorcycle"
#     16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
#     18: 4,     # "truck"
#     20: 5,     # "other-vehicle"
#     30: 6,     # "person"
#     31: 7,     # "bicyclist"
#     32: 8,     # "motorcyclist"
#     40: 9,     # "road"
#     44: 10,    # "parking"
#     48: 11,    # "sidewalk"
#     49: 12,    # "other-ground"
#     50: 13,    # "building"
#     51: 14,    # "fence"
#     52: 0,    # "other-structure" mapped to "unlabeled" ------------------mapped
#     60: 9,     # "lane-marking" to "road" ---------------------------------mapped
#     70: 15,    # "vegetation"
#     71: 16,    # "trunk"
#     72: 17,    # "terrain"
#     80: 18,    # "pole"
#     81: 19,    # "traffic-sign"
#     99: 0,    # "other-object" to "unlabeled" ----------------------------mapped
#     252: 1,    # "moving-car" to "car" ------------------------------------mapped
#     253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
#     254: 6,    # "moving-person" to "person" ------------------------------mapped
#     255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
#     256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
#     257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
#     258: 4,    # "moving-truck" to "truck" --------------------------------mapped
#     259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
# }

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=259)

input_modality = dict(use_lidar=True, use_camera=False)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/semantickitti/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type=PointSegClassMapping),
    dict(
        type=RandomFlip3D,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type=Pack3DDetInputs, keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type=PointSegClassMapping),
    dict(type=Pack3DDetInputs, keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type=Pack3DDetInputs, keys=['points'])
]
tta_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type=PointSegClassMapping),
    dict(
        type=TestTimeAug,
        transforms=[[
            dict(
                type=RandomFlip3D,
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type=RandomFlip3D,
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type=RandomFlip3D,
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type=RandomFlip3D,
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type=GlobalRotScaleTrans,
                            rot_range=[pcd_rotate_range, pcd_rotate_range],
                            scale_ratio_range=[
                                pcd_scale_factor, pcd_scale_factor
                            ],
                            translation_std=[0, 0, 0])
                        for pcd_rotate_range in [-0.78539816, 0.0, 0.78539816]
                        for pcd_scale_factor in [0.95, 1.0, 1.05]
                    ], [dict(type=Pack3DDetInputs, keys=['points'])]])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=SemanticKittiDataset,
        data_root=data_root,
        ann_file='semantickitti_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=SemanticKittiDataset,
        data_root=data_root,
        ann_file='semantickitti_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        test_mode=True,
        backend_args=backend_args))

val_dataloader = test_dataloader

val_evaluator = dict(type=SegMetric)
test_evaluator = val_evaluator

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=Det3DLocalVisualizer, vis_backends=vis_backends, name='visualizer')

tta_model = dict(type=Seg3DTTAModel)