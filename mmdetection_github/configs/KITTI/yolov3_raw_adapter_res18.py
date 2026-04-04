_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

kitti_classes = (
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    'Misc')

model = dict(
    type='YOLOV3',
    data_preprocessor={'type': 'DetDataPreprocessor', 'mean': [0.0, 0.0, 0.0], 'std': [255.0, 255.0, 255.0], 'bgr_to_rgb': True, 'pad_size_divisor': 32},
    backbone={'type': 'RAW_ResNet', 'depth': 18, 'num_stages': 4, 'out_indices': (1, 2, 3), 'lut_dim': 32, 'k_size': 9, 'fea_c_s': [256, 512, 1024], 'ada_c_s': [24, 48, 96], 'mid_c_s': [64, 64, 128], 'w_lut': True, 'merge_ratio': 1, 'light_mode': {'type': 'normal'}, 'frozen_stages': -1, 'norm_cfg': {'type': 'BN', 'requires_grad': True}, 'norm_eval': True, 'style': 'pytorch', 'init_cfg': {'type': 'Pretrained', 'checkpoint': 'torchvision://resnet18'}},
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[512, 256, 128],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=8,
        in_channels=[512, 256, 128],
        out_channels=[512, 256, 128],
        featmap_strides=[32, 16, 8]),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))

dataset_type = 'CocoDataset'
data_root = './data/KITTI/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1242, 375), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1242, 375), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

metainfo = dict(classes=kitti_classes)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/kitti_train.json',
        data_prefix=dict(img='training/image_2/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/kitti_val.json',
        data_prefix=dict(img='training/image_2/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/kitti_val.json', metric='bbox')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

max_epochs = 12
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs)

auto_scale_lr = dict(base_batch_size=16)
