_base_ = [
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py',
]

# Load COCO Pretrain Model
#load_from = r'https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'

# './centernet_tta.py'

# model settings
model = dict(
    type='YOLOV3',
    data_preprocessor={'type': 'DetDataPreprocessor', 'mean': [0.0, 0.0, 0.0], 'std': [255.0, 255.0, 255.0], 'bgr_to_rgb': True, 'pad_size_divisor': 32},
    backbone={'type': 'RAW_ResNet', 'depth': 50, 'num_stages': 4, 'out_indices': (1, 2, 3), 'lut_dim': 32, 'k_size': 9, 'fea_c_s': [256, 512, 1024], 'ada_c_s': [24, 48, 96], 'mid_c_s': [64, 64, 128], 'w_lut': True, 'merge_ratio': 1, 'light_mode': {'type': 'normal'}, 'frozen_stages': -1, 'norm_cfg': {'type': 'BN', 'requires_grad': True}, 'norm_eval': True, 'style': 'pytorch', 'init_cfg': {'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'}},
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[2048, 1024, 512],
        out_channels=[1024, 512, 256]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=80,
        in_channels=[1024, 512, 256],
        out_channels=[1024, 512, 256],
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
data_root = '/home/jing/datasets/COCO/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(667, 400), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(667, 400), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_trainval2017/annotations/instances_train2017.json',
        data_prefix=dict(img='train2017_SynRAW/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        ann_file='annotations_trainval2017/annotations/instances_val2017.json',
        data_prefix=dict(img='val2017_SynRAW/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations_trainval2017/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
# optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

max_epochs = 30  # the real epoch is 7*5 = 35
# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=500),
]
train_cfg = dict(max_epochs=max_epochs)  # the real epoch is 5*2 = 10


auto_scale_lr = dict(base_batch_size=4)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=500),
)
