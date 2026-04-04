_base_ = ['./COCO_R_Net_raw_adapter.py']

# Keep the same COCO json dataloaders/paths as COCO_R_Net_raw_adapter.py,
# only switch detector stack from RetinaNet to YOLOV3.
# Channel check:
# - RAW_ResNet stage outputs for out_indices=(1,2,3): [512, 1024, 2048]
# - YOLOV3Neck expects features from high->low channels as [2048, 1024, 512]
# - YOLOV3Head in_channels matches neck out_channels [1024, 512, 256]
model = dict(
    _delete_=True,
    type='YOLOV3',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.00, 255.00, 255.00],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='RAW_ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        lut_dim=32,
        k_size=9,
        fea_c_s=[256, 512, 1024],
        ada_c_s=[24, 48, 96],
        mid_c_s=[64, 64, 128],
        w_lut=True,
        merge_ratio=1,
        light_mode=dict(type='normal'),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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

# Keep optimizer defaults unchanged from the base config.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
