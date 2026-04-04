_base_ = ['./Low_Light_raw_adapter_res18.py']

model = dict(
    type='YOLOV3',
    backbone=dict(out_indices=(1, 2, 3)),
    neck=dict(
        _delete_=True,
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[512, 256, 128],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        _delete_=True,
        type='YOLOV3Head',
        num_classes=3,
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
