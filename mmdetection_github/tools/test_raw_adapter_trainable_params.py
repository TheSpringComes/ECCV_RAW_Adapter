#!/usr/bin/env python3
"""统计 RAW_ResNet 全骨干的可训练参数量，并分解为 RAW_Adapter vs ResNet 主体；可选对比 light_mode。"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# 从仓库根目录或 mmdetection_github 目录运行均可
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mmengine.registry import init_default_scope  # noqa: E402

from mmdet.models.backbones.RAW_resnet import RAW_ResNet  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402


def _normalize_light_mode(light_mode):
    """与配置文件一致：可为 'normal'/'low' 或 dict(type='...')."""
    if isinstance(light_mode, dict):
        return light_mode.get('type', 'normal')
    return light_mode


def count_params(module, trainable_only: bool) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def summarize_adapter(backbone: RAW_ResNet) -> dict:
    parts = {
        'pre_encoder (Input-level)': backbone.pre_encoder,
        'model_adapter (Model-level)': backbone.model_adapter,
        'merge_1': backbone.merge_1,
        'merge_2': backbone.merge_2,
        'merge_3': backbone.merge_3,
    }
    rows = {}
    train_total = 0
    all_total = 0
    for name, m in parts.items():
        t = count_params(m, trainable_only=True)
        a = sum(p.numel() for p in m.parameters())
        rows[name] = {'trainable': t, 'all': a}
        train_total += t
        all_total += a
    rows['_raw_adapter_trainable_total'] = train_total
    rows['_raw_adapter_all_total'] = all_total
    return rows


def summarize_resnet_stem_and_layers(backbone: RAW_ResNet) -> dict:
    """ResNet 部分：stem（conv1+norm1）与 layer1–layer4，不含 RAW_Adapter。"""
    stem = backbone.conv1
    norm1 = getattr(backbone, backbone.norm1_name)
    stem_train = count_params(stem, True) + count_params(norm1, True)
    layers_train = []
    for i in range(1, 5):
        layers_train.append(count_params(getattr(backbone, f'layer{i}'), True))
    return {
        'stem (conv1+norm1)': stem_train,
        'layer1': layers_train[0],
        'layer2': layers_train[1],
        'layer3': layers_train[2],
        'layer4': layers_train[3],
        'resnet_subtotal': stem_train + sum(layers_train),
    }


def summarize_whole_backbone(backbone: RAW_ResNet) -> dict:
    """整段 RAW_ResNet：总可训练参数 + 与 RAW_Adapter / ResNet 分解（三者应对齐）。"""
    adapter = summarize_adapter(backbone)
    ada_train = adapter['_raw_adapter_trainable_total']
    resnet_parts = summarize_resnet_stem_and_layers(backbone)
    resnet_train = resnet_parts['resnet_subtotal']
    total_train = count_params(backbone, True)
    ok = ada_train + resnet_train == total_train
    return {
        'whole_backbone_trainable': total_train,
        'whole_backbone_all_params': sum(p.numel() for p in backbone.parameters()),
        'raw_adapter_trainable': ada_train,
        'resnet_stem_layers_trainable': resnet_train,
        'decomposition_ok': ok,
        'resnet_breakdown': resnet_parts,
    }


def print_report(title: str, rows: dict) -> None:
    print(f'\n=== {title} ===')
    for k, v in rows.items():
        if k.startswith('_'):
            continue
        print(f'  {k}: trainable={v["trainable"]:,}  all={v["all"]:,}')
    print(
        f'  RAW_Adapter 合计: trainable={rows["_raw_adapter_trainable_total"]:,}  '
        f'all={rows["_raw_adapter_all_total"]:,}'
    )


def build_retinanet_raw_res50(light_mode: str = 'normal', num_classes: int = 3):
    """与 configs/PASCALRAW_Res50/Over_Exp_raw_adapter_res50.py 中 backbone+neck+head 一致（init_cfg=None 不落盘权重）。"""
    init_default_scope('mmdet')
    lm = _normalize_light_mode(light_mode)
    return MODELS.build(
        dict(
            type='RetinaNet',
            data_preprocessor=dict(
                type='DetDataPreprocessor',
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
                bgr_to_rgb=True,
                pad_size_divisor=32),
            backbone=dict(
                type='RAW_ResNet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                light_mode=lm,
                w_lut=True,
                lut_dim=32,
                k_size=3,
                fea_c_s=[256, 512, 1024],
                ada_c_s=[24, 48, 96],
                mid_c_s=[64, 64, 128],
                merge_ratio=1.0,
                frozen_stages=-1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                style='pytorch',
                init_cfg=None),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_input',
                num_outs=5),
            bbox_head=dict(
                type='RetinaHead',
                num_classes=num_classes,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))))


def print_full_detector_report(light_mode: str = 'normal') -> None:
    """RetinaNet 整模型：backbone + FPN + RetinaHead（data_preprocessor 无可训练参数）。"""
    det = build_retinanet_raw_res50(light_mode=light_mode)
    total = count_params(det, True)
    parts = [
        ('backbone (RAW_ResNet)', det.backbone),
        ('neck (FPN)', det.neck),
        ('bbox_head (RetinaHead)', det.bbox_head),
    ]
    print(f'\n=== RetinaNet 整模型 light_mode={light_mode!r} ===')
    print(f'  可训练参数总计: {total:,}')
    sub = 0
    for name, m in parts:
        t = count_params(m, True)
        sub += t
        print(f'  {name}: {t:,}')
    assert sub == total, f'子模块之和 {sub} != 总计 {total}'


def print_whole_backbone_report(bb: RAW_ResNet) -> None:
    w = summarize_whole_backbone(bb)
    rb = w['resnet_breakdown']
    print('  --- 整段 RAW_ResNet 骨干 ---')
    print(f'  可训练参数总计: {w["whole_backbone_trainable"]:,}')
    print(f'  全部参数（含冻结）: {w["whole_backbone_all_params"]:,}')
    print(
        f'  分解: RAW_Adapter={w["raw_adapter_trainable"]:,} + '
        f'ResNet(stem+layer1–4)={w["resnet_stem_layers_trainable"]:,} '
        f'（校验一致: {w["decomposition_ok"]}）'
    )
    print(
        f'  ResNet 细分: stem={rb["stem (conv1+norm1)"]:,}, '
        f'layer1={rb["layer1"]:,}, layer2={rb["layer2"]:,}, '
        f'layer3={rb["layer3"]:,}, layer4={rb["layer4"]:,}'
    )


def kernel_gain_info(backbone: RAW_ResNet) -> str:
    pk = backbone.pre_encoder.Predictor_K
    gb = pk.gain_base
    return (
        f'Kernel_Predictor.gain_base: requires_grad={gb.requires_grad}, '
        f'numel={gb.numel()}, value={gb.detach().cpu().tolist()}'
    )


def dump_model_structure(module: torch.nn.Module, title: str) -> None:
    """打印 nn.Module 的层级结构（与 print(model) 相同）。"""
    print(f'\n=== {title} — module tree ===')
    print(module)


def dump_named_parameters(module: torch.nn.Module, title: str) -> None:
    """逐行打印 name / shape / numel / requires_grad。"""
    print(f'\n=== {title} — named parameters ===')
    hdr = f'{"name":<88} {"shape":<28} {"numel":>14} {"train":>6}'
    print(hdr)
    print('-' * len(hdr))
    total = 0
    trainable = 0
    for name, p in module.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        print(
            f'{name:<88} {str(tuple(p.shape)):<28} {p.numel():>14,} '
            f'{str(p.requires_grad):>6}'
        )
    print('-' * len(hdr))
    print(
        f'{"TOTAL":<88} {"":<28} {total:>14,}  '
        f'(trainable: {trainable:,})'
    )


def run_forward(backbone: RAW_ResNet, device: str) -> None:
    backbone = backbone.to(device)
    backbone.train()
    x = torch.rand(1, 3, 128, 128, device=device)
    with torch.no_grad():
        outs = backbone(x)
    assert isinstance(outs, tuple) and len(outs) == 4
    print(f'  forward OK: 输出 stage 特征形状 = {[tuple(t.shape) for t in outs]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help='cpu 或 cuda')
    parser.add_argument(
        '--no-retina',
        action='store_true',
        help='不统计 RetinaNet 整模型，仅 RAW_ResNet 骨干与前向/反向检查',
    )
    parser.add_argument(
        '--print-model',
        action='store_true',
        help='打印模型结构（module 树）',
    )
    parser.add_argument(
        '--print-params',
        action='store_true',
        help='打印每个可学习参数的名称、shape、元素数、是否训练',
    )
    parser.add_argument(
        '--print-all',
        action='store_true',
        help='等价于同时指定 --print-model 与 --print-params',
    )
    args = parser.parse_args()
    args.retina = not args.no_retina
    if args.print_all:
        args.print_model = True
        args.print_params = True
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA 不可用，改用 CPU')
        device = 'cpu'

    common = dict(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        w_lut=True,
        lut_dim=32,
        k_size=3,
        fea_c_s=[256, 512, 1024],
        ada_c_s=[24, 48, 96],
        mid_c_s=[64, 64, 128],
        merge_ratio=1.0,
        frozen_stages=-1,
        norm_eval=True,
        init_cfg=None,
    )

    for lm in ('normal', 'low'):
        lm_resolved = _normalize_light_mode(lm)
        bb = RAW_ResNet(light_mode=lm_resolved, **common)
        rows = summarize_adapter(bb)
        print_report(f'light_mode={lm!r} (resolved={lm_resolved!r})', rows)
        print_whole_backbone_report(bb)
        print(f'  {kernel_gain_info(bb)}')

    # 与配置文件相同的 dict 写法
    lm_cfg = _normalize_light_mode(dict(type='normal'))
    bb_cfg = RAW_ResNet(light_mode=dict(type='normal'), **common)
    print(f'\n  dict(type=\'normal\') 解析为 mode={lm_cfg!r}；'
          f'gain_base.requires_grad={bb_cfg.pre_encoder.Predictor_K.gain_base.requires_grad}')

    if args.retina:
        print_full_detector_report('normal')
        print_full_detector_report('low')

    if args.print_model or args.print_params:
        if args.retina:
            m = build_retinanet_raw_res50('normal')
            label = 'RetinaNet (RAW_ResNet + FPN + RetinaHead)'
        else:
            m = RAW_ResNet(light_mode='normal', **common)
            label = 'RAW_ResNet backbone'
        if args.print_model:
            dump_model_structure(m, label)
        if args.print_params:
            dump_named_parameters(m, label)

    print('\n=== 一次前向（验证计算图）===')
    run_forward(RAW_ResNet(light_mode='normal', **common), device)

    # 训练模式下梯度是否流到 RAW_Adapter
    bb = RAW_ResNet(light_mode='normal', **common).to(device)
    bb.train()
    x = torch.rand(2, 3, 64, 64, device=device, requires_grad=False)
    y = bb(x)[0].sum()
    y.backward()
    gpe = bb.pre_encoder.Predictor_K.q.grad
    assert gpe is not None and gpe.abs().sum() > 0, 'pre_encoder 应有梯度'
    print('  backward OK: pre_encoder.Predictor_K.q.grad 非零')


if __name__ == '__main__':
    main()
