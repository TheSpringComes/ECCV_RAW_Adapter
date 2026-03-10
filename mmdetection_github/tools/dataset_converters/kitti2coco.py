#!/usr/bin/env python3
"""Convert KITTI 2D detection labels to COCO json.

Expected KITTI layout (default):
- data_root/
  - training/
    - image_2/
    - label_2/
  - ImageSets/
    - train.txt
    - val.txt
    - test.txt (optional)

Each split txt contains image ids (without extension), one per line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

DEFAULT_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    'Misc'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='KITTI label to COCO converter')
    parser.add_argument('--data-root', required=True, help='KITTI root path')
    parser.add_argument(
        '--split-file', required=True, help='Split file path, e.g. ImageSets/train.txt')
    parser.add_argument(
        '--image-dir', default='training/image_2', help='Image folder relative to data-root')
    parser.add_argument(
        '--label-dir', default='training/label_2', help='Label folder relative to data-root')
    parser.add_argument('--out-file', required=True, help='Output COCO json file path')
    parser.add_argument(
        '--classes',
        nargs='+',
        default=DEFAULT_CLASSES,
        help='Class list used to keep/encode annotations')
    return parser.parse_args()


def load_split_ids(split_file: Path) -> List[str]:
    ids = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if line:
            ids.append(line)
    return ids


def read_kitti_label(label_file: Path) -> List[Tuple[str, float, float, float, float]]:
    annos = []
    if not label_file.exists():
        return annos
    for line in label_file.read_text().splitlines():
        fields = line.strip().split()
        if len(fields) < 8:
            continue
        cls_name = fields[0]
        if cls_name == 'DontCare':
            continue
        x1, y1, x2, y2 = map(float, fields[4:8])
        annos.append((cls_name, x1, y1, x2, y2))
    return annos


def convert(data_root: Path, split_file: Path, image_dir: str, label_dir: str,
            out_file: Path, classes: List[str]) -> None:
    cls2id: Dict[str, int] = {c: i + 1 for i, c in enumerate(classes)}
    image_root = data_root / image_dir
    label_root = data_root / label_dir

    image_ids = load_split_ids(split_file)
    coco = {
        'images': [],
        'annotations': [],
        'categories': [{
            'id': cid,
            'name': cname
        } for cname, cid in cls2id.items()]
    }

    ann_id = 1
    for img_id_num, img_stem in enumerate(image_ids, start=1):
        image_file = image_root / f'{img_stem}.png'
        if not image_file.exists():
            alt = image_root / f'{img_stem}.jpg'
            if alt.exists():
                image_file = alt
            else:
                raise FileNotFoundError(f'Image not found: {img_stem}(.png/.jpg) in {image_root}')

        width, height = Image.open(image_file).size
        coco['images'].append({
            'id': img_id_num,
            'file_name': image_file.name,
            'width': width,
            'height': height,
        })

        label_file = label_root / f'{img_stem}.txt'
        for cls_name, x1, y1, x2, y2 in read_kitti_label(label_file):
            if cls_name not in cls2id:
                continue
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id_num,
                'category_id': cls2id[cls_name],
                'bbox': [x1, y1, w, h],
                'area': w * h,
                'iscrowd': 0,
            })
            ann_id += 1

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(coco))
    print(f'Wrote {out_file} | images={len(coco["images"])} anns={len(coco["annotations"])}')


if __name__ == '__main__':
    args = parse_args()
    convert(
        data_root=Path(args.data_root),
        split_file=Path(args.split_file),
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        out_file=Path(args.out_file),
        classes=args.classes,
    )
