# -*- coding: utf-8 -*-
"""在 camnorm 缓存帧上画 3D 框和机体坐标轴（x 红 / y 绿 / z 蓝），人工核对两域的机体轴语义是否一致
（z 应朝上、x 应指向机头 / 云台相机一侧）。可选 --aug：走数据集的训练增广（翻转/缩放）后再画，核对增广后标签。

    python vis_camnorm_axes.py --cache E:/mmcache/mav6d_cn --split test --n 24 --out ../output/camnorm/vis/axes_mav6d.jpg
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def draw(img, K, boxes, seq='xyz', scale=2):
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
    Ks = np.asarray(K, dtype=np.float64).copy()
    Ks[:2] *= scale
    Ks[0, 2] = (K[0, 2] + 0.5) * scale - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * scale - 0.5
    for box in boxes:
        _draw_one(img, Ks, np.asarray(box, dtype=np.float64), seq)
    return img


def _draw_one(img, Ks, box, seq):
    Rm = R.from_euler(seq, box[6:9]).as_matrix()
    c = (PROTO8 * box[3:6]) @ Rm.T + box[:3]
    if c[:, 2].min() <= 0.05:
        return

    def p(x):
        u = Ks @ x
        return int(round(u[0] / u[2])), int(round(u[1] / u[2]))
    for a, b in EDGES:
        cv2.line(img, p(c[a]), p(c[b]), (0, 255, 255), 1, cv2.LINE_AA)
    L = 0.75 * float(max(box[3:6]))
    o = box[:3]
    for k, col in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))):
        cv2.arrowedLine(img, p(o), p(o + Rm[:, k] * L), col, 2, cv2.LINE_AA, tipLength=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--crop', type=int, default=0, help='>0 时以目标为中心裁这么大（缓存像素）再放大，看清机身')
    ap.add_argument('--aug', action='store_true', help='用数据集训练增广后的样本（需要 --cfg）')
    ap.add_argument('--cfg', default='cfgs/dataset_configs/uavdet_3d/camnorm_mav6d.yaml')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    idx = pickle.load(open(os.path.join(args.cache, args.split, 'index.pkl'), 'rb'))
    rgb = np.load(os.path.join(args.cache, args.split, 'rgb.npy'), mmap_mode='r')
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(idx['valid_idx'], size=min(args.n, len(idx['valid_idx'])), replace=False)
    ds = None
    if args.aug:
        from easydict import EasyDict
        from uavdet3d.config import cfg_from_yaml_file
        from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset
        cfg = EasyDict()
        cfg_from_yaml_file(args.cfg, cfg)
        cfg.DATA_PATH = args.cache
        cfg.DATA_SPLIT['train'] = args.split
        cfg.MIN_RGB_VIS = 0.0
        cfg.pop('NORM_MEAN', None)
        cfg.pop('NORM_STD', None)
        ds = MMCache_Det_Dataset(cfg, training=True)
        np.random.seed(args.seed)
    tiles = []
    for i in picks:
        m = idx['metas'][int(i)]
        if ds is not None:
            img = (np.ascontiguousarray(rgb[int(i)]).astype(np.float32).transpose(2, 0, 1) / 255.0)
            img, boxes, _, K = ds._augment(img, m['boxes9d'].astype(np.float64).copy(), list(m['names']),
                                           ds.cache_intrinsic(m))
            img = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        else:
            img = np.ascontiguousarray(rgb[int(i)])
            boxes, K = m['boxes9d'], np.asarray(m['K_in'], dtype=np.float64)
        img = np.ascontiguousarray(img)
        img = draw(img, K, boxes)
        if args.crop and len(boxes):
            b = np.asarray(boxes[0], np.float64)
            u = K @ b[:3]
            cu, cv_ = int(u[0] / u[2] * 2), int(u[1] / u[2] * 2)
            h = args.crop
            pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT)
            img = cv2.resize(pad[cv_:cv_ + 2 * h, cu:cu + 2 * h], (2 * h * 2, 2 * h * 2))
        txt = '%s %s' % (m['seq'].split('/')[-1][:24], m.get('cls', ''))
        cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    th, tw = tiles[0].shape[:2]
    rows = []
    for r in range(0, len(tiles), args.cols):
        row = [cv2.resize(t, (tw, th)) for t in tiles[r:r + args.cols]]
        row += [np.zeros_like(tiles[0])] * (args.cols - len(row))
        rows.append(np.concatenate(row, 1))
    sheet = np.concatenate(rows, 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('写出', args.out, sheet.shape)


if __name__ == '__main__':
    main()
