# -*- coding: utf-8 -*-
"""仿真域（mm 缓存）上的预测可视化：RGB | IR | LiDAR 深度 三个面板，GT 绿框 / 预测红框。

用于看教师（6 通道）和学生（RGB）在同一帧上的表现。学生模型即使配置里是 CenterDetKD，
这里也只跑它自己（蒸馏权重全部置 0，不去构建教师）。

用法：
    python tools/vis_mm_pred.py --cfg cfgs/models/uavdet_3d/mmcache/teacher.yaml \
        --ckpt <teacher>.pth --num 8 --out ../output/vis_pred_teacher
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

from uavdet3d.config import cfg_from_yaml_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.model import build_network, load_data_to_gpu
from uavdet3d.utils import common_utils, frame_convention

GREEN, RED, MAGENTA, WHITE = (60, 220, 60), (40, 40, 255), (255, 0, 255), (255, 255, 255)


def box_corners(b):
    c = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
    pts = c * np.asarray(b[3:6])
    return pts @ R.from_euler(frame_convention.get_default_euler_seq(), b[6:9]).as_matrix().T + np.asarray(b[0:3])


def draw9d(img, b, K, color, th=2, text=None):
    pts = box_corners(b)
    if np.any(pts[:, 2] <= 1e-3):
        return
    uv = (K @ pts.T).T
    uv = (uv[:, :2] / uv[:, 2:3])
    if np.any(np.abs(uv) > 20 * max(img.shape[:2])):
        return
    uv = np.round(uv).astype(int)
    for a, b2 in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(uv[a]), tuple(uv[b2]), color, th, cv2.LINE_AA)
    if text:
        x, y = int(uv[:, 0].min()), max(12, int(uv[:, 1].min()) - 4)
        cv2.putText(img, text, (max(0, min(x, img.shape[1] - 160)), y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def header(panel, lines):
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-path', default=None)
    ap.add_argument('--num', type=int, default=8)
    ap.add_argument('--score', type=float, default=0.3)
    ap.add_argument('--out', default='../output/vis_mm_pred')
    ap.add_argument('--stride-pick', type=int, default=97, help='每隔多少帧取一张，避免全是同一序列')
    ap.add_argument('--tag', default=None, help='写在图上的名字')
    ap.add_argument('--no-norm', action='store_true', help='评 2026-09-06 归一化改动之前训出的权重：去掉 NORM_MEAN/STD')
    args = ap.parse_args()

    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(args.cfg, cfg)
    if args.data_path:
        cfg.DATA_CONFIG.DATA_PATH = args.data_path
    if args.no_norm:
        cfg.DATA_CONFIG.pop('NORM_MEAN', None)
        cfg.DATA_CONFIG.pop('NORM_STD', None)
    # 只画学生自己：不构建教师
    if 'DISTILL' in cfg.MODEL:
        for k in list(cfg.MODEL.DISTILL.keys()):
            if k.startswith('W_'):
                cfg.MODEL.DISTILL[k] = 0.0
    # 数据集总是吐全部四个模态供画图；模型按自己的 INPUT_CHANNELS 切
    cfg.DATA_CONFIG.MODALITIES = ['rgb', 'ir', 'depth', 'tag']
    in_ch = int(cfg.MODEL.BACKBONE_2D.INPUT_CHANNELS)

    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False,
                                     workers=0, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()
    tag = args.tag or os.path.splitext(os.path.basename(args.cfg))[0]

    os.makedirs(args.out, exist_ok=True)
    n, i = 0, 0
    sx, sy = ds.W / 1280.0, ds.H / 720.0
    with torch.no_grad():
        for batch in loader:
            i += 1
            if (i - 1) % args.stride_pick:
                continue
            if n >= args.num:
                break
            full = batch['image'].copy()                      # (1,1,6,H,W) numpy
            batch['image'] = full[:, :, :in_ch]
            load_data_to_gpu(batch)
            out = model(batch)
            pred = np.asarray(out['pred_boxes9d'][0])
            conf = np.asarray(out['confidence'][0])
            gt = batch['gt_box9d'][0]
            gt = gt.cpu().numpy() if hasattr(gt, 'cpu') else np.asarray(gt)
            gt = gt[np.abs(gt).sum(1) > 0]
            K = np.asarray(batch['intrinsic'][0])[0].copy()
            K[0] *= sx
            K[1] *= sy

            img = full[0, 0].copy()
            # 数据集若按域归一化了 rgb（NORM_MEAN/STD），显示前要反归一化回 [0,1]
            nm, ns = cfg.DATA_CONFIG.get('NORM_MEAN', None), cfg.DATA_CONFIG.get('NORM_STD', None)
            if nm and ns:
                img[:3] = img[:3] * np.asarray(ns, np.float32)[:, None, None] + np.asarray(nm, np.float32)[:, None, None]
            rgb = np.ascontiguousarray((np.clip(img[:3], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))   # 缓存里就是 BGR
            ir = cv2.cvtColor((img[3] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            dep = img[4]
            depv = cv2.applyColorMap((255 - np.clip(dep, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            depv[dep <= 0] = (25, 25, 25)
            depv[img[5] > 0] = MAGENTA

            errs = []
            for panel in (rgb, ir, depv):
                for g in gt:
                    draw9d(panel, g, K, GREEN, 2)
                for p, c in zip(pred, conf):
                    if c < args.score:
                        continue
                    d = np.linalg.norm(gt[:, :3] - p[:3], axis=1) if len(gt) else np.array([np.nan])
                    draw9d(panel, p, K, RED, 1, '%.2f  z%.1f' % (c, p[2]))
                    if panel is rgb and len(gt):
                        errs.append(float(d.min()))
            seq = batch['seq_id'][0]
            header(rgb, ['%s  %s' % (tag, seq.split('/')[0] + '/' + seq.split('/')[3]),
                         'GT %d  pred %d  pos err med %.2fm' % (len(gt), sum(conf >= args.score), np.median(errs) if errs else -1)])
            header(ir, ['IR (shifted to RGB view at nearest target depth)'])
            header(depv, ['LiDAR depth (turbo)  magenta = drone-tagged  noise %.1fm' % ds.metas[int(ds.valid_idx[i - 1])]['lidar_noise']])
            sep = np.full((rgb.shape[0], 3, 3), 90, np.uint8)
            comp = np.concatenate([rgb, sep, ir, sep, depv], axis=1)
            op = os.path.join(args.out, '%s_%02d_%s.jpg' % (tag, n, seq.replace('/', '_')[:50]))
            cv2.imwrite(op, comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print('写出', op)
            n += 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
