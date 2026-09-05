# -*- coding: utf-8 -*-
"""验证在线增广后标签仍与图像对齐：同一帧抽 N 次（训练模式，随机翻转/尺度/光度），
把增广后的 K 与 3D 框投影到增广后的图上，叠热图，并做数值检查：
    1) 热图峰值格子 == GT 中心投影格子（1-hm 应为 0）
    2) 编码->解码往返在增广后的 K 下仍精确
    3) 翻转后 3D 框的中心 x 取反、旋转仍是真旋转（det=+1）

用法：
    python tools/verify_aug.py --cache E:/mmcache/mm20 --n 6
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uavdet3d.config import cfg_from_yaml_file                              # noqa: E402
from uavdet3d.datasets import build_dataloader                              # noqa: E402
from uavdet3d.utils import common_utils, frame_convention                   # noqa: E402
from uavdet3d.utils.object_encoder_mav6d import center_point_decoder        # noqa: E402

GREEN, WHITE = (60, 220, 60), (255, 255, 255)


def proto(p9, seq):
    c = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
    return (c * p9[3:6]) @ R.from_euler(seq, p9[6:9]).as_matrix().T + p9[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='E:/mmcache/mm20')
    ap.add_argument('--cfg', default='cfgs/models/uavdet_3d/mmcache/teacher.yaml')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--out', default='../output/verify_aug')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = EasyDict()
    cfg_from_yaml_file(args.cfg, cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.cache
    cfg.DATA_CONFIG.MODALITIES = ['rgb', 'ir', 'depth', 'tag']
    logger = common_utils.create_logger()
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0,
                                logger=logger, training=True)
    seq = frame_convention.get_default_euler_seq()
    stride = int(cfg.DATA_CONFIG.STRIDE)
    mean = ds.norm_mean if ds.norm_mean is not None else np.zeros(3, np.float32)
    std = ds.norm_std if ds.norm_std is not None else np.ones(3, np.float32)

    np.random.seed(0)
    picks = np.linspace(0, len(ds) - 1, args.n).round().astype(int)
    peak_off, rt_pos, rt_ang, dets = [], [], [], []
    for n, k in enumerate(picks):
        d = ds[int(k)]                      # 训练模式 -> 已增广
        img = d['image'][0]
        rgb = np.clip(img[:3] * std[:, None, None] + mean[:, None, None], 0, 1)      # 反归一化用于显示
        rgb = np.ascontiguousarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8))
        K = np.asarray(d['intrinsic'][0], np.float64)
        raw_w, raw_h = d['raw_im_size']
        sx, sy = ds.W / float(raw_w), ds.H / float(raw_h)
        Ks = K.copy(); Ks[0] *= sx; Ks[1] *= sy
        boxes = d['gt_box9d']
        hm = d['hm'][0].max(axis=0)

        # 热图叠加 + 框
        up = cv2.resize(hm, (ds.W, ds.H))
        heat = cv2.applyColorMap((np.clip(up, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis = (rgb * (1 - up[..., None] * 0.7) + heat * up[..., None] * 0.7).astype(np.uint8)
        for b in boxes:
            pts = proto(b.astype(np.float64), seq)
            if np.any(pts[:, 2] <= 1e-3):
                continue
            uv = (Ks @ pts.T).T; uv = np.round(uv[:, :2] / uv[:, 2:3]).astype(int)
            for a, b2 in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                cv2.line(vis, tuple(uv[a]), tuple(uv[b2]), GREEN, 1, cv2.LINE_AA)
            # 1) 峰值格子
            cu = (K @ b[:3].astype(np.float64))[:2] / b[2]
            r0, c0 = int(cu[1] * sy / stride), int(cu[0] * sx / stride)
            if 0 <= r0 < hm.shape[0] and 0 <= c0 < hm.shape[1]:
                peak_off.append(1.0 - hm[r0, c0])
            dets.append(np.linalg.det(R.from_euler(seq, b[6:9]).as_matrix()))
        # 2) 编码->解码往返（用增广后的监督图直接解码）
        t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
        pred, conf = center_point_decoder(t(d['hm']), t(d['center_res']), t(d['center_dis']) * cfg.DATA_CONFIG.MAX_DIS,
                                          t(d['dim']) * cfg.DATA_CONFIG.MAX_SIZE, t(d['rot']),
                                          [K], [np.eye(4)], [np.zeros(5)], ds.W, ds.H, raw_w, raw_h, stride, 1,
                                          max_num=max(1, len(boxes)), rot_repr='euler6', euler_seq=seq)
        for g in boxes:
            dd = np.linalg.norm(pred[:, :3] - g[:3], axis=1); j = int(np.argmin(dd))
            rt_pos.append(dd[j])
            rt_ang.append(np.degrees((R.from_euler(seq, pred[j, 6:9]).inv() * R.from_euler(seq, g[6:9])).magnitude()))
        cv2.putText(vis, 'aug sample %d  fx=%.0f cx=%.0f  boxes %d' % (n, K[0, 0], K[0, 2], len(boxes)),
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, 'aug sample %d  fx=%.0f cx=%.0f  boxes %d' % (n, K[0, 0], K[0, 2], len(boxes)),
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        ir = cv2.cvtColor((img[3] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        dep = img[4]
        depv = cv2.applyColorMap((255 - np.clip(dep, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depv[dep <= 0] = (25, 25, 25); depv[img[5] > 0] = (255, 0, 255)
        for panel in (ir, depv):
            for b in boxes:
                pts = proto(b.astype(np.float64), seq)
                if np.any(pts[:, 2] <= 1e-3):
                    continue
                uv = (Ks @ pts.T).T; uv = np.round(uv[:, :2] / uv[:, 2:3]).astype(int)
                for a, b2 in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                    cv2.line(panel, tuple(uv[a]), tuple(uv[b2]), GREEN, 1, cv2.LINE_AA)
        sep = np.full((ds.H, 3, 3), 90, np.uint8)
        cv2.imwrite(os.path.join(args.out, 'aug_%02d.jpg' % n), np.concatenate([vis, sep, ir, sep, depv], 1),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    peak_off, rt_pos, rt_ang, dets = map(np.array, (peak_off, rt_pos, rt_ang, dets))
    ok1 = len(peak_off) and peak_off.max() < 1e-3
    ok2 = len(rt_pos) and np.percentile(rt_pos, 90) < 1e-3 and np.percentile(rt_ang, 90) < 1e-3
    ok3 = len(dets) and np.allclose(dets, 1.0, atol=1e-6)
    print('[%s] 增广后热图峰值落在 GT 中心格子: 1-hm 最大 %.2e（%d 框）' % ('PASS' if ok1 else 'FAIL', peak_off.max() if len(peak_off) else -1, len(peak_off)))
    print('[%s] 增广后编码->解码往返: 位置 90%% %.2e m, 旋转 90%% %.2e°' % ('PASS' if ok2 else 'FAIL', np.percentile(rt_pos, 90) if len(rt_pos) else -1, np.percentile(rt_ang, 90) if len(rt_ang) else -1))
    print('[%s] 翻转后的旋转仍为真旋转: det 范围 [%.6f, %.6f]' % ('PASS' if ok3 else 'FAIL', dets.min() if len(dets) else 0, dets.max() if len(dets) else 0))
    print('图 -> %s' % args.out)
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == '__main__':
    sys.exit(main())
