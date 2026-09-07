# -*- coding: utf-8 -*-
"""MAV6D 在线增广的正确性自检：几何变换必须与内参 / 畸变 / 3D 框保持自洽。

    D:/Miniconda3/envs/city/python.exe tools/verify_mav6d_aug.py

检查项：
  A1 关掉增广时，样本与改动前完全一致（热图峰值位置、框、内参）
  A2 水平翻转：翻转后 GT 中心的投影像素 u 应等于 (raw_w-1) - u_原（畸变项已按 p2 变号补偿）
  A3 水平翻转：热图峰值格子应镜像；框的旋转矩阵应满足 R' = M R M
  A4 随机尺度：投影中心必须仍在画面内，且热图峰值与用新内参重算的投影一致
  A5 光度增广：不改几何（热图峰值、框、内参都不变）
  A6 编码器一致性：无论怎么增广，用返回的 K/D 重算的投影都应落在热图峰值格子里
"""
import os
import sys

import numpy as np
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, 'E:/Open3DUAVDet')
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file  # noqa: E402
from uavdet3d.datasets import build_dataloader  # noqa: E402

CFG = 'cfgs/models/uavdet_3d/mav6d/centerdet.yaml'
OK = []


def check(name, cond, msg):
    OK.append(bool(cond))
    print('%-4s %-5s %s' % (name, 'OK' if cond else 'FAIL', msg))


def build(aug=None, training=True):
    cfg = EasyDict()
    cfg_from_yaml_file(CFG, cfg)
    cfg_from_list(['DATA_CONFIG.DATA_PATH', 'E:/MAV6D', 'DATA_CONFIG.SAMPLED_INTERVAL.train', '200'], cfg)
    if aug is not None:
        cfg.DATA_CONFIG.AUG = aug
    import logging
    logging.basicConfig(level=logging.WARNING)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0,
                                training=training, logger=logging.getLogger('v'))
    return ds


def project(box, K, D):
    x, y, z = box[:3]
    xp, yp = x / z, y / z
    r2 = xp * xp + yp * yp
    rad = 1.0 + D[0] * r2 + D[1] * r2 * r2 + D[4] * r2 * r2 * r2
    xd = xp * rad + 2.0 * D[2] * xp * yp + D[3] * (r2 + 2.0 * xp * xp)
    yd = yp * rad + D[2] * (r2 + 2.0 * yp * yp) + 2.0 * D[3] * xp * yp
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]


def peak(d):
    hm = np.asarray(d['hm'])[0]
    i = np.unravel_index(np.argmax(hm), hm.shape)
    return i[2], i[1]          # (w_idx, h_idx)


def main():
    os.chdir('E:/Open3DUAVDet/tools')
    np.random.seed(0)
    ds0 = build(aug=None)
    d0 = ds0[3]
    K0, D0 = np.asarray(d0['intrinsic'])[0], np.asarray(d0['distortion'])[0]
    b0 = np.asarray(d0['gt_box9d'])[0]
    u0, v0 = project(b0, K0, D0)
    raw_w, raw_h = float(ds0.raw_im_width), float(ds0.raw_im_hight)
    sx = ds0.new_im_width / raw_w / ds0.stride
    sy = ds0.new_im_hight / raw_h / ds0.stride
    p0 = peak(d0)
    check('A1', abs(u0 * sx - p0[0]) < 1.0 and abs(v0 * sy - p0[1]) < 1.0,
          '关增广：投影 (%.1f, %.1f) -> 热图格 (%.1f, %.1f)，峰值 %s' % (u0, v0, u0 * sx, v0 * sy, p0))

    # ---- 只翻转 ----
    ds1 = build(aug={'hflip': 1.0})
    d1 = ds1[3]
    K1, D1 = np.asarray(d1['intrinsic'])[0], np.asarray(d1['distortion'])[0]
    b1 = np.asarray(d1['gt_box9d'])[0]
    u1, v1 = project(b1, K1, D1)
    check('A2', abs(u1 - ((raw_w - 1) - u0)) < 0.5 and abs(v1 - v0) < 0.5,
          '翻转后投影 u=%.2f，期望 %.2f（镜像）；v=%.2f vs %.2f' % (u1, (raw_w - 1) - u0, v1, v0))
    M = np.diag([-1.0, 1.0, 1.0])
    R0 = R.from_euler('xyz', b0[6:9]).as_matrix()
    R1 = R.from_euler('xyz', b1[6:9]).as_matrix()
    check('A3', float(np.abs(R1 - M @ R0 @ M).max()) < 1e-5 and abs(b1[0] + b0[0]) < 1e-5,
          'R\' = M R M 最大差 %.2e；x 取反 %.3f vs %.3f' % (float(np.abs(R1 - M @ R0 @ M).max()), b1[0], b0[0]))
    p1 = peak(d1)
    check('A3b', abs(u1 * sx - p1[0]) < 1.0, '翻转后热图峰值 %s，投影格 %.1f' % (p1, u1 * sx))

    # ---- 只缩放 ----
    ds2 = build(aug={'scale': [0.8, 1.25]})
    bad = 0
    for i in range(12):
        d = ds2[i]
        K, D = np.asarray(d['intrinsic'])[0], np.asarray(d['distortion'])[0]
        b = np.asarray(d['gt_box9d'])[0]
        u, v = project(b, K, D)
        pk = peak(d)
        if not (0 <= u < raw_w and 0 <= v < raw_h) or abs(u * sx - pk[0]) > 1.0 or abs(v * sy - pk[1]) > 1.0:
            bad += 1
    check('A4', bad == 0, '随机尺度 12 个样本：投影都在画面内且与热图峰值一致（不一致 %d 个）' % bad)

    # ---- 只光度 ----
    ds3 = build(aug={'photometric': True, 'noise': 0.01})
    d3 = ds3[3]
    K3, D3 = np.asarray(d3['intrinsic'])[0], np.asarray(d3['distortion'])[0]
    b3 = np.asarray(d3['gt_box9d'])[0]
    check('A5', np.abs(K3 - K0).max() < 1e-9 and np.abs(D3 - D0).max() < 1e-9 and np.abs(b3 - b0).max() < 1e-6
          and peak(d3) == p0, '光度增广不改几何（K/D/框/热图峰值都不变）')

    # ---- 全开 ----
    ds4 = build(aug={'hflip': 0.5, 'scale': [0.85, 1.2], 'photometric': True, 'noise': 0.01})
    bad = 0
    for i in range(20):
        d = ds4[i]
        K, D = np.asarray(d['intrinsic'])[0], np.asarray(d['distortion'])[0]
        b = np.asarray(d['gt_box9d'])[0]
        u, v = project(b, K, D)
        pk = peak(d)
        if abs(u * sx - pk[0]) > 1.0 or abs(v * sy - pk[1]) > 1.0:
            bad += 1
    check('A6', bad == 0, '全开 20 个样本：用返回的 K/D 重算的投影都落在热图峰值格（不一致 %d 个）' % bad)

    # ---- 测试时不增广 ----
    ds5 = build(aug={'hflip': 1.0, 'scale': [0.5, 0.5]}, training=False)
    d5 = ds5[3]
    check('A7', np.abs(np.asarray(d5['intrinsic'])[0] - K0).max() < 1e-9,
          '测试集不做增广（training=False 时 self.aug 为空）')

    print('\n%d/%d 通过' % (sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == '__main__':
    sys.exit(main())
