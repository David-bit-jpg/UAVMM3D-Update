# -*- coding: utf-8 -*-
"""双域一致性审查：直接从缓存数据实测两域的几何与光度约定，而不是读配置或代码。

2026-09-16 起因：`camnorm_sim_pp_realsize.yaml` 只改了 DATA_PATH，沿用上一个缓存的 NORM_MEAN/NORM_STD，
网络实际看到的输入是 均值 [+0.25,+0.11,-0.13] 而不是 0 —— 这类问题配置 diff 能看出来，
但「标签手性、机头轴、上轴朝向、内参主点、分辨率、锐度」这些只能从数据量。建新缓存后都该跑一次。

    python tools/audit_domain_consistency.py --sim E:/mmcache/pp_realsize --real E:/mmcache/mav6d_cn
"""
import argparse
import os
import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def load(root, split):
    idx = pickle.load(open(os.path.join(root, split, 'index.pkl'), 'rb'))
    return idx, [idx['metas'][int(i)] for i in idx['valid_idx']]


def frames(root, split, sel):
    jb = os.path.join(root, split, 'rgb_jpg.bin')
    if os.path.exists(jb):
        ji = np.load(os.path.join(root, split, 'rgb_jpg_index.npy'), mmap_mode='r')
        with open(jb, 'rb') as f:
            for i in sel:
                o, ln = ji[int(i)]
                f.seek(int(o))
                yield cv2.imdecode(np.frombuffer(f.read(int(ln)), np.uint8), cv2.IMREAD_COLOR), True
    else:
        mm = np.load(os.path.join(root, split, 'rgb.npy'), mmap_mode='r')
        for i in sel:
            yield np.asarray(mm[int(i)]), False


def axes_stats(metas):
    """标签机体三轴在相机系里的朝向：x=机头 y=左 z=上。相机系 y 朝下，所以水平飞行时 z 应 ≈ (0,-1,0)。"""
    ax = {0: [], 1: [], 2: []}
    lwh, zs, zv = [], [], []
    for m in metas:
        K = np.asarray(m['K_in']).reshape(3, 3)
        f = float(np.sqrt(K[0, 0] * K[1, 1]))
        for b in np.asarray(m['boxes9d']).reshape(-1, 9):
            if np.abs(b).sum() == 0 or b[2] <= 1e-6:
                continue
            M = R.from_euler('xyz', b[6:9]).as_matrix()
            for k in range(3):
                ax[k].append(M[:, k])
            lwh.append(b[3:6]); zs.append(b[2]); zv.append(b[2] * 512 / f)
    return {k: np.array(v) for k, v in ax.items()}, np.array(lwh), np.array(zs), np.array(zv)


def report(tag, root, split, nimg=120):
    idx, metas = load(root, split)
    H, W = idx.get('H'), idx.get('W')
    Ks = np.array([np.asarray(m['K_in']).reshape(3, 3) for m in metas])
    ax, lwh, zs, zv = axes_stats(metas)
    sel = [int(i) for i in list(idx['valid_idx'])[:: max(1, len(idx['valid_idx']) // nimg)][:nimg]]
    lap, shape, jpg = [], None, None
    for img, is_jpg in frames(root, split, sel):
        if img is None:
            continue
        jpg = is_jpg
        shape = img.shape[:2]
        small = cv2.resize(img, (512, 288), interpolation=cv2.INTER_AREA) if img.shape[1] != 512 else img
        lap.append(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    print('\n' + '=' * 108)
    print('%s   %s/%s   帧 %d，目标 %d' % (tag, root, split, len(metas), len(lwh)))
    print('  缓存分辨率 index H,W = %s,%s；实际图 %s；存储 = %s' % (H, W, shape, 'JPEG(有损)' if jpg else 'RAW'))
    print('  内参 K_in: fx %.1f~%.1f  fy %.1f~%.1f  fx/fy 中位 %.4f  cx/W 中位 %.4f  cy/H 中位 %.4f' % (
        Ks[:, 0, 0].min(), Ks[:, 0, 0].max(), Ks[:, 1, 1].min(), Ks[:, 1, 1].max(),
        np.median(Ks[:, 0, 0] / Ks[:, 1, 1]), np.median(Ks[:, 0, 2]) / (W or 1), np.median(Ks[:, 1, 2]) / (H or 1)))
    print('  深度 Z 中位 %.2f m (p10 %.2f p90 %.2f) | 虚拟深度 Zv 中位 %.2f (p10 %.2f p90 %.2f)' % (
        np.median(zs), np.percentile(zs, 10), np.percentile(zs, 90),
        np.median(zv), np.percentile(zv, 10), np.percentile(zv, 90)))
    print('  框尺寸 l,w,h 中位 %s  对角线 中位 %.3f 均值 %.3f' % (
        np.round(np.median(lwh, 0), 3), np.median(np.linalg.norm(lwh, axis=1)), np.linalg.norm(lwh, axis=1).mean()))
    for k, nm in ((0, '机头 x'), (1, '左 y'), (2, '上 z')):
        v = ax[k]
        print('    机体轴 %s 在相机系: 中位 (%+.2f, %+.2f, %+.2f)  |  朝上分量 -y 为正的比例 %.3f' % (
            nm, *np.median(v, 0), float((v[:, 1] < 0).mean())))
    print('  画面锐度（拉普拉斯方差，512x288 上）中位 %.1f  [p25 %.1f, p75 %.1f]' % (
        np.median(lap), np.percentile(lap, 25), np.percentile(lap, 75)))
    return dict(lap=np.median(lap), zv=np.median(zv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='E:/mmcache/pp_realsize')
    ap.add_argument('--real', default='E:/mmcache/mav6d_cn')
    ap.add_argument('--also', nargs='*', default=[])
    a = ap.parse_args()
    report('仿真（训练域）', a.sim, 'train')
    report('真实 MAV6D（目标域 test）', a.real, 'test')
    for r in a.also:
        report('其它', r, 'train')


if __name__ == '__main__':
    main()
