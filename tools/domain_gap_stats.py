# -*- coding: utf-8 -*-
"""量化源域(仿真 UAV-MM3D 近距离子集) 与 目标域(真实 MAV6D) 的差异。

迁移基准里需要一节「域差异有多大」的客观描述，否则「迁移失败」只是个现象。
这里只做不需要 GPU 的部分：相机、几何、外观三层统计，两边同口径采样。

用法：
    python tools/domain_gap_stats.py --n 200 --out ../output/domain_gap.json
"""
import argparse
import json
import os
import pickle
import random
import sys

import cv2
import numpy as np

SRC_ROOT = 'E:/data_collect'
SRC_LIST = 'cfgs/subsets/near15/near_train.txt'
DST_ROOT = 'E:/MAV6D'
# mav6d_det_dataset.py 里写死的标定
DST_K = np.array([[1979.4, 0.3984, 976.8189], [0., 1980.4, 561.4514], [0., 0., 1.]])
DST_WH = (1920, 1080)


def img_stats(img):
    """外观层统计：亮度/对比度/通道均值/边缘密度/灰度熵。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    return dict(brightness=float(g.mean()), contrast=float(g.std()),
                edge_density=float((mag > 50).mean()), entropy=ent,
                saturation=float(sat))


def collect_source(n, seed):
    rows = []
    with open(SRC_LIST) as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith('#'):
                p = ln.split()
                rows.append((p[0], p[1]))
    random.Random(seed).shuffle(rows)

    app, depth, size, pxsize, uv = [], [], [], [], []
    wh = None
    got = 0
    for seq, frame in rows:
        if got >= n:
            break
        ip = os.path.join(SRC_ROOT, seq, 'images_rgb', frame)
        bp = os.path.join(SRC_ROOT, seq, 'boxes_rgb', os.path.splitext(frame)[0] + '.pkl')
        if not (os.path.exists(ip) and os.path.exists(bp)):
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        wh = (img.shape[1], img.shape[0])
        try:
            with open(bp, 'rb') as f:
                raw = pickle.load(f)
        except Exception:
            continue
        fx = img.shape[1] / 2.0     # CARLA 默认 90 度水平 FOV
        hit = False
        for row in raw:
            if not isinstance(row[0], str):
                continue
            pts = np.array(row[1:], dtype=np.float32).reshape(8, 3)
            c3 = pts.mean(axis=0)
            z = float(c3[2])
            if z <= 0.1:
                continue
            d = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
            depth.append(z)
            size.append(d)
            pxsize.append(d * fx / z)
            uv.append([float(c3[0] / z * fx + img.shape[1] / 2.0),
                       float(c3[1] / z * fx + img.shape[0] / 2.0)])
            hit = True
        if hit:
            app.append(img_stats(img))
            got += 1
    return dict(name='source_sim_near15', n_frames=got,
                image_wh=list(wh) if wh else None, fov_x_deg=90.0,
                app=app, depth=depth, size=size, pxsize=pxsize, uv=uv)


def collect_target(n, seed):
    from uavdet3d.datasets.mav6d.mav6d_utils import read_truth_Rt
    cands = []
    for cls in ('mavic2', 'phantom4'):
        sp = os.path.join(DST_ROOT, cls, 'split', 'train.txt')
        if not os.path.exists(sp):
            continue
        for ln in open(sp):
            ln = ln.strip()
            if ln:
                cands.append((cls, ln.split('/')[-3:]))
    random.Random(seed).shuffle(cands)

    fx = DST_K[0, 0]
    app, depth, size, pxsize, uv = [], [], [], [], []
    got = 0
    OB = 0.34       # 官方 util.py 给的 phantom4 角点范围，两个型号共用
    for cls, (scene, seq, frame) in cands:
        if got >= n:
            break
        ip = os.path.join(DST_ROOT, cls, 'JPEGImages', scene, seq, frame)
        lp = os.path.join(DST_ROOT, cls, 'labels', scene, seq,
                          os.path.splitext(frame)[0] + '.txt')
        if not (os.path.exists(ip) and os.path.exists(lp)):
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        try:
            _, t = read_truth_Rt(lp)
        except Exception:
            continue
        z = float(t[2])
        if z <= 0.1:
            continue
        depth.append(z)
        size.append(OB)
        pxsize.append(OB * fx / z)
        uv.append([float(t[0] / z * fx + DST_K[0, 2]),
                   float(t[1] / z * DST_K[1, 1] + DST_K[1, 2])])
        app.append(img_stats(img))
        got += 1
    fov = float(np.degrees(2 * np.arctan(DST_WH[0] / 2.0 / fx)))
    return dict(name='target_real_mav6d', n_frames=got, image_wh=list(DST_WH),
                fov_x_deg=fov, app=app, depth=depth, size=size, pxsize=pxsize, uv=uv)


def summarize(d):
    out = {'name': d['name'], 'n_frames': d['n_frames'],
           'image_wh': d['image_wh'], 'fov_x_deg': d['fov_x_deg']}
    keys = list(d['app'][0].keys()) if d['app'] else []
    for k in keys:
        a = np.array([x[k] for x in d['app']])
        out['app_' + k] = [float(a.mean()), float(a.std())]
    for k in ('depth', 'size', 'pxsize'):
        a = np.array(d[k])
        out[k] = dict(median=float(np.median(a)), mean=float(a.mean()),
                      p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                      min=float(a.min()), max=float(a.max()), n=int(len(a)))
    uv = np.array(d['uv'])
    out['center_uv_std_px'] = [float(uv[:, 0].std()), float(uv[:, 1].std())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='../output/domain_gap.json')
    args = ap.parse_args()

    s = summarize(collect_source(args.n, args.seed))
    t = summarize(collect_target(args.n, args.seed))

    print('\n%-26s %-24s %-24s' % ('维度', '源域(仿真)', '目标域(真实)'))
    print('-' * 76)
    print('%-26s %-24s %-24s' % ('采样帧数', s['n_frames'], t['n_frames']))
    print('%-26s %-24s %-24s' % ('图像分辨率', 'x'.join(map(str, s['image_wh'])),
                                 'x'.join(map(str, t['image_wh']))))
    print('%-26s %-24s %-24s' % ('水平视场角 (deg)', '%.1f' % s['fov_x_deg'],
                                 '%.1f' % t['fov_x_deg']))
    for label, k in (('目标深度 (m)', 'depth'), ('目标物理尺寸 (m)', 'size'),
                     ('目标像素大小 (px)', 'pxsize')):
        f = lambda d: '%.2f  [%.2f, %.2f]' % (d[k]['median'], d[k]['p05'], d[k]['p95'])
        print('%-26s %-24s %-24s' % (label + ' 中位[5,95]', f(s), f(t)))
    for label, k in (('亮度', 'app_brightness'), ('对比度', 'app_contrast'),
                     ('边缘密度', 'app_edge_density'), ('灰度熵', 'app_entropy'),
                     ('饱和度', 'app_saturation')):
        print('%-26s %-24s %-24s' % (label,
                                     '%.3f ± %.3f' % tuple(s[k]),
                                     '%.3f ± %.3f' % tuple(t[k])))
    print('%-26s %-24s %-24s' % ('目标中心散布 (px std)',
                                 '%.0f, %.0f' % tuple(s['center_uv_std_px']),
                                 '%.0f, %.0f' % tuple(t['center_uv_std_px'])))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'source': s, 'target': t}, f, indent=2)
    print('\n写出 %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
