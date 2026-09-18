# -*- coding: utf-8 -*-
"""可视化：同一个标注框里，两域「看得见的结构」各占多少（2026-09-18）。

每个小块都把裁块缩放成「标注 2D 框 = 96 px」，所以两域可以直接横向比。
画了三样东西：绿色 = 标注 3D 框的投影线框；黄色 = 2D 外接框；红色虚线 = 框的 0.64 倍（phantom4 网格被压缩的比例）。
第三行是「抹掉 0.64 以外环带」的仿真图，也就是消融实验真正喂给模型的东西。

    python annot_audit/viz_target_fill.py [--n 7] [--out <jpg>]
"""
import argparse
import os
import sys

import cv2
import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(TOOLS)
from scipy.spatial.transform import Rotation as R   # noqa: E402

import visible_extent as VE   # noqa: E402
from prop_ablation import erase_region   # noqa: E402

import os as _os
BOX = int(_os.environ.get('VIZ_BOX', '96'))   # 每块里标注框的边长（像素）
PAD = 2.4          # 裁块 = 标注框的 2.4 倍
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(b):
    return (VE.PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]


def tile(img, K, b, erase_core=None, label=''):
    C = corners(b)
    uv = (K @ C.T).T
    if (uv[:, 2] <= 1e-6).any():
        return None
    uv = uv[:, :2] / uv[:, 2:3]
    x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
    side = max(x1 - x0, y1 - y0)
    if side < float(_os.environ.get('VIZ_MINPX', '12')):
        return None
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = side * PAD / 2
    ix0, iy0, ix1, iy1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    if ix0 < 0 or iy0 < 0 or ix1 > img.shape[1] or iy1 > img.shape[0]:
        return None
    src = img
    if erase_core is not None:
        src = erase_region(img, np.array([x0, y0, x1, y1]), erase_core, 1.0)
        if src is None:
            return None
    crop = src[iy0:iy1, ix0:ix1]
    n = int(round(BOX * PAD))
    s = n / float(crop.shape[1])
    crop = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA if crop.shape[0] > n else cv2.INTER_CUBIC)
    out = np.ascontiguousarray(crop)
    # 线框与框（坐标映射到裁块）
    m = lambda p: (int(round((p[0] - ix0) * s)), int(round((p[1] - iy0) * s)))
    for i, j in EDGES:
        cv2.line(out, m(uv[i]), m(uv[j]), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(out, m((x0, y0)), m((x1, y1)), (0, 255, 255), 1)
    hw = side * 0.64 / 2
    p0, p1 = m((cx - hw, cy - hw)), m((cx + hw, cy + hw))
    for k in range(p0[0], p1[0], 6):                     # 红色虚线框 = 0.64 倍
        cv2.line(out, (k, p0[1]), (min(k + 3, p1[0]), p0[1]), (0, 0, 255), 1)
        cv2.line(out, (k, p1[1]), (min(k + 3, p1[0]), p1[1]), (0, 0, 255), 1)
    for k in range(p0[1], p1[1], 6):
        cv2.line(out, (p0[0], k), (p0[0], min(k + 3, p1[1])), (0, 0, 255), 1)
        cv2.line(out, (p1[0], k), (p1[0], min(k + 3, p1[1])), (0, 0, 255), 1)
    if label:
        cv2.putText(out, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=7)
    ap.add_argument('--out', default='E:/UavIndoorSim/Saved/quality_final/target_fill.jpg')
    a = ap.parse_args()
    rows = []
    for name, gen, erase in (('real', VE.load_mav(600), None), ('sim', VE.load_sim(900), None), ('sim_erased', VE.load_sim(900), 0.64)):
        tiles = []
        for img, K, b in gen:
            t = tile(img, K, b, erase_core=erase, label='%.2fm %.0fpx' % (b[2], max(1.0, 1.0)))
            if t is not None:
                tiles.append(t)
            if len(tiles) >= a.n:
                break
        if tiles:
            rows.append(np.hstack(tiles))
    if not rows:
        print('没有可用样例')
        return
    w = min(r.shape[1] for r in rows)
    grid = np.vstack([r[:, :w] for r in rows])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, grid)
    print('上排 MAV6D 真实 / 中排 仿真原图 / 下排 仿真抹掉 0.64 以外环带 ->', a.out)
    print('绿线 = 标注 3D 框投影，黄框 = 2D 外接框（每块都缩放成 96 px），红虚线 = 框的 0.64 倍')


if __name__ == '__main__':
    main()
