# -*- coding: utf-8 -*-
"""两域「目标实际可见的像素跨度 / 标注框跨度」对比（2026-09-18）。

为什么要量这个：纯仿真模型在真实图上，隐含认为目标的表观大小比标注框小 21%（深度偏大 1.27 倍，
斜率 1.14、相关性 0.74，不是塌回先验）。两域标注框与投影的几何关系已验证一致（形状因子 1.36 vs 1.38），
所以差别只可能在「图上真正看得见多大」——仿真桨叶静止清晰可见、真实桨叶转起来几乎看不见就是这一类差别。

做法：在标注 2D 框外扩 2 倍裁块，缩到固定大小（标注框 = 64 px），算梯度幅值，
以外围环带的中位数当背景，扣掉背景后取横/纵方向累积能量的 10%~90% 跨度作为「可见跨度」。
两域同一口径，只比相对值。

    python annot_audit/visible_extent.py [--n 300] [--save-fig]
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
os.chdir(TOOLS)
from scipy.spatial.transform import Rotation as R   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
BOX_PX = 64          # 裁块内标注框的边长（两域统一）
PAD = 2.0            # 裁块 = 标注框的 2 倍


def gt_box2d(b, K, W, H):
    C = (PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
    uv = (K @ C.T).T
    if (uv[:, 2] <= 1e-6).any():
        return None
    uv = uv[:, :2] / uv[:, 2:3]
    x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
    if x0 < 2 or y0 < 2 or x1 > W - 3 or y1 > H - 3:
        return None                      # 贴边的目标不要，可见跨度会被画面切断
    return np.array([x0, y0, x1, y1])


def visible_span(img, box):
    """-> (可见跨度 / 标注跨度) 的 (横, 纵)，以及裁块（给出图用）。"""
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    side = max(box[2] - box[0], box[3] - box[1]) * PAD
    half = side / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None, None
    crop = img[y0:y1, x0:x1]
    n = int(round(BOX_PX * PAD))
    crop = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA if crop.shape[0] > n else cv2.INTER_LINEAR)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    G = np.hypot(gx, gy)
    ring = np.ones((n, n), bool)
    m = int(round(BOX_PX * PAD / 2 - BOX_PX * 0.75))      # 环带 = 裁块里标注框 1.5 倍以外
    ring[m:n - m, m:n - m] = False
    bg = np.median(G[ring]) if ring.any() else 0.0
    w = np.clip(G - bg, 0, None)
    w[ring] = 0                                            # 只用标注框 1.5 倍以内，避免背景杂物
    if w.sum() < 1e-6:
        return None, crop
    out = []
    for axis in (0, 1):                                    # 0: 按列求和 -> 横向跨度
        prof = w.sum(axis=1 - axis)
        c = np.cumsum(prof) / prof.sum()
        lo = np.searchsorted(c, 0.10)
        hi = np.searchsorted(c, 0.90)
        out.append((hi - lo) / BOX_PX)                     # 标注框在裁块里就是 BOX_PX
    return np.array(out), crop


def load_mav(n):
    root = 'E:/mmcache/mav6d_cn/test'
    idx = pickle.load(open(os.path.join(root, 'index.pkl'), 'rb'))
    rgb = np.load(os.path.join(root, 'rgb.npy'), mmap_mode='r')
    step = max(1, len(idx['valid_idx']) // n)
    for i in idx['valid_idx'][::step][:n]:
        m = idx['metas'][int(i)]
        K = np.asarray(m['K_in']).reshape(3, 3)
        img = np.ascontiguousarray(rgb[int(i)])
        for b in np.asarray(m['boxes9d']).reshape(-1, 9):
            if np.abs(b).sum() == 0 or b[2] <= 0:
                continue
            yield img, K, b


def load_sim(n, max_size=0.45):
    root = 'E:/mmcache/full_v2/train'
    idx = pickle.load(open(os.path.join(root, 'index.pkl'), 'rb'))
    blob = np.memmap(os.path.join(root, 'rgb_jpg.bin'), dtype=np.uint8, mode='r')
    ji = np.load(os.path.join(root, 'rgb_jpg_index.npy'), mmap_mode='r')
    step = max(1, len(idx['valid_idx']) // (n * 3))
    cnt = 0
    for i in idx['valid_idx'][::step]:
        if cnt >= n:
            return
        m = idx['metas'][int(i)]
        K = np.asarray(m['K_in']).reshape(3, 3)
        img = None
        for b in np.asarray(m['boxes9d']).reshape(-1, 9):
            if np.abs(b).sum() == 0 or b[2] <= 0 or np.linalg.norm(b[3:6]) > max_size * 1.6:
                continue
            if img is None:
                o, ln = ji[int(i)]
                img = cv2.imdecode(np.asarray(blob[int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
            cnt += 1
            yield img, K, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--save-fig', action='store_true')
    a = ap.parse_args()
    res = {}
    crops = {}
    for name, gen in (('MAV6D 真实', load_mav(a.n)), ('仿真 小机', load_sim(a.n))):
        rows, ex = [], []
        for img, K, b in gen:
            box = gt_box2d(b, K, img.shape[1], img.shape[0])
            if box is None:
                continue
            span, crop = visible_span(img, box)
            if span is None:
                continue
            rows.append(span)
            if len(ex) < 8:
                ex.append(crop)
        res[name] = np.array(rows)
        crops[name] = ex
        r = res[name]
        print('%-10s n=%4d | 可见跨度/标注跨度  横 中位 %.2f (p25 %.2f p75 %.2f)  纵 中位 %.2f (p25 %.2f p75 %.2f)' % (
            name, len(r), np.median(r[:, 0]), *np.percentile(r[:, 0], [25, 75]),
            np.median(r[:, 1]), *np.percentile(r[:, 1], [25, 75])), flush=True)
    if len(res) == 2:
        a_, b_ = res['仿真 小机'], res['MAV6D 真实']
        print('仿真 / 真实 = 横 %.2f  纵 %.2f  —— >1 表示同样标注框下仿真图上看得见的部分更大（模型会因此低估真实目标的表观大小 -> 深度估大）' % (
            np.median(a_[:, 0]) / np.median(b_[:, 0]), np.median(a_[:, 1]) / np.median(b_[:, 1])))
    if a.save_fig and crops:
        out = 'E:/UavIndoorSim/Saved/quality_final/visible_extent.jpg'
        rowsim = [np.hstack(crops[k][:8]) for k in crops if len(crops[k]) >= 8]
        if rowsim:
            cv2.imwrite(out, np.vstack(rowsim))
            print('样例（上排 MAV6D，下排 仿真小机）->', out)


if __name__ == '__main__':
    main()
