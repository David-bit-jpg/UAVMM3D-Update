# -*- coding: utf-8 -*-
"""给 mm 缓存的每一帧算一个「RGB 可见度」分：最近合格目标处的 RGB 局部对比度（灰度级）。

    对比度 = |目标窗口内均值 − 外围环带均值|，窗口半径 ≈ 目标在图上的半宽（至少 3 px），环带取 3 倍半径。
    每帧取所有合格目标里的最大值；没有合格目标记 -1。

为什么需要它：mm20 有 61% 是夜间帧，实测夜间目标对比度中位只有 2.3–2.9（目标亮度 3–21/255），
白天 8.5–15.4。对只看 RGB 的学生来说，目标在 RGB 里不可见的帧是纯噪声；而目标域 MAV6D 是明亮室内。
按对比度筛（而不是按天气标签）能保留有路灯的夜景和白天雾雨雪里仍看得见的目标。

数据集通过 MIN_RGB_VIS 读取 <split>/vis_score.npy 做过滤。用法：
    python tools/mm_vis_score.py --cache E:/mmcache/mm20
"""
import argparse
import collections
import os
import pickle
import sys
import time

import numpy as np


def score_split(cache, split):
    idx = pickle.load(open(os.path.join(cache, split, 'index.pkl'), 'rb'))
    rgb = np.load(os.path.join(cache, split, 'rgb.npy'), mmap_mode='r')
    H, W = idx['H'], idx['W']
    raw_w, raw_h = None, None
    N = len(idx['metas'])
    score = np.full(N, -1.0, np.float32)
    for i in idx['valid_idx']:
        m = idx['metas'][i]
        if raw_w is None:
            raw_w, raw_h = m['raw_wh']
        sx, sy = W / float(raw_w), H / float(raw_h)
        K = m['K_raw'].astype(np.float64).copy()
        K[0] *= sx
        K[1] *= sy
        g = np.asarray(rgb[i]).astype(np.float32).mean(axis=2)
        best = -1.0
        for b, q in zip(m['boxes9d'], m['qualified']):
            if not q:
                continue
            uv = (K @ b[:3].astype(np.float64))[:2] / b[2]
            u, v = int(uv[0]), int(uv[1])
            r = max(3, int(0.5 * max(b[3:6]) * K[0, 0] / b[2]))
            y0, y1, x0, x1 = max(0, v - r), min(H, v + r + 1), max(0, u - r), min(W, u + r + 1)
            Y0, Y1, X0, X1 = max(0, v - 3 * r), min(H, v + 3 * r + 1), max(0, u - 3 * r), min(W, u + 3 * r + 1)
            inner, outer = g[y0:y1, x0:x1], g[Y0:Y1, X0:X1]
            if inner.size == 0 or outer.size <= inner.size:
                continue
            ring = (outer.sum() - inner.sum()) / (outer.size - inner.size)
            best = max(best, abs(float(inner.mean()) - float(ring)))
        score[i] = best
    np.save(os.path.join(cache, split, 'vis_score.npy'), score)
    s = score[idx['valid_idx']]
    weather = [idx['metas'][i]['seq'].split('/')[3] for i in idx['valid_idx']]
    print('%s: %d 帧  ' % (split, len(s)) + '  '.join('>=%g: %d' % (t, (s >= t).sum()) for t in (3, 5, 8, 15)))
    for t in (5,):
        keep = collections.Counter(w for w, v in zip(weather, s) if v >= t)
        print('   >=%g 按天气: %s' % (t, dict(sorted(keep.items()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='E:/mmcache/mm20')
    ap.add_argument('--splits', nargs='*', default=['train', 'test'])
    args = ap.parse_args()
    t0 = time.time()
    for sp in args.splits:
        score_split(args.cache, sp)
    print('用时 %.0f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
