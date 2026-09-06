# -*- coding: utf-8 -*-
"""为交叉贴增广挑帧：从 mm20 索引里选「源帧」（白天、RGB 可见、最近合格无人机不太远、各机型配额）和
「背景帧」（白天、按天气分层随机），合并写成 build_mm_cache.py 能读的列表。

    D:/Miniconda3/envs/city/python.exe tools/select_paste_frames.py --out E:/data_collect/aug_paste_v1/frames.txt
"""
import argparse
import collections
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mm_cache import class_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='E:/mmcache/mm20', help='取索引与可见度分数的缓存')
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', required=True)
    ap.add_argument('--src-max-range', type=float, default=12.0, help='源帧最近合格无人机的距离上限（远了拉近要放大太多倍）')
    ap.add_argument('--per-class', type=int, default=250, help='每个机型最多多少源帧')
    ap.add_argument('--n-bg', type=int, default=1200, help='背景帧数（按天气均分）')
    ap.add_argument('--min-vis', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    d = os.path.join(args.cache, args.split)
    idx = pickle.load(open(os.path.join(d, 'index.pkl'), 'rb'))
    metas, valid = idx['metas'], idx['valid_idx']
    vis = np.load(os.path.join(d, 'vis_score.npy'))
    rng = random.Random(args.seed)

    def row(m):
        return '%s %s %d %d 0 0 %.1f' % (m['seq'], m['frame'], len(m['boxes9d']), int(m['qualified'].sum()), m['lidar_noise'])

    src_by_cls, bg_by_w = collections.defaultdict(list), collections.defaultdict(list)
    for i in valid:
        m = metas[i]
        w = m['seq'].split('/')[3]
        if not w.endswith('_day'):
            continue
        bg_by_w[w].append(row(m))
        q = [(np.linalg.norm(b[:3]), n) for b, n, qq in zip(m['boxes9d'], m['names'], m['qualified']) if qq]
        if not q or vis[i] < args.min_vis:
            continue
        r, n = min(q)
        if r <= args.src_max_range:
            src_by_cls[class_of(n) or n].append(row(m))
    src = []
    for c in sorted(src_by_cls):
        pool = src_by_cls[c]
        src += rng.sample(pool, min(args.per_class, len(pool)))
    bg = []
    per_w = max(1, args.n_bg // max(1, len(bg_by_w)))
    for w in sorted(bg_by_w):
        bg += rng.sample(bg_by_w[w], min(per_w, len(bg_by_w[w])))
    rows = list(dict.fromkeys(src + bg))          # 去重、保序
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('# <seq_name> <frame.png> <n_obj> <n_qualified> <rmin> <rmax> <lidar_noise_std_m>\n')
        f.write('# paste-aug pool: %d source frames (day, vis>=%.0f, nearest<=%.0fm, <=%d/class) + %d background frames (day, %d/weather)\n'
                % (len(src), args.min_vis, args.src_max_range, args.per_class, len(bg), per_w))
        f.write('\n'.join(rows) + '\n')
    print('源帧 %d：%s' % (len(src), {c: min(args.per_class, len(v)) for c, v in sorted(src_by_cls.items())}))
    print('背景帧 %d：%s' % (len(bg), {w: min(per_w, len(v)) for w, v in sorted(bg_by_w.items())}))
    print('合并去重 %d 帧 -> %s' % (len(rows), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
