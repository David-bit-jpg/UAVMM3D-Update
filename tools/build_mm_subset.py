# -*- coding: utf-8 -*-
"""为多模态预训练筛源域帧：近距离 + 相机看得见框 + 三个模态文件齐全。

与 build_near_subset.py 的区别（那边是为了和 MAV6D 对齐深度分布，条件很苛刻）：
这里的目标是【保留仿真数据的丰富性】—— 多目标、8 种天气、5 个采集批次都留着，
只要求：一帧里【至少有一个】目标满足

    1) 到相机的距离 <= --max-range（默认 20 m，取框中心到相机的欧氏距离）
    2) 相机系深度 Z > 0.5 m（在相机前方）
    3) 框的 8 个角点里至少 --min-inside 个投影落在画面内（默认 8，即整个框可见）

并且这一帧的 rgb / ir / lidar 三个文件都存在。LiDAR 按 --lidar-offset 取
「后 6 帧」的点云（这是数据集的配对约定，实测 tag==1 的点云只有在偏移 6 时才
落在框中心上），所以每条序列末尾 6 帧没有配对点云，直接不要。

其它目标（更远的、出画的）照常留在框文件里当监督 —— 它们是真实目标，
不是噪声；训练时由 filter_box_outside 处理出画的框。

两阶段：先用 distance_info.txt 的欧氏距离粗筛（毫秒级），再多进程读 boxes_rgb
做投影精筛。全库 394,209 帧，any<=20m 的粗筛约 9.7 万帧，8 进程约 10 分钟。

用法：
    python tools/build_mm_subset.py --max-range 20 --out-dir cfgs/subsets/mm20 --workers 8
输出 near_train.txt / near_test.txt（列: seq frame n_obj n_qualified rmin rmax），
格式与 build_near_subset 一致，FRAME_SUBSET 可以直接吃。
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import pickle
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_near_subset import scan, frame_sort_key   # noqa: E402


def load_seq_meta(root, seq):
    """每条序列读一次：rgb 内参、图像宽高、LiDAR 测距噪声。

    LiDAR 的 NoiseStdDev 是【逐序列随机】的传感器参数，实测五个采集批次里都混着
    {0.0, 2.5, 3.5, 4.5} m 四档，约 3/4 的序列噪声 >= 2.5 m。这个值直接决定
    LiDAR 深度能不能当教师用，所以每帧都记下来，训练时可以按它筛或加权。
    """
    with open(os.path.join(root, seq, 'im_info.pkl'), 'rb') as f:
        info = pickle.load(f)
    K = np.array(info['rgb']['intrinsic'], dtype=np.float64)
    im_dir = os.path.join(root, seq, 'images_rgb')
    first = sorted(os.listdir(im_dir), key=frame_sort_key)[0]
    img = cv2.imread(os.path.join(im_dir, first), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    noise = -1.0
    p = os.path.join(root, seq, 'lidar_radar_info.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            lr = pickle.load(f)
        noise = float(lr['lidars'][0]['attributes'].get('NoiseStdDev', -1.0))
    return K, w, h, noise


def _worker(task):
    """(root, seq, frame, K, W, H, max_range, min_inside, min_z, lidar_frame) -> 行 或 None"""
    root, seq, fr, K, W, H, max_range, min_inside, min_z, lidar_frame = task
    stem = os.path.splitext(fr)[0]
    base = os.path.join(root, seq)
    box_p = os.path.join(base, 'boxes_rgb', stem + '.pkl')
    ir_p = os.path.join(base, 'images_ir', fr)
    ld_p = os.path.join(base, 'lidar_1', os.path.splitext(lidar_frame)[0] + '.npy')
    if not (os.path.exists(box_p) and os.path.exists(ir_p) and os.path.exists(ld_p)):
        return None
    try:
        with open(box_p, 'rb') as f:
            raw = pickle.load(f)
    except Exception:
        return None
    K = np.asarray(K, dtype=np.float64)
    n_obj = 0
    n_q = 0
    ranges = []
    for row in raw:
        pts = np.array(row[1:] if isinstance(row[0], str) else row,
                       dtype=np.float64).reshape(8, 3)
        n_obj += 1
        c = pts.mean(axis=0)
        z = float(c[2])
        r = float(np.linalg.norm(c))
        if z <= min_z or r > max_range:
            continue
        if np.any(pts[:, 2] <= 1e-6):
            continue
        uv = (K @ pts.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        inside = int(((uv[:, 0] >= 0) & (uv[:, 0] < W) &
                      (uv[:, 1] >= 0) & (uv[:, 1] < H)).sum())
        if inside >= min_inside:
            n_q += 1
            ranges.append(r)
    if n_q == 0:
        return None
    return (seq, fr, n_obj, n_q, float(min(ranges)), float(max(ranges)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='E:/data_collect')
    ap.add_argument('--cache', default='near_subset_cache.pkl',
                    help='build_near_subset.py 的全库扫描缓存，复用它省掉几分钟')
    ap.add_argument('--max-range', type=float, default=20.0,
                    help='框中心到相机的欧氏距离上限(米)')
    ap.add_argument('--min-inside', type=int, default=8,
                    help='8 个角点里至少几个投影在画面内才算「看得见」；8=整框可见')
    ap.add_argument('--min-z', type=float, default=0.5)
    ap.add_argument('--lidar-offset', type=int, default=6,
                    help='LiDAR 配对帧偏移，与 laam6d.yaml 的 LIDAR_OFFSET 一致')
    ap.add_argument('--prefilter-scale', type=float, default=1.15,
                    help='粗筛阈值 = max-range x 该系数；distance_info 与框中心距离略有差异')
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default='cfgs/subsets/mm20')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    table = scan(args.root, args.cache)
    coarse = args.max_range * args.prefilter_scale

    # ---- 粗筛：任一目标的欧氏距离 <= coarse，并且后面还有 lidar_offset 帧 ----
    tasks = []
    meta_cache = {}
    n_seq_hit = 0
    for seq, v in table.items():
        d = v['dist']                                   # (n_obj, L)
        frames = v['frames']
        L = len(frames)
        idx = np.nonzero(d.min(axis=0) <= coarse)[0]
        idx = idx[idx + args.lidar_offset < L]
        if len(idx) == 0:
            continue
        try:
            K, W, H, noise = load_seq_meta(args.root, seq)
        except Exception as e:
            print('  跳过 %s: %s' % (seq, e))
            continue
        meta_cache[seq] = (W, H, noise)
        n_seq_hit += 1
        for i in idx:
            tasks.append((args.root, seq, frames[i], K, W, H, args.max_range,
                          args.min_inside, args.min_z, frames[i + args.lidar_offset]))
    print('\n粗筛(任一目标欧氏距离 <= %.1f m，且有配对 LiDAR 帧): %d 条序列, %d 帧'
          % (coarse, n_seq_hit, len(tasks)))

    # ---- 精筛：投影 + 文件存在性，多进程 ----
    print('精筛中（%d 进程，读 boxes_rgb 做投影可见性判断）...' % args.workers)
    kept = collections.defaultdict(list)
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=128)):
            if r is not None:
                kept[r[0]].append(r[1:])
            if (i + 1) % 20000 == 0:
                print('   ...%d/%d' % (i + 1, len(tasks)))
    for seq in kept:
        kept[seq].sort(key=lambda x: frame_sort_key(x[0]))
    n_frames = sum(len(v) for v in kept.values())
    print('精筛后: %d 条序列, %d 帧 (粗筛的 %.1f%%)'
          % (len(kept), n_frames, 100.0 * n_frames / max(len(tasks), 1)))
    if n_frames == 0:
        print('!! 没有帧满足条件')
        return 1

    # ---- 统计：批次 / 天气 / 距离 / 每帧目标数 ----
    by_coll = collections.Counter()
    by_weather = collections.Counter()
    by_noise = collections.Counter()
    n_obj_hist = collections.Counter()
    rmins = []
    for seq, rows in kept.items():
        parts = seq.split('/')
        by_coll[parts[0]] += len(rows)
        by_weather[parts[3] if len(parts) > 3 else '?'] += len(rows)
        by_noise[meta_cache[seq][2]] += len(rows)
        for fr, n_obj, n_q, rmin, rmax in rows:
            n_obj_hist[n_obj] += 1
            rmins.append(rmin)
    rmins = np.array(rmins)
    print('\n按采集批次:', dict(sorted(by_coll.items())))
    print('按天气    :', dict(sorted(by_weather.items())))
    print('按 LiDAR 测距噪声 std(m):', dict(sorted(by_noise.items())))
    print('每帧目标数:', dict(sorted(n_obj_hist.items())))
    print('最近合格目标的距离(m): 中位 %.2f  5%% %.2f  95%% %.2f  最小 %.2f'
          % (np.median(rmins), np.percentile(rmins, 5), np.percentile(rmins, 95), rmins.min()))

    # ---- 按序列切分并写出 ----
    seqs = sorted(kept.keys())
    rng = random.Random(args.seed)
    rng.shuffle(seqs)
    n_tr = max(1, int(round(len(seqs) * args.train_ratio)))
    split = {'train': set(seqs[:n_tr]), 'test': set(seqs[n_tr:])}
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {'root': args.root, 'max_range': args.max_range, 'min_inside': args.min_inside,
            'min_z': args.min_z, 'lidar_offset': args.lidar_offset, 'mode': 'any',
            'modalities': ['rgb', 'ir', 'lidar'], 'train_ratio': args.train_ratio,
            'seed': args.seed}
    print()
    for name, sel in split.items():
        p = os.path.join(args.out_dir, 'near_%s.txt' % name)
        n = 0
        with open(p, 'w') as f:
            f.write('# <seq_name> <frame.png> <n_obj> <n_qualified> <rmin> <rmax> <lidar_noise_std_m>\n')
            f.write('# %s\n' % json.dumps(meta, ensure_ascii=False))
            for s in sorted(sel):
                noise = meta_cache[s][2]
                for fr, n_obj, n_q, rmin, rmax in kept[s]:
                    f.write('%s %s %d %d %.2f %.2f %.1f\n' % (s, fr, n_obj, n_q, rmin, rmax, noise))
                    n += 1
        print('  %-5s %4d 条序列, %6d 帧 -> %s' % (name, len(sel), n, p))
    return 0


if __name__ == '__main__':
    sys.exit(main())
