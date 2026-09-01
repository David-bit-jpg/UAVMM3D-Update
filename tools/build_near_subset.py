# -*- coding: utf-8 -*-
"""从 UAV-MM3D 仿真数据里抽出【近距离】子集，作为与 MAV6D 尺度对齐的源域。

动机
----
源域整体是室外几十米量级（中位 42 m，MAX_DIS=150），MAV6D 是室内 ≤8 m。
`center_dis` 头学的是 Z / MAX_DIS，两边差 18.8 倍 —— 这是最大的一个域差。
把仿真数据里近距离的那部分单独抽出来，就能得到一个和 MAV6D 尺度接近、
但仍然带完整 6-DoF 标注的源域，迁移时不必再跨越这个数量级。

距离怎么来
----------
每个序列目录下都有 `distance_info.txt`，逐行是
    <目标名>: d0, d1, d2, ...
其长度与该序列的帧数严格一致（实测 299 == 299），按时间顺序排列。
读它比逐帧读 boxes_rgb/*.pkl 快几百倍（19 个/秒 -> 全量要 5 个多小时）。

它给的是**欧氏距离**，而 center_dis 学的是**相机系深度 Z**，且恒有 Z <= 欧氏距离。
所以按欧氏距离卡阈值是保守的：选出来的帧其 Z 一定也在阈值内。
用 --verify 可以再抽样读 pkl，核对真实 Z 的分布。

帧顺序
------
帧名是秒数时间戳（9.7944.png / 10.0612.png），**必须按数值排序**，
字典序会把 10.x 排到 9.x 前面；实测约 8% 的序列会因此错位。
这里和 laam6d_det_dataset._frame_sort_key 用的是同一套顺序。

用法
----
    # 扫描 + 统计分布（结果缓存，后续 build 秒出）
    python tools/build_near_subset.py scan --root E:/data_collect

    # 生成子集（默认 <=8m、整帧所有目标都在范围内、按序列切 train/test）
    python tools/build_near_subset.py build --root E:/data_collect \\
        --max-dist 8 --mode all --out-dir cfgs/subsets/near8

    # 核对选出来的帧真实的相机系深度
    python tools/build_near_subset.py build --root E:/data_collect --verify 300
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np


# --------------------------------------------------------------------------- #
def frame_sort_key(name):
    """与 laam6d_det_dataset._frame_sort_key 保持一致的数值排序键。"""
    stem = os.path.splitext(name)[0]
    try:
        return (0, float(stem), '')
    except ValueError:
        return (1, 0.0, stem)


def parse_distance_info(path):
    """-> {目标名: np.array([...])}"""
    out = {}
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            name, rest = line.split(':', 1)
            try:
                vals = [float(x) for x in rest.split(',') if x.strip()]
            except ValueError:
                continue
            if vals:
                out[name.strip()] = np.array(vals, dtype=np.float32)
    return out


def iter_seq_dirs(root):
    """产出 (seq_name, 绝对路径)，seq_name 与 dataset 里的写法一致。"""
    for m in sorted(os.listdir(root)):
        cr = os.path.join(root, m, 'carla_data')
        if not os.path.isdir(cr):
            continue
        for seq in sorted(os.listdir(cr)):
            sp = os.path.join(cr, seq)
            if not os.path.isdir(sp):
                continue
            for w in sorted(os.listdir(sp)):
                wp = os.path.join(sp, w)
                if not os.path.isdir(wp):
                    continue
                for d in sorted(os.listdir(wp)):
                    dp = os.path.join(wp, d)
                    if os.path.isdir(dp):
                        yield '%s/carla_data/%s/%s/%s' % (m, seq, w, d), dp


def box_center_z(pkl_path):
    """从 boxes_rgb 的 pkl 读相机系中心深度 Z（pkl 存的就是 OpenCV 相机系 8 角点）。"""
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    zs = []
    for row in raw:
        pts = np.array(row[1:] if isinstance(row[0], str) else row,
                       dtype=np.float32).reshape(8, 3)
        c = ((pts[0] + pts[6]) + (pts[1] + pts[7]) +
             (pts[2] + pts[4]) + (pts[3] + pts[5])) / 8.0
        zs.append(float(c[2]))
    return zs


# --------------------------------------------------------------------------- #
def scan(root, cache_path, quiet=False):
    """扫描全库，返回 {seq_name: {'frames': [...], 'dist': (n_obj, L) 数组}}。"""
    if cache_path and os.path.exists(cache_path):
        if not quiet:
            print('复用缓存: %s' % cache_path)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    table = {}
    n_seq = n_skip = 0
    for seq_name, dp in iter_seq_dirs(root):
        di = os.path.join(dp, 'distance_info.txt')
        im = os.path.join(dp, 'images_rgb')
        if not (os.path.exists(di) and os.path.isdir(im)):
            n_skip += 1
            continue
        frames = sorted([f for f in os.listdir(im) if f.endswith('.png')],
                        key=frame_sort_key)
        info = parse_distance_info(di)
        if not frames or not info:
            n_skip += 1
            continue
        L = min(len(frames), min(len(v) for v in info.values()))
        if L <= 0:
            n_skip += 1
            continue
        dist = np.stack([v[:L] for v in info.values()])   # (n_obj, L)
        table[seq_name] = {'frames': frames[:L],
                           'names': list(info.keys()),
                           'dist': dist}
        n_seq += 1
        if not quiet and n_seq % 200 == 0:
            print('  已扫描 %d 条序列...' % n_seq)

    if not quiet:
        print('扫描完成: %d 条序列可用, %d 条跳过' % (n_seq, n_skip))
    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(table, f)
        if not quiet:
            print('缓存写入: %s' % cache_path)
    return table


def report(table):
    all_d, frame_min, frame_max, n_objs = [], [], [], []
    for v in table.values():
        d = v['dist']
        all_d.append(d.reshape(-1))
        frame_min.append(d.min(axis=0))
        frame_max.append(d.max(axis=0))
        n_objs.append(np.full(d.shape[1], d.shape[0]))
    all_d = np.concatenate(all_d)
    frame_min = np.concatenate(frame_min)
    frame_max = np.concatenate(frame_max)
    n_objs = np.concatenate(n_objs)

    print('\n总帧数 %d，总目标样本 %d' % (len(frame_min), len(all_d)))
    print('单目标距离(欧氏)  min %.2f  中位 %.2f  max %.2f' %
          (all_d.min(), np.median(all_d), all_d.max()))
    print('每帧目标数分布    %s' %
          {int(k): int(v) for k, v in zip(*np.unique(n_objs, return_counts=True))})
    print('\n阈值      any(至少一个在范围内)      all(全部在范围内)')
    for t in [5, 8, 10, 15, 20, 25, 30]:
        n_any = int((frame_min <= t).sum())
        n_all = int((frame_max <= t).sum())
        print('  <=%-4dm  %8d 帧 (%5.2f%%)      %8d 帧 (%5.2f%%)'
              % (t, n_any, 100.0 * n_any / len(frame_min),
                 n_all, 100.0 * n_all / len(frame_min)))
    return frame_min, frame_max


def build(table, root, max_dist, mode, max_objects, train_ratio, seed,
          out_dir, verify, metric='z', prefilter_scale=1.5, min_z=0.5):
    # --- 第一阶段：用 distance_info 的欧氏距离粗筛（快） ---
    # 卡一个放宽的阈值，避免漏掉"深度不远但横向偏得多"的帧
    coarse_thresh = max_dist * (prefilter_scale if metric == 'z' else 1.0)
    picked = {}          # seq_name -> [(frame, n_obj, dmin, dmax)]
    for seq_name, v in table.items():
        d = v['dist']                       # (n_obj, L)
        if max_objects and d.shape[0] > max_objects:
            continue
        dmin = d.min(axis=0)
        dmax = d.max(axis=0)
        ok = (dmax <= coarse_thresh) if mode == 'all' else (dmin <= coarse_thresh)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        picked[seq_name] = [(v['frames'][i], d.shape[0],
                             float(dmin[i]), float(dmax[i])) for i in idx]

    n_coarse = sum(len(x) for x in picked.values())
    print('\n筛选条件: <=%.1f m, mode=%s, metric=%s%s' %
          (max_dist, mode, metric,
           ', 每帧目标数<=%d' % max_objects if max_objects else ''))
    print('粗筛(欧氏 <= %.1f m): %d 条序列, %d 帧' %
          (coarse_thresh, len(picked), n_coarse))

    # --- 第二阶段：读 boxes_rgb 按真实相机系深度 Z 精筛 ---
    # 欧氏距离小 != 在视野里：实测有 Z 为负（目标在相机后方）的帧，
    # 编码器遇到 Z<=0 会直接跳过，那样的帧就是一张空监督图，必须剔掉。
    if metric == 'z':
        print('精筛中（读 %d 个 boxes_rgb 求相机系 Z）...' % n_coarse)
        refined = {}
        n_read_fail = 0
        for seq_name, items in picked.items():
            keep = []
            for fr, nb, dmn, dmx in items:
                p = os.path.join(root, seq_name, 'boxes_rgb', fr.replace('.png', '.pkl'))
                if not os.path.exists(p):
                    n_read_fail += 1
                    continue
                try:
                    zs = box_center_z(p)
                except Exception:
                    n_read_fail += 1
                    continue
                if not zs:
                    continue
                zs = np.array(zs, dtype=np.float32)
                in_range = (zs >= min_z) & (zs <= max_dist)
                good = in_range.all() if mode == 'all' else in_range.any()
                if good:
                    keep.append((fr, nb, float(zs.min()), float(zs.max())))
            if keep:
                refined[seq_name] = keep
        picked = refined
        n_frames = sum(len(x) for x in picked.values())
        print('精筛后(%.1f m >= Z >= %.1f m): %d 条序列, %d 帧 (粗筛的 %.1f%%，读失败 %d)'
              % (max_dist, min_z, len(picked), n_frames,
                 100.0 * n_frames / max(n_coarse, 1), n_read_fail))
    else:
        n_frames = n_coarse
    print('最终: %d 条序列, %d 帧' % (len(picked), n_frames))
    if n_frames == 0:
        print('!! 没有帧满足条件，放宽 --max-dist 或改用 --mode any')
        return 1

    # 按序列切分，避免相邻帧泄漏
    seqs = sorted(picked.keys())
    rng = random.Random(seed)
    rng.shuffle(seqs)
    n_tr = max(1, int(round(len(seqs) * train_ratio)))
    train_seqs, test_seqs = set(seqs[:n_tr]), set(seqs[n_tr:])
    if not test_seqs:
        print('!! 只有 1 条序列，无法按序列切分')
        return 1

    os.makedirs(out_dir, exist_ok=True)
    meta = {'root': root, 'max_dist': max_dist, 'mode': mode, 'metric': metric,
            'min_z': min_z, 'max_objects': max_objects,
            'train_ratio': train_ratio, 'seed': seed}
    col = 'zmin> <zmax' if metric == 'z' else 'dmin> <dmax'
    for name, sel in [('train', train_seqs), ('test', test_seqs)]:
        p = os.path.join(out_dir, 'near_%s.txt' % name)
        n = 0
        with open(p, 'w') as f:
            f.write('# <seq_name> <frame.png> <n_obj> <%s>\n' % col)
            f.write('# %s\n' % json.dumps(meta, ensure_ascii=False))
            for s in sorted(sel):
                for fr, nb, dmn, dmx in picked[s]:
                    f.write('%s %s %d %.2f %.2f\n' % (s, fr, nb, dmn, dmx))
                    n += 1
        print('  %-5s %4d 条序列, %6d 帧 -> %s' % (name, len(sel), n, p))

    if verify:
        print('\n=== 核对真实相机系深度 Z（抽 %d 帧读 boxes_rgb）===' % verify)
        flat = [(s, fr) for s in picked for fr, _, _, _ in picked[s]]
        rng.shuffle(flat)
        zs = []
        bad = 0
        for s, fr in flat[:verify]:
            p = os.path.join(root, s, 'boxes_rgb', fr.replace('.png', '.pkl'))
            if not os.path.exists(p):
                bad += 1
                continue
            try:
                zs.extend(box_center_z(p))
            except Exception:
                bad += 1
        if zs:
            zs = np.array(zs)
            print('  目标样本 %d 个 (读失败 %d)' % (len(zs), bad))
            print('  Z: min %.2f  中位 %.2f  95%% %.2f  max %.2f'
                  % (zs.min(), np.median(zs), np.percentile(zs, 95), zs.max()))
            good = ((zs >= min_z) & (zs <= max_dist)).mean()
            print('  %.1f m >= Z >= %.1f m 的比例: %.1f%%' % (max_dist, min_z, 100 * good))
            if metric == 'z':
                print('  (已按 Z 精筛过，这里应当是 100%%；若出现负值说明筛漏了)')
    return 0


def main():
    ap = argparse.ArgumentParser(description='抽取 UAV-MM3D 近距离子集作为对齐源域')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--root', default='E:/data_collect')
        p.add_argument('--cache', default='near_subset_cache.pkl')

    p = sub.add_parser('scan', help='扫描并统计距离分布')
    common(p)

    p = sub.add_parser('build', help='生成近距离子集的 train/test 帧列表')
    common(p)
    p.add_argument('--max-dist', type=float, default=8.0,
                   help='距离阈值(米)，默认 8 与 MAV6D 的 MAX_DIS 一致')
    p.add_argument('--mode', choices=['all', 'any'], default='all',
                   help='all=整帧所有目标都在范围内(推荐，深度分布最接近 MAV6D)；'
                        'any=至少一个在范围内(帧更多，但画面里仍有远处目标)')
    p.add_argument('--max-objects', type=int, default=0,
                   help='只保留目标数 <= 该值的序列；MAV6D 每帧 1 个目标，设 1 最贴近')
    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='cfgs/subsets/near8')
    p.add_argument('--verify', type=int, default=0,
                   help='抽 N 帧读 boxes_rgb 核对真实相机系深度 Z')
    p.add_argument('--metric', choices=['z', 'euclid'], default='z',
                   help='z=两阶段(欧氏粗筛+真实相机系深度精筛，推荐)；'
                        'euclid=只用 distance_info 的欧氏距离(快，但会混进相机后方的目标)')
    p.add_argument('--prefilter-scale', type=float, default=1.5,
                   help='metric=z 时粗筛阈值相对 --max-dist 的放宽倍数')
    p.add_argument('--min-z', type=float, default=0.5,
                   help='相机系深度下限(米)。实测存在 Z 为负的帧(目标在相机后方)，'
                        '编码器遇到 Z<=0 会跳过，留着就是空监督')

    args = ap.parse_args()

    table = scan(args.root, args.cache)
    report(table)

    if args.cmd == 'scan':
        return 0
    return build(table, args.root, args.max_dist, args.mode, args.max_objects,
                 args.train_ratio, args.seed, args.out_dir, args.verify,
                 metric=args.metric, prefilter_scale=args.prefilter_scale,
                 min_z=args.min_z)


if __name__ == '__main__':
    sys.exit(main())
