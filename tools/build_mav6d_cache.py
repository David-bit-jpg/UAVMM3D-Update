# -*- coding: utf-8 -*-
"""把 MAV6D 打包成与仿真同格式的「相机归一化」缓存（camnorm-v1），训练 / 验证 / 测试都读它。

每帧处理（全部只依赖这台相机自己的标定，换任何带畸变的相机都是同一套）：
    1. 去畸变：cv2.remap 到针孔相机 K_pin（主点保持标定值、去掉斜切，焦距按「原图边界全部留在画面内」缩小，
       camera_geometry.undistorted_pinhole_K；MAV6D 为 0.932 倍），之后全链路按针孔处理
    2. 等比缩放到 --width x --height（1920x1080 -> 512x288，审查 P18：原来 512x256 不等比）
    3. K_in = K_pin 按 cv2.resize 像素中心约定精确换算到缓存分辨率，写进 meta
    4. 标签：read_truth_Rt（MAV 坐标系 -> 相机系），欧拉 'xyz'，尺寸用官方 phantom4 角点范围 0.34x0.34x0.23
       （两机型共用，审查 P13：尺寸误差在 MAV6D 上无意义，报告不引用）

划分：
    test  = 官方 test.txt（两机型全部帧，与旧评测同一批 4800 帧）
    val   = 从官方 train.txt 里【按序列】留出约 --val-frac 的帧（每个机型各自抽，固定种子），
            只用于选轮次；与 train / test 序列都不重叠（脚本末尾断言）
    train = 官方 train.txt 去掉 val 序列
帧按 机型/场景/序列/时间戳 排序，SAMPLED_INTERVAL 抽帧即序列内均匀抽样。

用法：
    python tools/build_mav6d_cache.py --root E:/MAV6D --out E:/mmcache/mav6d_cn
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.utils import camera_geometry as cg   # noqa: E402

CLASSES = ['mavic2', 'phantom4']
K_CALIB = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.1, 533.9717], [0.0, 0.0, 1.0]])
D_CALIB = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
RAW_WH = (1920, 1080)
OB_SIZE = (0.34, 0.34, 0.23)
CAMERA2VICON = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                         [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                         [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                         [0, 0, 0, 1]])


def read_truth_Rt(labpath):
    """与 uavdet3d/datasets/mav6d/mav6d_utils.read_truth_Rt 逐行一致（MAV 坐标系在相机系下的位姿）。"""
    with open(labpath, 'r') as f:
        contents = f.readlines()[0].strip().split(' ')
    pose = list(map(float, contents))
    uav_pose = pose[9:]
    vicon2uav = np.eye(4)
    vicon2uav[:3, :3] = R.from_quat([uav_pose[-4], uav_pose[-3], uav_pose[-2], uav_pose[-1]]).as_matrix()
    vicon2uav[:3, 3] = uav_pose[0:3]
    T = CAMERA2VICON.dot(vicon2uav)
    return T[:3, :3], T[:3, 3]


def frame_key(line):
    scene, seq, fname = line.strip().split('/')[-3:]
    stem = os.path.splitext(fname)[0]
    return scene, seq, (int(stem) if stem.isdigit() else stem), fname


def read_split(root, cls, split):
    rows = []
    with open(os.path.join(root, cls, 'split', split + '.txt')) as f:
        for ln in f:
            if ln.strip():
                scene, seq, k, fname = frame_key(ln)
                rows.append((cls, scene, seq, k, fname))
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    return rows


def pick_val_sequences(rows, frac, seed):
    """每个机型内按序列随机留出约 frac 的帧。返回 {(cls, scene, seq)}。"""
    val = set()
    rng = np.random.RandomState(seed)
    for cls in CLASSES:
        counts = {}
        for r in rows:
            if r[0] == cls:
                counts[(r[0], r[1], r[2])] = counts.get((r[0], r[1], r[2]), 0) + 1
        keys = sorted(counts)
        total = sum(counts.values())
        target = frac * total
        got = 0
        for i in rng.permutation(len(keys)):
            k = keys[i]
            if got >= 0.9 * target:
                break
            if got + counts[k] <= 1.25 * target:
                val.add(k)
                got += counts[k]
    return val


def process(task):
    root, cls, scene, seq, fname, W, H = task
    img = cv2.imread(os.path.join(root, cls, 'JPEGImages', scene, seq, fname), cv2.IMREAD_COLOR)
    if img is None or (img.shape[1], img.shape[0]) != RAW_WH:
        return None
    m1, m2, K_pin = cg.undistort_maps(K_CALIB, D_CALIB, RAW_WH)
    und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    small = cv2.resize(und, (W, H), interpolation=cv2.INTER_AREA)
    K_in = cg.scale_K(K_pin, W / float(RAW_WH[0]), H / float(RAW_WH[1]))

    Rm, t = read_truth_Rt(os.path.join(root, cls, 'labels', scene, seq, os.path.splitext(fname)[0] + '.txt'))
    eul = R.from_matrix(Rm).as_euler('xyz')
    box = np.concatenate([t, OB_SIZE, eul]).astype(np.float32)[None]
    uv = K_in @ t
    inside = bool(t[2] > 0 and 0 <= uv[0] / uv[2] < W and 0 <= uv[1] / uv[2] < H)
    # 原始（带畸变）图上中心是否在画面内：用于统计去畸变裁掉了多少边缘目标
    uvd, _ = cv2.projectPoints(t.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K_CALIB, D_CALIB)
    uvd = uvd.reshape(2)
    inside_raw = bool(0 <= uvd[0] < RAW_WH[0] and 0 <= uvd[1] < RAW_WH[1])
    meta = {
        'seq': '%s/%s/%s' % (cls, scene, seq), 'frame': fname, 'cls': cls,
        'K_in': K_in, 'K_raw': K_pin, 'raw_wh': RAW_WH, 'undistorted': True,
        'boxes9d': box, 'names': ['drone'], 'qualified': np.array([True]),
        'center_inside': inside, 'center_inside_raw': inside_raw,
    }
    return small, meta


def build_split(args, name, rows):
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    N, H, W = len(rows), args.height, args.width
    rgb_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'rgb.npy'), 'w+', np.uint8, (N, H, W, 3))
    metas = [None] * N
    tasks = [(args.root, c, sc, sq, fn, W, H) for (c, sc, sq, _k, fn) in rows]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap(process, tasks, chunksize=16)):
            if res is not None:
                rgb_mm[i], metas[i] = res
            if (i + 1) % 1000 == 0:
                el = time.time() - t0
                print('   [%s] %d/%d  %.1f 帧/s' % (name, i + 1, N, (i + 1) / el), flush=True)
    rgb_mm.flush()
    valid = np.array([i for i, m in enumerate(metas) if m is not None], np.int64)
    index = {'valid_idx': valid, 'metas': metas, 'H': H, 'W': W, 'classes': ['drone'], 'split': name,
             'root': args.root, 'format': 'camnorm-v1', 'undistorted': True,
             'val_frac': args.val_frac, 'seed': args.seed}
    with open(os.path.join(out_dir, 'index.pkl'), 'wb') as f:
        pickle.dump(index, f)
    ms = [m for m in metas if m]
    Z = np.array([m['boxes9d'][0, 2] for m in ms])
    ins = np.array([m['center_inside'] for m in ms])
    ins_raw = np.array([m['center_inside_raw'] for m in ms])
    seqs = sorted(set(m['seq'] for m in ms))
    print('[%s] %d/%d 帧有效, %d 个序列, 深度 中位 %.2f (%.2f~%.2f) m, 去畸变后中心在画面内 %.2f%% (原图 %.2f%%), %.1f 分钟'
          % (name, len(valid), N, len(seqs), np.median(Z), Z.min(), Z.max(), 100 * ins.mean(), 100 * ins_raw.mean(),
             (time.time() - t0) / 60), flush=True)
    return seqs


def compute_norm(out, n=600):
    a = np.load(os.path.join(out, 'train', 'rgb.npy'), mmap_mode='r')
    idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    s1 = np.zeros(3, np.float64)
    s2 = np.zeros(3, np.float64)
    cnt = 0
    for i in idx:
        v = a[i].astype(np.float64).reshape(-1, 3) / 255.0
        s1 += v.sum(0)
        s2 += (v * v).sum(0)
        cnt += v.shape[0]
    mean = s1 / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean * mean, 0))
    txt = 'NORM_MEAN: [%.4f, %.4f, %.4f]\nNORM_STD: [%.4f, %.4f, %.4f]\nframes %d\n' % (tuple(mean) + tuple(std) + (len(idx),))
    open(os.path.join(out, 'norm.txt'), 'w').write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='E:/MAV6D')
    ap.add_argument('--out', default='E:/mmcache/mav6d_cn')
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--height', type=int, default=288)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--splits', nargs='*', default=['train', 'val', 'test'])
    ap.add_argument('--limit', type=int, default=0, help='调试用：每个划分只打包前 N 帧（按均匀间隔抽）')
    ap.add_argument('--list-out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       'cfgs', 'subsets', 'mav6d_camnorm'))
    args = ap.parse_args()
    assert abs(args.width / args.height - RAW_WH[0] / RAW_WH[1]) < 1e-6, '必须等比缩放'

    train_all = sum([read_split(args.root, c, 'train') for c in CLASSES], [])
    test = sum([read_split(args.root, c, 'test') for c in CLASSES], [])
    val_keys = pick_val_sequences(train_all, args.val_frac, args.seed)
    val = [r for r in train_all if (r[0], r[1], r[2]) in val_keys]
    train = [r for r in train_all if (r[0], r[1], r[2]) not in val_keys]

    seq_of = lambda rows: set((r[0], r[1], r[2]) for r in rows)   # noqa: E731
    frm_of = lambda rows: set((r[0], r[1], r[2], r[4]) for r in rows)   # noqa: E731
    assert not (seq_of(train) & seq_of(val)), 'train/val 序列重叠'
    assert not (frm_of(train) & frm_of(test)) and not (frm_of(val) & frm_of(test)), 'train/val 与 test 帧重叠'
    for c in CLASSES:
        nt = sum(1 for r in train if r[0] == c)
        nv = sum(1 for r in val if r[0] == c)
        print('%-9s train %5d 帧 / %2d 序列   val %5d 帧 / %2d 序列 (%.1f%%)   test %5d 帧 / %d 序列'
              % (c, nt, len(set(k for k in seq_of(train) if k[0] == c)), nv,
                 len(set(k for k in seq_of(val) if k[0] == c)), 100.0 * nv / max(nt + nv, 1),
                 sum(1 for r in test if r[0] == c), len(set(k for k in seq_of(test) if k[0] == c))))
    seq_test_overlap = seq_of(test) & (seq_of(train) | seq_of(val))
    print('test 与 train/val 同名序列: %d（官方划分，帧不重叠已断言）' % len(seq_test_overlap))

    os.makedirs(args.list_out, exist_ok=True)
    with open(os.path.join(args.list_out, 'val_sequences.txt'), 'w') as f:
        f.write('# MAV6D 验证集序列（build_mav6d_cache.py --val-frac %g --seed %d），机型/场景/序列\n' % (args.val_frac, args.seed))
        for k in sorted(val_keys):
            f.write('%s/%s/%s\n' % k)

    for name, rows in (('train', train), ('val', val), ('test', test)):
        if name in args.splits:
            if args.limit:
                rows = [rows[i] for i in np.linspace(0, len(rows) - 1, args.limit).astype(int)]
            build_split(args, name, rows)
    if 'train' in args.splits:
        compute_norm(args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
