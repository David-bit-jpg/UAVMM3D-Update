# -*- coding: utf-8 -*-
"""把 mm20 子集打包成训练用的多模态缓存（放 SSD），训练时不再碰 HDD 上的 PNG/npy。

为什么要缓存：直接从 E: 盘读 rgb(1.5MB PNG)+ir(0.8MB PNG)+lidar(1.7MB npy)，
实测每帧 4 MB、5 次文件打开、两张 1280x720 PNG 解码，机械盘上只有 0.3 it/s，
30 轮要 27 小时。打包成 512x288 的 uint8/uint16 memmap 后训练变成 GPU 瓶颈。

每帧产出四个通道组（都在 RGB 相机的像平面上、同一分辨率）：
    rgb    (H,W,3) uint8
    ir     (H,W)   uint8    IR 相机与 RGB 相机相距约 2.4 m，用「最近合格目标深度处的视差」
                            整体平移到 RGB 视角（与仓库 register_images_by_center 同一思路，
                            但视差直接由几何算出）。只对齐那一个深度平面，其余深度有残差。
    depth  (H,W)   uint16   LiDAR 点投到 RGB 相机的深度，厘米，0 = 无点；z-buffer 取最近
    tag    (H,W)   uint8    该像素有 tag==1（打在无人机上）的 LiDAR 点

标签直接算成 9 参数 [x,y,z,l,w,h,a1,a2,a3]（RGB 相机 OpenCV 系，欧拉 'xyz'）。
旋转从 8 角点【正确地】解出：x=(c1-c0), y=(c3-c0), z=(c4-c0) 是三条互相正交的边
（实测两两夹角余弦 0.0000）—— 不用仓库里那对不互逆的 convert_9points_to_9params。

用法：
    python tools/build_mm_cache.py --list cfgs/subsets/mm20/near_train.txt --split train --every 3
    python tools/build_mm_cache.py --list cfgs/subsets/mm20/near_test.txt  --split test  --every 3
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_near_subset import frame_sort_key   # noqa: E402

CARLA_TO_OPENCV = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
OPENCV_TO_CARLA = np.linalg.inv(CARLA_TO_OPENCV)
CLASSES = ['DJI-avata2', 'DJI-phantom4', 'drone-unk3', 'm210-rtk',
           'DJI-mavic-mini', 'Matrice-600-Pro', 'matrix-300-RTK']


def read_list(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith('#'):
                p = ln.split()
                rows.append((p[0], p[1]))
    return rows


def class_of(name):
    for c in CLASSES:
        if c in name:
            return c
    return None


def corners_to_9params(c, seq='xyz'):
    """8 角点(相机系) -> [x,y,z,l,w,h,a1,a2,a3]。角点顺序为标准原型，c1-c0/c3-c0/c4-c0 是三条边。"""
    ex, ey, ez = c[1] - c[0], c[3] - c[0], c[4] - c[0]
    l, w, h = np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez)
    Rm = np.stack([ex / l, ey / w, ez / h], axis=1)
    if np.linalg.det(Rm) < 0:
        Rm[:, 2] *= -1
    # 数值上再正交化一次，防止 float32 角点带来的微小非正交
    u, _, vt = np.linalg.svd(Rm)
    Rm = u @ vt
    ang = R.from_matrix(Rm).as_euler(seq)
    center = c.mean(axis=0)
    return np.concatenate([center, [l, w, h], ang]).astype(np.float32)


def cam_to_world(pts_cv, E):
    """RGB 相机 OpenCV 系 -> CARLA 世界系（E 为相机外参 4x4）。"""
    P = np.c_[pts_cv, np.ones(len(pts_cv))]
    return (E @ (OPENCV_TO_CARLA @ P.T)).T[:, :3]


def world_to_pixel(pts_w, E, K):
    P = np.c_[pts_w, np.ones(len(pts_w))]
    cv = (CARLA_TO_OPENCV @ (np.linalg.inv(E) @ P.T)).T[:, :3]
    uv = (K @ cv.T).T
    return uv[:, :2] / uv[:, 2:3], cv[:, 2]


def process_frame(task):
    """一帧 -> dict(rgb, ir, depth, tag, meta) 或 None。"""
    (root, seq, frame, W, H, lidar_offset, max_label_range, max_range, min_inside, crop_wh, crop_seed) = task
    base = os.path.join(root, seq)
    stem = os.path.splitext(frame)[0]
    try:
        with open(os.path.join(base, 'im_info.pkl'), 'rb') as f:
            info = pickle.load(f)
        with open(os.path.join(base, 'lidar_radar_info.pkl'), 'rb') as f:
            lr = pickle.load(f)
    except Exception:
        return None
    K_rgb = np.array(info['rgb']['intrinsic'], dtype=np.float64)
    E_rgb = np.array(info['rgb']['extrinsic'], dtype=np.float64)
    K_ir = np.array(info['ir']['intrinsic'], dtype=np.float64)
    E_ir = np.array(info['ir']['extrinsic'], dtype=np.float64)
    L_ext = np.array(lr['lidars'][0]['extrinsic'], dtype=np.float64)
    noise = float(lr['lidars'][0]['attributes'].get('NoiseStdDev', -1.0))

    rgb = cv2.imread(os.path.join(base, 'images_rgb', frame), cv2.IMREAD_COLOR)
    ir = cv2.imread(os.path.join(base, 'images_ir', frame), cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        return None
    raw_h, raw_w = rgb.shape[:2]
    sx, sy = W / float(raw_w), H / float(raw_h)
    K_s = K_rgb.copy()
    K_s[0] *= sx
    K_s[1] *= sy

    # ---- 标签 ----
    try:
        with open(os.path.join(base, 'boxes_rgb', stem + '.pkl'), 'rb') as f:
            raw = pickle.load(f)
    except Exception:
        return None
    boxes, names, qual = [], [], []
    near_q = None
    for row in raw:
        name = row[0] if isinstance(row[0], str) else '?'
        cls = class_of(name)
        if cls is None:
            continue
        c = np.array(row[1:] if isinstance(row[0], str) else row, dtype=np.float64).reshape(8, 3)
        p9 = corners_to_9params(c)
        z, rng = float(p9[2]), float(np.linalg.norm(p9[:3]))
        if z <= 0.5 or rng > max_label_range:
            continue
        uv = (K_rgb @ c.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        inside = int(((uv[:, 0] >= 0) & (uv[:, 0] < raw_w) & (uv[:, 1] >= 0) & (uv[:, 1] < raw_h)).sum())
        cu = (K_rgb @ p9[:3])[:2] / p9[2]
        if not (0 <= cu[0] < raw_w and 0 <= cu[1] < raw_h):
            continue                    # 中心出画的框编码器放不下，丢掉
        q = bool(inside >= min_inside and rng <= max_range)
        boxes.append(p9)
        names.append(cls)
        qual.append(q)
        if q and (near_q is None or z < near_q[2]):
            near_q = p9
    if not boxes or near_q is None:
        return None
    boxes = np.stack(boxes)

    # ---- 可选：按 MAV6D 的归一化内参裁剪（尺度对齐）----
    # MAV6D fx/W = 1979.4/1920 = 1.031，仿真 640/1280 = 0.5 -> 裁到 621x349 再缩放，输入 fx 就和 MAV6D 一致，
    # 目标表观大小放大 1.65x。窗口位置随机（seed 由帧名决定，可复现），但必须包住最近的合格目标，
    # 避免「目标总在画面中心」的位置偏置。落在窗口外的框从标签里去掉（中心出窗）。
    crop_x0 = crop_y0 = 0
    if crop_wh is not None:
        cw, ch = crop_wh
        cu = (K_rgb @ near_q[:3].astype(np.float64))[:2] / near_q[2]
        rng = np.random.RandomState(crop_seed)
        lo_x, hi_x = int(max(0, cu[0] - cw + 8)), int(min(raw_w - cw, cu[0] - 8))
        lo_y, hi_y = int(max(0, cu[1] - ch + 8)), int(min(raw_h - ch, cu[1] - 8))
        if hi_x < lo_x or hi_y < lo_y:
            return None
        crop_x0 = int(rng.randint(lo_x, hi_x + 1))
        crop_y0 = int(rng.randint(lo_y, hi_y + 1))
        K_rgb = K_rgb.copy()
        K_rgb[0, 2] -= crop_x0
        K_rgb[1, 2] -= crop_y0
        rgb = rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
        keep = []
        for j, b in enumerate(boxes):
            cuj = (K_rgb @ b[:3].astype(np.float64))[:2] / b[2]
            ok = 0 <= cuj[0] < cw and 0 <= cuj[1] < ch
            if ok:
                # 重新判「整框可见」：8 角点是否都在窗口内
                cb = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                               [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]]) * b[3:6]
                cb = cb @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
                uvc = (K_rgb @ cb.T).T
                uvc = uvc[:, :2] / uvc[:, 2:3]
                inside_c = int(((uvc[:, 0] >= 0) & (uvc[:, 0] < cw) & (uvc[:, 1] >= 0) & (uvc[:, 1] < ch)).sum())
                qual[j] = bool(inside_c >= min_inside and np.linalg.norm(b[:3]) <= max_range)
            keep.append(ok)
        keep = np.array(keep, bool)
        boxes, names, qual = boxes[keep], [n for n, k in zip(names, keep) if k], [q for q, k in zip(qual, keep) if k]
        if len(boxes) == 0 or not any(qual):
            return None
        raw_w, raw_h = cw, ch
        sx, sy = W / float(raw_w), H / float(raw_h)
        K_s = K_rgb.copy()
        K_s[0] *= sx
        K_s[1] *= sy

    # ---- IR 对齐到 RGB：用最近合格目标中心处的视差整体平移 ----
    cw = cam_to_world(near_q[None, :3], E_rgb)
    uv_rgb, _ = world_to_pixel(cw, E_rgb, K_rgb)
    uv_ir, _ = world_to_pixel(cw, E_ir, K_ir)
    shift = (uv_rgb[0] - uv_ir[0])
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    ir_al = cv2.warpAffine(ir, M, ir.shape[1::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if crop_wh is not None:
        ir_al = ir_al[crop_y0:crop_y0 + raw_h, crop_x0:crop_x0 + raw_w]

    # ---- LiDAR 深度 / tag（直接投到缓存分辨率）----
    frames = sorted([f for f in os.listdir(os.path.join(base, 'images_rgb')) if f.endswith('.png')],
                    key=frame_sort_key)
    try:
        li = min(frames.index(frame) + lidar_offset, len(frames) - 1)
    except ValueError:
        return None
    npy = os.path.join(base, 'lidar_1', os.path.splitext(frames[li])[0] + '.npy')
    depth = np.zeros((H, W), np.uint16)
    tag = np.zeros((H, W), np.uint8)
    if os.path.exists(npy):
        P = np.load(npy)
        xyz = P[:, :3].astype(np.float64)
        t = P[:, 4]
        world = (L_ext @ np.c_[xyz, np.ones(len(xyz))].T).T
        cv = (CARLA_TO_OPENCV @ (np.linalg.inv(E_rgb) @ world.T)).T[:, :3]
        m = cv[:, 2] > 0.1
        cv, t = cv[m], t[m]
        uv = (K_s @ cv.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        ui = np.floor(uv[:, 0]).astype(int)
        vi = np.floor(uv[:, 1]).astype(int)
        ins = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, z, t = ui[ins], vi[ins], cv[ins, 2], t[ins]
        # z-buffer：按深度降序写入，近点最后覆盖
        order = np.argsort(-z)
        zc = np.clip(z * 100.0, 1, 65535).astype(np.uint16)
        depth[vi[order], ui[order]] = zc[order]
        tag[vi[t == 1], ui[t == 1]] = 1

    rgb_s = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    ir_s = cv2.resize(ir_al, (W, H), interpolation=cv2.INTER_AREA)

    meta = {
        'seq': seq, 'frame': frame, 'K_raw': K_rgb.astype(np.float32), 'raw_wh': (raw_w, raw_h),
        'boxes9d': boxes.astype(np.float32), 'names': names, 'qualified': np.array(qual, bool),
        'lidar_noise': noise, 'ir_shift': shift.astype(np.float32), 'lidar_frame': frames[li],
        'crop_xy': (crop_x0, crop_y0), 'crop_wh': crop_wh,
    }
    return rgb_s, ir_s, depth, tag, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--split', required=True, choices=['train', 'test'])
    ap.add_argument('--root', default='E:/data_collect')
    ap.add_argument('--out', default='C:/mmcache/mm20', help='缓存根目录（放 SSD）')
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--height', type=int, default=288)
    ap.add_argument('--every', type=int, default=3, help='每条序列每 N 帧取 1 帧（相邻帧几乎重复）')
    ap.add_argument('--lidar-offset', type=int, default=6)
    ap.add_argument('--max-range', type=float, default=20.0, help='「合格」目标的距离上限，与筛选一致')
    ap.add_argument('--min-inside', type=int, default=8)
    ap.add_argument('--max-label-range', type=float, default=40.0,
                    help='更远的框不进标签（归一化深度会超 1，且只有几个像素大）')
    ap.add_argument('--crop-to-mav6d', action='store_true',
                    help='按 MAV6D 归一化内参裁剪 621x349 再缩放，目标表观大小放大 1.65x，输入 fx 与 MAV6D 一致')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='调试用：只处理前 N 帧')
    args = ap.parse_args()

    rows = read_list(args.list)
    # 按序列分组后等间隔抽帧
    by_seq = {}
    for seq, fr in rows:
        by_seq.setdefault(seq, []).append(fr)
    picked = []
    for seq, frs in by_seq.items():
        frs = sorted(frs, key=frame_sort_key)
        picked += [(seq, f) for f in frs[::args.every]]
    if args.limit:
        picked = picked[:args.limit]
    print('列表 %d 帧 / %d 序列 -> 每 %d 帧取 1 -> %d 帧' % (len(rows), len(by_seq), args.every, len(picked)))

    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    N, H, W = len(picked), args.height, args.width
    rgb_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'rgb.npy'), 'w+', np.uint8, (N, H, W, 3))
    ir_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'ir.npy'), 'w+', np.uint8, (N, H, W))
    dep_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'depth.npy'), 'w+', np.uint16, (N, H, W))
    tag_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'tag.npy'), 'w+', np.uint8, (N, H, W))

    crop_wh = None
    if args.crop_to_mav6d:
        # 1280 * (640/1280) / (1979.4/1920) = 621 ；高按 16:9
        crop_wh = (621, 349)
        print('裁剪到 %dx%d（归一化 fx 与 MAV6D 一致）' % crop_wh)
    tasks = [(args.root, s, f, W, H, args.lidar_offset, args.max_label_range, args.max_range, args.min_inside,
              crop_wh, abs(hash(s + f)) % (2 ** 31))
             for s, f in picked]
    metas = [None] * N
    ok = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap(process_frame, tasks, chunksize=8)):
            if res is None:
                continue
            rgb_mm[i], ir_mm[i], dep_mm[i], tag_mm[i], metas[i] = res
            ok += 1
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print('   %d/%d  有效 %d  %.1f 帧/s  剩余 %.0f 分钟' %
                      (i + 1, N, ok, (i + 1) / el, (N - i - 1) / max((i + 1) / el, 1e-6) / 60))
    for m in (rgb_mm, ir_mm, dep_mm, tag_mm):
        m.flush()

    valid = [i for i, m in enumerate(metas) if m is not None]
    index = {
        'valid_idx': np.array(valid, np.int64), 'metas': metas, 'H': H, 'W': W,
        'classes': CLASSES, 'split': args.split, 'every': args.every,
        'lidar_offset': args.lidar_offset, 'max_range': args.max_range,
        'max_label_range': args.max_label_range, 'root': args.root, 'list': args.list,
        'crop_wh': crop_wh,
    }
    with open(os.path.join(out_dir, 'index.pkl'), 'wb') as f:
        pickle.dump(index, f)
    nq = sum(int(m['qualified'].sum()) for m in metas if m)
    nb = sum(len(m['boxes9d']) for m in metas if m)
    noise = np.array([m['lidar_noise'] for m in metas if m])
    print('\n完成: %d/%d 帧有效, 框 %d 个(其中合格 %d), 用时 %.1f 分钟 -> %s'
          % (len(valid), N, nb, nq, (time.time() - t0) / 60, out_dir))
    print('LiDAR 噪声分布(帧):', {float(v): int((noise == v).sum()) for v in np.unique(noise)})
    return 0


if __name__ == '__main__':
    sys.exit(main())
