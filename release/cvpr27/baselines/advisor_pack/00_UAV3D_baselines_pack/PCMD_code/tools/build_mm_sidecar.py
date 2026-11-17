# -*- coding: utf-8 -*-
"""给 RGB-only 的 JPEG 缓存（build_mm_cache --store jpeg --rgb-only，如 @@MMCACHE@@/full_v2）补上其余模态的「边车」文件
（2026-09-23，导师方向：多模态纯仿真教师 → 单模态学生蒸馏）。

不重建缓存、不动 index.pkl / rgb_jpg.bin：按 index.pkl 的 metas 逐帧回原始数据取 IR / LiDAR，在【原图分辨率】上生成三张图，
与 RGB 同一像平面、同一 (seq, frame)、同一裁剪：
    ir     灰度 JPEG（q90）     images_ir/<frame>.png 转灰度；IR 相机与 RGB 若不同位，按最近合格目标深度处的视差整体平移
                                （同 build_mm_cache；这批 UE 室内数据 IR 与 RGB 同位，视差 = 0）
    depth  uint16 PNG，厘米     lidar_1/<frame>.npy 投到 RGB 相机，z-buffer 取最近。LiDAR 点在 1280x720 上只有 ~10% 覆盖
                                （256 线、每帧 12.9 万点），按最近点补密：倒数深度 max-dilate 5x5 -> ~77%（其余是超量程/无回波）。0 = 无点
    tag    uint8 PNG（0/1）     tag==1（打在无人机上）的 LiDAR 点，同样 5x5 膨胀
写成 <split>/{ir_jpg,depth_png,tag_png}.bin + *_index.npy（(N,2) 偏移/长度，长度 0 = 该帧无数据），与 rgb_jpg 同一套索引约定。
数据集 MODALITIES 含 ir/depth/tag 时才打开这些文件；只写 ['rgb'] 时行为与之前逐位相同。

    python build_mm_sidecar.py --cache @@MMCACHE@@/full_v2 --root @@DATA_COLLECT@@ --split train test --workers 8
    python build_mm_sidecar.py --cache @@MMCACHE@@/full_v2 --root @@DATA_COLLECT@@ --split test --limit 40 --out <scratch>   # 冒烟
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_mm_cache as B   # noqa: E402
from build_near_subset import frame_sort_key   # noqa: E402
from uavdet3d.utils import camera_geometry as cg   # noqa: E402

FILES = {'ir': 'ir_jpg', 'depth': 'depth_png', 'tag': 'tag_png'}
_FRAMES = {}      # 序列目录 -> 排好序的帧名（每个 worker 自己缓存）
_INFO = {}        # 序列目录 -> (im_info, lidar_radar_info)


def _seq_frames(base):
    if base not in _FRAMES:
        _FRAMES[base] = sorted([f for f in os.listdir(os.path.join(base, 'images_rgb')) if f.endswith('.png')],
                               key=frame_sort_key)
    return _FRAMES[base]


def _seq_info(base):
    if base not in _INFO:
        with open(os.path.join(base, 'im_info.pkl'), 'rb') as f:
            info = pickle.load(f)
        with open(os.path.join(base, 'lidar_radar_info.pkl'), 'rb') as f:
            lr = pickle.load(f)
        _INFO[base] = (info, lr)
    return _INFO[base]


def process(task):
    (root, i, seq, frame, K_raw, raw_wh, crop_xy, crop_wh, boxes, qual, legacy_fixed, lidar_offset, dil, ir_q) = task
    base = os.path.join(root, seq)
    try:
        info, lr = _seq_info(base)
        E_rgb = np.array(info['rgb']['extrinsic'], np.float64)
        K_ir = np.array(info['ir']['intrinsic'], np.float64)
        E_ir = np.array(info['ir']['extrinsic'], np.float64)
        L_ext = np.array(lr['lidars'][0]['extrinsic'], np.float64)
        ir = cv2.imread(os.path.join(base, 'images_ir', frame), cv2.IMREAD_GRAYSCALE)
        if ir is None:
            return i, None, None, None, 'no ir'
        if legacy_fixed:
            K_ir, _ = cg.legacy_sim_intrinsic_fix(K_ir, ir.shape[1], ir.shape[0])
        raw_w, raw_h = int(raw_wh[0]), int(raw_wh[1])
        K_raw = np.asarray(K_raw, np.float64).reshape(3, 3)
        cx0, cy0 = (int(crop_xy[0]), int(crop_xy[1])) if crop_xy is not None else (0, 0)
        K_full = K_raw.copy()                      # 缓存的 K_raw 已含裁剪平移；未裁剪原图上的 K 把平移加回去
        K_full[0, 2] += cx0
        K_full[1, 2] += cy0

        # ---- IR 对齐到 RGB：最近合格目标中心处的视差整体平移（同 build_mm_cache）----
        b = np.asarray(boxes, np.float64).reshape(-1, 9)
        q = np.asarray(qual, bool).reshape(-1) if qual is not None and len(qual) == len(b) else np.ones(len(b), bool)
        cand = b[q] if q.any() else b
        near = cand[np.argmin(cand[:, 2])]
        cw = B.cam_to_world(near[None, :3], E_rgb)
        uv_rgb, _ = B.world_to_pixel(cw, E_rgb, K_full)
        uv_ir, _ = B.world_to_pixel(cw, E_ir, K_ir)
        shift = uv_rgb[0] - uv_ir[0]
        if np.abs(shift).max() > 0.01:
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            ir = cv2.warpAffine(ir, M, ir.shape[1::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if crop_wh is not None:
            ir = ir[cy0:cy0 + raw_h, cx0:cx0 + raw_w]
        if ir.shape[:2] != (raw_h, raw_w):
            ir = cv2.resize(ir, (raw_w, raw_h), interpolation=cv2.INTER_AREA)

        # ---- LiDAR -> 深度 / tag（裁剪后的原图分辨率；K_raw 已含裁剪平移，投影后按 raw_wh 裁掉出界点）----
        frames = _seq_frames(base)
        li = min(frames.index(frame) + lidar_offset, len(frames) - 1)
        npy = os.path.join(base, 'lidar_1', os.path.splitext(frames[li])[0] + '.npy')
        depth = np.zeros((raw_h, raw_w), np.float32)
        tag = np.zeros((raw_h, raw_w), np.uint8)
        n_pts = 0
        if os.path.exists(npy):
            P = np.load(npy)
            xyz, t = P[:, :3].astype(np.float64), P[:, 4]
            world = (L_ext @ np.c_[xyz, np.ones(len(xyz))].T).T
            cv = (B.CARLA_TO_OPENCV @ (np.linalg.inv(E_rgb) @ world.T)).T[:, :3]
            m = cv[:, 2] > 0.1
            cv, t = cv[m], t[m]
            uv = (K_raw @ cv.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            ui = np.floor(uv[:, 0]).astype(int)
            vi = np.floor(uv[:, 1]).astype(int)
            ins = (ui >= 0) & (ui < raw_w) & (vi >= 0) & (vi < raw_h)
            ui, vi, z, t = ui[ins], vi[ins], cv[ins, 2], t[ins]
            n_pts = int(ins.sum())
            order = np.argsort(-z)                    # z-buffer：深度降序写入，近点最后覆盖
            depth[vi[order], ui[order]] = z[order]
            tag[vi[t == 1], ui[t == 1]] = 1
            if dil > 1:
                ker = np.ones((dil, dil), np.uint8)
                inv = np.where(depth > 0, 1.0 / np.maximum(depth, 1e-3), 0.0).astype(np.float32)
                inv = cv2.dilate(inv, ker)            # 倒数深度取 max = 最近点补洞
                depth = np.where(inv > 0, 1.0 / np.maximum(inv, 1e-6), 0.0).astype(np.float32)
                tag = cv2.dilate(tag, ker)
        d16 = np.where(depth > 0, np.clip(depth * 100.0, 1, 65535), 0).astype(np.uint16)
        ok1, b_ir = cv2.imencode('.jpg', ir, [cv2.IMWRITE_JPEG_QUALITY, int(ir_q)])
        ok2, b_d = cv2.imencode('.png', d16, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        ok3, b_t = cv2.imencode('.png', tag, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not (ok1 and ok2 and ok3):
            return i, None, None, None, 'encode'
        stat = (n_pts, float((depth > 0).mean()), int(tag.sum()), float(shift[0]), float(shift[1]))
        return i, b_ir.tobytes(), b_d.tobytes(), b_t.tobytes(), stat
    except Exception as e:      # noqa: BLE001
        return i, None, None, None, 'ERR ' + repr(e)


RADAR_VMAX = 40.0     # m/s；室内录制器雷达 |径向速度| 实测 6~36 m/s（静止背景恒为 0），255 = 40 m/s


def process_radar(task):
    """雷达 (N,5) = 速度, 方位, 俯仰, 距离, tag（雷达系，CARLA 约定）→ RGB 像平面（原图分辨率）的 |速度| 热图：
    逐像素取 max、7x7 高斯模糊（用户原 radar_to_velocity_heatmap 的做法，只是改在原分辨率上做）。tag 不参与渲染。"""
    (root, i, seq, frame, K_raw, raw_wh, radar_offset) = task
    base = os.path.join(root, seq)
    try:
        info, lr = _seq_info(base)
        E_rgb = np.array(info['rgb']['extrinsic'], np.float64)
        R_ext = np.array(lr['radars'][0]['extrinsic'], np.float64)
        raw_w, raw_h = int(raw_wh[0]), int(raw_wh[1])
        K = np.asarray(K_raw, np.float64).reshape(3, 3)
        frames = _seq_frames(base)
        ri = min(frames.index(frame) + radar_offset, len(frames) - 1)
        rp = os.path.join(base, 'radar_1', os.path.splitext(frames[ri])[0] + '.npy')
        hm = np.zeros((raw_h, raw_w), np.float32)
        n_in, n_mov = 0, 0
        if os.path.exists(rp):
            Q = np.load(rp)
            az, al, d = Q[:, 1], Q[:, 2], Q[:, 3]
            xyz = np.stack([d * np.cos(al) * np.cos(az), d * np.cos(al) * np.sin(az), d * np.sin(al)], 1)
            world = (R_ext @ np.c_[xyz, np.ones(len(xyz))].T).T
            cam = (B.CARLA_TO_OPENCV @ (np.linalg.inv(E_rgb) @ world.T)).T[:, :3]
            v = np.abs(Q[:, 0])
            m = cam[:, 2] > 0.1
            cam, v = cam[m], v[m]
            uv = (K @ cam.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            ui, vi = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
            ins = (ui >= 0) & (ui < raw_w) & (vi >= 0) & (vi < raw_h)
            ui, vi, v = ui[ins], vi[ins], v[ins]
            n_in, n_mov = int(ins.sum()), int((v > 0.1).sum())
            order = np.argsort(v)                               # 升序写入，同一像素留最大值
            hm[vi[order], ui[order]] = v[order]
            hm = cv2.GaussianBlur(hm, (7, 7), 0)
        q = np.clip(hm / RADAR_VMAX * 255.0 * 4.0, 0, 255).astype(np.uint8)   # x4：模糊把单点峰值摊到 ~1/4，补回来
        ok, buf = cv2.imencode('.png', q, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return i, None, 'encode'
        return i, buf.tobytes(), (n_in, n_mov, float((q > 0).mean()))
    except Exception as e:      # noqa: BLE001
        return i, None, 'ERR ' + repr(e)


def build_radar_split(cache, out, split, root, workers, radar_offset, limit):
    d_in, d_out = os.path.join(cache, split), os.path.join(out, split)
    os.makedirs(d_out, exist_ok=True)
    o = os.path.join(d_out, 'radar_png')
    if os.path.exists(o + '.bin') and os.path.exists(o + '_index.npy'):
        print('%s: 雷达边车已存在，跳过' % split, flush=True)
        return
    with open(os.path.join(d_in, 'index.pkl'), 'rb') as f:
        idx = pickle.load(f)
    metas = idx['metas']
    N = len(metas)
    tasks = [(root, i, m['seq'], m['frame'], m['K_raw'], m['raw_wh'], radar_offset) for i, m in enumerate(metas) if m]
    if limit:
        tasks = tasks[:limit]
    print('%s: 雷达 %d/%d 帧，%d 进程' % (split, len(tasks), N, workers), flush=True)
    index = np.zeros((N, 2), np.int64)
    pos, stats, fails = 0, [], []
    t0 = time.time()
    with open(o + '.bin.tmp', 'wb') as fh, mp.Pool(workers) as pool:
        for k, (i, b, st) in enumerate(pool.imap(process_radar, tasks, chunksize=8)):
            if b is None:
                fails.append((i, st))
                continue
            fh.write(b)
            index[i] = (pos, len(b))
            pos += len(b)
            stats.append(st)
            if (k + 1) % 2000 == 0 or k + 1 == len(tasks):
                print('  %d/%d  %.1f it/s  失败 %d' % (k + 1, len(tasks), (k + 1) / max(time.time() - t0, 1e-6), len(fails)), flush=True)
    os.replace(o + '.bin.tmp', o + '.bin')
    with open(o + '_index.npy.tmp', 'wb') as f:
        np.save(f, index)
    os.replace(o + '_index.npy.tmp', o + '_index.npy')
    S = np.array(stats, np.float64) if stats else np.zeros((0, 3))
    summ = {'split': split, 'n_tasks': len(tasks), 'n_ok': len(stats), 'n_fail': len(fails), 'fails': fails[:20],
            'radar_offset': radar_offset, 'vmax': RADAR_VMAX, 'bytes': int(pos), 'seconds': round(time.time() - t0, 1),
            'pts_in_image_median': float(np.median(S[:, 0])) if len(S) else None,
            'moving_pts_median': float(np.median(S[:, 1])) if len(S) else None,
            'nonzero_px_frac_median': float(np.median(S[:, 2])) if len(S) else None}
    with open(os.path.join(d_out, 'sidecar_radar.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'fails'}, ensure_ascii=False), flush=True)


def build_split(cache, out, split, root, workers, lidar_offset, dil, ir_q, limit):
    d_in = os.path.join(cache, split)
    d_out = os.path.join(out, split)
    os.makedirs(d_out, exist_ok=True)
    outs = {m: os.path.join(d_out, FILES[m]) for m in FILES}
    if all(os.path.exists(o + '.bin') and os.path.exists(o + '_index.npy') for o in outs.values()):
        print('%s: 边车已存在，跳过' % split, flush=True)
        return
    with open(os.path.join(d_in, 'index.pkl'), 'rb') as f:
        idx = pickle.load(f)
    assert idx.get('store') == 'jpeg', '只给 --store jpeg 的缓存补边车（npy 缓存本来就带全部模态）'
    metas = idx['metas']
    N = len(metas)
    tasks = []
    for i, m in enumerate(metas):
        if not m:
            continue
        tasks.append((root, i, m['seq'], m['frame'], m['K_raw'], m['raw_wh'], m.get('crop_xy'), m.get('crop_wh'),
                      m['boxes9d'], m.get('qualified'), bool(m.get('legacy_intrinsic_fixed', False)), lidar_offset, dil, ir_q))
        if limit and len(tasks) >= limit:
            break
    print('%s: %d/%d 帧，%d 进程' % (split, len(tasks), N, workers), flush=True)
    fh = {m: open(outs[m] + '.bin.tmp', 'wb') for m in FILES}
    index = {m: np.zeros((N, 2), np.int64) for m in FILES}
    pos = {m: 0 for m in FILES}
    n_ok, fails, stats = 0, [], []
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for k, (i, b_ir, b_d, b_t, st) in enumerate(pool.imap(process, tasks, chunksize=4)):
            if isinstance(st, tuple):
                for m, bb in (('ir', b_ir), ('depth', b_d), ('tag', b_t)):
                    fh[m].write(bb)
                    index[m][i] = (pos[m], len(bb))
                    pos[m] += len(bb)
                n_ok += 1
                stats.append(st)
            else:
                fails.append((i, st))
            if (k + 1) % 500 == 0 or k + 1 == len(tasks):
                el = time.time() - t0
                print('  %d/%d  %.1f it/s  写 %.2f GB  失败 %d' % (
                    k + 1, len(tasks), (k + 1) / max(el, 1e-6), sum(pos.values()) / 1e9, len(fails)), flush=True)
    for m in FILES:
        fh[m].close()
        os.replace(outs[m] + '.bin.tmp', outs[m] + '.bin')
        with open(outs[m] + '_index.npy.tmp', 'wb') as f:
            np.save(f, index[m])
        os.replace(outs[m] + '_index.npy.tmp', outs[m] + '_index.npy')
    S = np.array(stats, np.float64) if stats else np.zeros((0, 5))
    summary = {'split': split, 'n_frames': N, 'n_tasks': len(tasks), 'n_ok': n_ok, 'n_fail': len(fails),
               'fails': fails[:50], 'dilate': dil, 'ir_jpeg_quality': ir_q, 'lidar_offset': lidar_offset,
               'bytes': {m: int(pos[m]) for m in FILES}, 'seconds': round(time.time() - t0, 1),
               'lidar_pts_in_image_median': float(np.median(S[:, 0])) if len(S) else None,
               'depth_coverage_median': float(np.median(S[:, 1])) if len(S) else None,
               'tag_px_median': float(np.median(S[:, 2])) if len(S) else None,
               'tag_px_zero_frac': float((S[:, 2] == 0).mean()) if len(S) else None,
               'ir_shift_absmax': float(np.abs(S[:, 3:5]).max()) if len(S) else None}
    with open(os.path.join(d_out, 'sidecar.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'fails'}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='@@MMCACHE@@/full_v2')
    ap.add_argument('--out', default=None, help='边车写到哪（缺省 = 缓存目录本身）')
    ap.add_argument('--root', default='@@DATA_COLLECT@@')
    ap.add_argument('--split', nargs='+', default=['train', 'test'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lidar-offset', type=int, default=0, help='新录制器同步写盘，0；旧 CARLA 数据是 6')
    ap.add_argument('--dilate', type=int, default=5, help='深度/tag 补洞核（最近点），1 = 不补')
    ap.add_argument('--ir-quality', type=int, default=90)
    ap.add_argument('--limit', type=int, default=0, help='调试：只处理前 N 帧')
    ap.add_argument('--only', default=None, choices=[None, 'radar'], help='radar = 只建雷达速度热图边车')
    ap.add_argument('--radar-offset', type=int, default=0, help='新录制器同帧（09-28 实测偏移 0 时无人机回波全在真值框内）；旧 CARLA 是 8')
    a = ap.parse_args()
    for sp in a.split:
        if a.only == 'radar':
            build_radar_split(a.cache, a.out or a.cache, sp, a.root, a.workers, a.radar_offset, a.limit)
        else:
            build_split(a.cache, a.out or a.cache, sp, a.root, a.workers, a.lidar_offset, a.dilate, a.ir_quality, a.limit)


if __name__ == '__main__':
    main()
