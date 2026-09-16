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
import zlib

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_near_subset import frame_sort_key   # noqa: E402
from uavdet3d.utils import camera_geometry as cg   # noqa: E402

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
    """8 角点(相机系) -> [x,y,z,l,w,h,a1,a2,a3]。角点顺序为标准原型，c1-c0/c3-c0/c4-c0 是三条边。

    仿真角点来自 UE（左手系：X 前、Y 右、Z 上），三条边 100% 构成左手系，必须翻一个轴才是真旋转。
    翻【宽度轴】：得到 X 前、Y 左、Z 上的右手机体系，与 MAV6D 的 MAV 坐标系一致（机体 z 朝上）。
    审查 P4：原来翻的是高度轴，机体 z 朝下，两域旋转标签差 180°，零样本角度 157°。
    """
    ex, ey, ez = c[1] - c[0], c[3] - c[0], c[4] - c[0]
    l, w, h = np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez)
    Rm = np.stack([ex / l, ey / w, ez / h], axis=1)
    if np.linalg.det(Rm) < 0:
        Rm[:, 1] *= -1
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
    (root, seq, frame, W, H, lidar_offset, max_label_range, max_range, min_inside, crop_wh, crop_seed, zoom_cfg,
     rgb_only, intrinsic_mode, store, jpeg_quality) = task
    base = os.path.join(root, seq)
    stem = os.path.splitext(frame)[0]
    try:
        with open(os.path.join(base, 'im_info.pkl'), 'rb') as f:
            info = pickle.load(f)
        lr = None
        if not rgb_only:
            with open(os.path.join(base, 'lidar_radar_info.pkl'), 'rb') as f:
                lr = pickle.load(f)
    except Exception:
        return None
    K_rgb = np.array(info['rgb']['intrinsic'], dtype=np.float64)
    E_rgb = np.array(info['rgb']['extrinsic'], dtype=np.float64)

    rgb = cv2.imread(os.path.join(base, 'images_rgb', frame), cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    raw_h, raw_w = rgb.shape[:2]
    # 审查 P7：录制器写的是扰动过的假内参（fy=fx*1.01, cx=W/2+2, cy=H/2-1.5），识别出来就换回真渲染内参
    legacy_fixed = False
    if intrinsic_mode == 'auto':
        K_rgb, legacy_fixed = cg.legacy_sim_intrinsic_fix(K_rgb, raw_w, raw_h)
    ir = None
    if not rgb_only:
        K_ir = np.array(info['ir']['intrinsic'], dtype=np.float64)
        E_ir = np.array(info['ir']['extrinsic'], dtype=np.float64)
        L_ext = np.array(lr['lidars'][0]['extrinsic'], dtype=np.float64)
        noise = float(lr['lidars'][0]['attributes'].get('NoiseStdDev', -1.0))
        ir = cv2.imread(os.path.join(base, 'images_ir', frame), cv2.IMREAD_GRAYSCALE)
        if ir is None:
            return None
        if intrinsic_mode == 'auto':
            K_ir, _ = cg.legacy_sim_intrinsic_fix(K_ir, ir.shape[1], ir.shape[0])
    else:
        noise = -1.0
    K_s = cg.scale_K(K_rgb, W / float(raw_w), H / float(raw_h))

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
        if zoom_cfg is not None:
            # 变焦裁剪：把最近合格目标的表观大小拉到 MAV6D 的区间 [px_min, px_max]（512 宽的缓存里的像素）。
            # 标准窗口 621x349 下 fx_in = 527.7；窗口缩小 z 倍 -> fx_in = 527.7 z，目标放大 z 倍（原图 1280 像素上采样，
            # z 越大越糊，所以封顶 zoom_max）。已经够大的目标随机取 z in [1, min(zoom_max, px_max/px0)]，增加尺寸多样性。
            zmax, px_min, px_max, ref = zoom_cfg
            # 口径与 MAV6D 统计一致：8 角点投影 2D 框的长边（在标准 621 窗口缩到 W 宽时的像素）
            cb0 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                            [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]]) * near_q[3:6]
            cb0 = cb0 @ R.from_euler('xyz', near_q[6:9]).as_matrix().T + near_q[:3]
            uv0 = (K_rgb @ cb0.T).T
            uv0 = uv0[:, :2] / uv0[:, 2:3]
            px0 = float(max(uv0[:, 0].max() - uv0[:, 0].min(), uv0[:, 1].max() - uv0[:, 1].min())) * W / float(cw)
            if ref is not None:
                # 目标尺寸直接从 MAV6D 经验分布抽（严格同分布）；只能放大不能缩小（窗口不能比整幅图大），
                # 抽到比当前还小的尺寸就保持 z=1（这类帧本来就落在 MAV6D 区间里），需要的放大超过 zmax 则丢弃
                target = float(rng.choice(ref))
                z = max(1.0, target / max(px0, 1e-3))
                if z > zmax:
                    return None
            elif px0 >= px_min:
                z = float(rng.uniform(1.0, max(1.0, min(zmax, px_max / px0))))
            else:
                z = px_min / px0
                if z > zmax:
                    return None
            cw, ch = int(round(cw / z)), int(round(ch / z))
            crop_wh = (cw, ch)
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
        K_s = cg.scale_K(K_rgb, W / float(raw_w), H / float(raw_h))

    if store == 'jpeg':
        # 原分辨率 JPEG 字节（不缩放）：数据集在线裁不同大小的窗口模拟不同焦距（见 mmcache_det_dataset 的 VIEW_AUG）
        ok_enc, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok_enc:
            return None
        rgb_s = buf.tobytes()
    else:
        rgb_s = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    meta = {
        'seq': seq, 'frame': frame, 'K_raw': K_rgb.astype(np.float64), 'raw_wh': (raw_w, raw_h),
        # K_in = 缓存分辨率上的真针孔内参（cv2.resize 像素中心约定精确换算），数据集直接用它
        'K_in': K_s.astype(np.float64), 'legacy_intrinsic_fixed': bool(legacy_fixed),
        'boxes9d': boxes.astype(np.float32), 'names': names, 'qualified': np.array(qual, bool),
        'lidar_noise': noise, 'crop_xy': (crop_x0, crop_y0), 'crop_wh': crop_wh,
    }
    if rgb_only:
        return rgb_s, None, None, None, meta

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

    ir_s = cv2.resize(ir_al, (W, H), interpolation=cv2.INTER_AREA)
    meta['ir_shift'] = shift.astype(np.float32)
    meta['lidar_frame'] = frames[li]
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
    # 2026-09-16 实测：新录制器（UavSim，同步逐帧写盘）的 LiDAR / Radar 与 RGB 同帧，偏移 0 最好
    #   —— 三个组合上扫 0~12 帧，tag=1 的点到最近框心的中位距离 0 帧 0.08 m，逐帧变差到 12 帧 0.7 m。
    # 旧 CARLA 时代的数据是 6（雷达 8），那是后台线程池异步写盘造成的滞后。采旧数据的缓存要显式传 --lidar-offset 6。
    ap.add_argument('--lidar-offset', type=int, default=0)
    ap.add_argument('--max-range', type=float, default=20.0, help='「合格」目标的距离上限，与筛选一致')
    ap.add_argument('--min-inside', type=int, default=8)
    ap.add_argument('--max-label-range', type=float, default=40.0,
                    help='更远的框不进标签（归一化深度会超 1，且只有几个像素大）')
    ap.add_argument('--crop-to-mav6d', action='store_true',
                    help='按 MAV6D 归一化内参裁剪 621x349 再缩放，目标表观大小放大 1.65x，输入 fx 与 MAV6D 一致')
    ap.add_argument('--zoom-to-mav6d', action='store_true',
                    help='在 --crop-to-mav6d 基础上按最近合格目标变焦：窗口缩小到目标表观大小落进 MAV6D 区间')
    ap.add_argument('--zoom-max', type=float, default=1.6, help='最大变焦倍数（原图上采样上限）')
    ap.add_argument('--zoom-px', default='38,105', help='目标表观大小区间（缓存像素），MAV6D 实测 5%%/95%% 分位')
    ap.add_argument('--zoom-ref', default='', help='MAV6D 表观大小经验分布 .npy（tools/mav6d_size_stats.py）；给了就从里面抽目标尺寸')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='调试用：只处理前 N 帧')
    ap.add_argument('--rgb-only', action='store_true', help='只打包 RGB（不读 IR / LiDAR，不写 ir/depth/tag.npy）')
    ap.add_argument('--store', default='npy', choices=['npy', 'jpeg'],
                    help='npy = 缩到 --width x --height 的 memmap；jpeg = 原分辨率 JPEG 字节（rgb_jpg.bin + rgb_jpg_index.npy），'
                         '供在线多焦距裁窗（--width/--height 忽略，K_in 为原图真内参）')
    ap.add_argument('--jpeg-quality', type=int, default=95)
    ap.add_argument('--allow-unfixed-nose', action='store_true',
                    help='允许读取没有 label_fix_nose_x.json 标记的序列（仅当确认原始标注已是机头 +x）')
    ap.add_argument('--intrinsic', default='auto', choices=['auto', 'stored'],
                    help='auto = 识别录制器的扰动假内参并换回真内参（审查 P7）；stored = 原样用 im_info 里的值')
    args = ap.parse_args()

    rows = read_list(args.list)
    # 按序列分组后等间隔抽帧
    by_seq = {}
    for seq, fr in rows:
        by_seq.setdefault(seq, []).append(fr)
    # 原始标注的机头修正标记（tools/fix_raw_nose_labels.py，2026-09-15）：Drone Pack 网格机头朝 actor +Y，
    # 未修正的序列里 c1-c0 不是机头。新旧数据已全部就地修正；没有标记的序列（比如以后用旧资产新采的）直接拒绝，
    # 免得混进两种约定。确认数据本身已经是机头 +x（例如资产改过朝向后采集）时用 --allow-unfixed-nose。
    unfixed = [s for s in by_seq if not os.path.exists(os.path.join(args.root, s, 'label_fix_nose_x.json'))]
    if unfixed and not args.allow_unfixed_nose:
        raise SystemExit('有 %d 个序列没有机头修正标记 label_fix_nose_x.json（例：%s）。先跑 tools/fix_raw_nose_labels.py，'
                         '或确认数据已是机头 +x 后加 --allow-unfixed-nose' % (len(unfixed), unfixed[0]))
    raw_label_fix = 'nose-x-v1-20260915' if not unfixed else ('mixed-or-native' if len(unfixed) < len(by_seq) else 'native(--allow-unfixed-nose)')
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
    if args.store == 'jpeg':
        # 原分辨率：从第一个可读帧取尺寸，任务里的 W/H 等于原图，process_frame 里 K_s = 原图真内参
        for s_, f_ in picked:
            im0 = cv2.imread(os.path.join(args.root, s_, 'images_rgb', f_), cv2.IMREAD_COLOR)
            if im0 is not None:
                H, W = im0.shape[:2]
                break
        print('原分辨率 JPEG 存储：%dx%d，质量 %d' % (W, H, args.jpeg_quality))
    if args.store == 'jpeg':
        assert args.rgb_only, '--store jpeg 只支持 --rgb-only'
        jpg_fh = open(os.path.join(out_dir, 'rgb_jpg.bin'), 'wb')
        jpg_index = np.zeros((N, 2), np.int64)          # (偏移, 长度)，长度 0 = 无效帧
        jpg_pos = 0
        rgb_mm = None
    else:
        rgb_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'rgb.npy'), 'w+', np.uint8, (N, H, W, 3))
    if not args.rgb_only:
        ir_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'ir.npy'), 'w+', np.uint8, (N, H, W))
        dep_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'depth.npy'), 'w+', np.uint16, (N, H, W))
        tag_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'tag.npy'), 'w+', np.uint8, (N, H, W))

    crop_wh = None
    zoom_cfg = None
    if args.crop_to_mav6d or args.zoom_to_mav6d:
        # 1280 * (640/1280) / (1979.4/1920) = 621 ；高按 16:9
        crop_wh = (621, 349)
        print('裁剪到 %dx%d（归一化 fx 与 MAV6D 一致）' % crop_wh)
    if args.zoom_to_mav6d:
        px_min, px_max = [float(v) for v in args.zoom_px.split(',')]
        ref = np.load(args.zoom_ref).astype(np.float64) if args.zoom_ref else None
        zoom_cfg = (args.zoom_max, px_min, px_max, ref)
        if ref is not None:
            print('变焦：目标表观大小从 %s 抽（%d 个，p05/p50/p95 = %.0f/%.0f/%.0f px），最大 %.2fx'
                  % (args.zoom_ref, len(ref), *np.percentile(ref, [5, 50, 95]), args.zoom_max))
        else:
            print('变焦：目标表观大小 -> [%.0f, %.0f] px，最大 %.2fx' % (px_min, px_max, args.zoom_max))
    # 裁剪窗口的随机种子由 序列名+帧名 决定（zlib.crc32 跨进程稳定；内置 hash() 每个进程随机加盐，不可复现）
    tasks = [(args.root, s, f, W, H, args.lidar_offset, args.max_label_range, args.max_range, args.min_inside,
              crop_wh, zlib.crc32((s + f).encode('utf-8')) % (2 ** 31), zoom_cfg, args.rgb_only, args.intrinsic,
              args.store, args.jpeg_quality)
             for s, f in picked]
    metas = [None] * N
    ok = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap(process_frame, tasks, chunksize=8)):
            if res is None:
                continue
            if args.rgb_only:
                if args.store == 'jpeg':
                    jpg_fh.write(res[0])
                    jpg_index[i] = (jpg_pos, len(res[0]))
                    jpg_pos += len(res[0])
                    metas[i] = res[4]
                else:
                    rgb_mm[i], metas[i] = res[0], res[4]
            else:
                rgb_mm[i], ir_mm[i], dep_mm[i], tag_mm[i], metas[i] = res
            ok += 1
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print('   %d/%d  有效 %d  %.1f 帧/s  剩余 %.0f 分钟' %
                      (i + 1, N, ok, (i + 1) / el, (N - i - 1) / max((i + 1) / el, 1e-6) / 60), flush=True)
    if args.store == 'jpeg':
        jpg_fh.close()
        np.save(os.path.join(out_dir, 'rgb_jpg_index.npy'), jpg_index)
        print('JPEG 字节 %.2f GB' % (jpg_pos / 1e9))
    else:
        for m in ([rgb_mm] if args.rgb_only else [rgb_mm, ir_mm, dep_mm, tag_mm]):
            m.flush()

    valid = [i for i, m in enumerate(metas) if m is not None]
    index = {
        'valid_idx': np.array(valid, np.int64), 'metas': metas, 'H': H, 'W': W,
        'classes': CLASSES, 'split': args.split, 'every': args.every,
        'lidar_offset': args.lidar_offset, 'max_range': args.max_range,
        'max_label_range': args.max_label_range, 'root': args.root, 'list': args.list,
        'crop_wh': crop_wh, 'rgb_only': bool(args.rgb_only), 'intrinsic_mode': args.intrinsic,
        # camnorm-v1：meta 带 K_in（缓存分辨率真针孔内参）、角点左手系翻宽度轴（机体 z 朝上）、欧拉 'xyz'
        'format': 'camnorm-v1', 'store': args.store,
        # 原始标注机头修正状态：'nose-x-v1-20260915' = 读的是已就地修正（机头 +x）的原始数据
        'raw_label_fix': raw_label_fix,
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
