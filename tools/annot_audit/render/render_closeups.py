# -*- coding: utf-8 -*-
"""annot_audit/render：高清近景拼图，供人工 / 视觉评审「红色 x 轴是否指向无人机的视觉机头（云台相机 / 机头灯一侧）」。

仿真：7 个机型各一张 3x2 拼图（6 个近景，来自 6 条不同序列，目标距离 < 3.5 m，视角尽量覆盖 上方/前/侧/后/下方），
      从 D:/data_collect 原始 1280x720 PNG 裁出（以框中心投影为中心，裁边 = 1.6 倍框投影范围，放大到 480x480），
      叠加 3D 框（黄）与机体轴 红x 绿y 蓝z（箭头长 = 0.7 倍框长边），真内参 fx=fy=640 cx=639.5 cy=359.5；
      若同序列存在下一帧，再叠加品红色的飞行速度方向箭头（同名目标的框中心位移，相机静止）。
      标签 9 参数用 build_mm_cache.corners_to_9params 从原始 8 角点现算（与缓存一致，脚本内断言）。
MAV6D：phantom4 / mavic2 各一张 3x2 拼图，从 E:/MAV6D 原始带畸变 JPEG 裁，cv2.projectPoints 用官方 K / dist；
      位姿用 build_mav6d_cache.read_truth_Rt（MAV 机体系 -> 相机系），轴画在 VICON/MAV 原点 t 上；
      黄框 = 官方 phantom4 角点范围（x[-0.18,0.16] y[-0.16,0.18] z[-0.17,0.06]，两机型共用）。

用法（cd tools，PYTHONPATH=E:/Open3DUAVDet）：
    python annot_audit/render/render_closeups.py --stats        # 只打印候选统计
    python annot_audit/render/render_closeups.py                # 出图 -> output/camnorm/annot_audit/render/<domain>_<model>.jpg
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.dirname(TOOLS))
from build_mm_cache import corners_to_9params, class_of, CLASSES   # noqa: E402
from build_near_subset import frame_sort_key   # noqa: E402
from build_mav6d_cache import read_truth_Rt, K_CALIB, D_CALIB, OB_SIZE   # noqa: E402

SIM_ROOT = 'D:/data_collect'
SIM_CACHE = 'E:/mmcache/indoor8cn'
MAV_ROOT = 'E:/MAV6D'
MAV_CACHE = 'E:/mmcache/mav6d_cn'
OUT_DIR = 'E:/Open3DUAVDet/output/camnorm/annot_audit/render'
K_SIM = np.array([[640.0, 0.0, 639.5], [0.0, 640.0, 359.5], [0.0, 0.0, 1.0]])
D_SIM = np.zeros(5)
# 官方 phantom4 角点范围（E:/MAV6D/util.py 52-68 行）：框几何中心相对 VICON 原点的机体系偏移
MAV_BOX_OFF = np.array([(-0.18 + 0.16) / 2, (-0.16 + 0.18) / 2, (-0.17 + 0.06) / 2])

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
FRONT_FACE_EDGES = [(1, 2), (2, 6), (6, 5), (5, 1)]          # +x 面（原型 x=+.5 的四个角点）
VIEW_ORDER = ['top', 'front', 'left', 'right', 'back', 'bottom']
TILE = 480
COLS, ROWS = 3, 2


def proj(pts, K, D):
    uv, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                              np.asarray(K, np.float64), np.asarray(D, np.float64))
    return uv.reshape(-1, 2)


def box_corners(center, size, Rm):
    return (PROTO8 * np.asarray(size)) @ Rm.T + np.asarray(center)


def view_class(center, Rm):
    """相机在机体系下的方向 -> (视角类别, 方位角°, 仰角°)。机体 X 前 Y 左 Z 上；仰角 > 0 = 相机在目标上方。"""
    d = -np.asarray(center, np.float64)
    d /= np.linalg.norm(d)
    vb = Rm.T @ d
    el = float(np.degrees(np.arcsin(np.clip(vb[2], -1, 1))))
    az = float(np.degrees(np.arctan2(vb[1], vb[0])))
    # 相机高 1.6 m、目标 < 3.5 m，仰角超过 +15° 的「俯视」已经是极少数，阈值放宽到 15°
    if el > 15:
        cls = 'top'
    elif el < -35:
        cls = 'bottom'
    elif abs(az) <= 45:
        cls = 'front'
    elif abs(az) >= 135:
        cls = 'back'
    else:
        cls = 'left' if az > 0 else 'right'
    return cls, az, el


def pick_diverse(cands, per=6):
    """cands: list of dict(seq, view, z, ...)，按 z 升序。不同序列、视角类别轮转覆盖，同类别内取最近的。"""
    picks, used = [], set()
    by_view = {v: [c for c in cands if c['view'] == v] for v in VIEW_ORDER}
    while len(picks) < per:
        progressed = False
        for v in VIEW_ORDER:
            for c in by_view[v]:
                if c['seq'] in used:
                    continue
                picks.append(c)
                used.add(c['seq'])
                progressed = True
                break
            if len(picks) >= per:
                break
        if not progressed:
            break
    return picks


# --------------------------------------------------------------------------- #
# 画一个近景 tile
# --------------------------------------------------------------------------- #
def put_lines(img, lines, org=(6, 6), scale=0.5, color=(255, 255, 255)):
    """左上角带半透明底的多行文字；某行太长就单独缩小字号，保证不出 tile。"""
    x, y = org
    lh = int(22 * scale / 0.5)
    max_w = img.shape[1] - x - 8
    scales = []
    for t in lines:
        s = scale
        while s > 0.3 and cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 1)[0][0] > max_w:
            s -= 0.02
        scales.append(s)
    w = min(max_w, max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 1)[0][0] for t, s in zip(lines, scales)) + 10)
    h = lh * len(lines) + 8
    ov = img.copy()
    cv2.rectangle(ov, (x - 3, y - 2), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for i, (t, s) in enumerate(zip(lines, scales)):
        cv2.putText(img, t, (x + 2, y + lh * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, s, color, 1, cv2.LINE_AA)


def render_tile(img, K, D, center, size, Rm, axis_origin, vel_dir, lines, others=(), size_px=TILE, crop_mult=1.6):
    """img 原图；center/size/Rm 黄框；axis_origin 轴原点（仿真 = 框中心，MAV6D = VICON 原点）；
    vel_dir 单位速度方向（相机系）或 None；others = [(center,size,Rm)...] 同帧其他目标（灰细线）。"""
    corners = box_corners(center, size, Rm)
    L = 0.7 * float(max(size))
    ax_pts = np.vstack([axis_origin, axis_origin + Rm[:, 0] * L, axis_origin + Rm[:, 1] * L, axis_origin + Rm[:, 2] * L])
    pts = np.vstack([corners, ax_pts, np.asarray(center)[None]])
    if vel_dir is not None:
        pts = np.vstack([pts, (np.asarray(axis_origin) + vel_dir * L)[None]])
    uv = proj(pts, K, D)
    uvc, uva = uv[:8], uv[8:12]
    cc = uv[12]
    ext = max(np.ptp(uvc[:, 0]), np.ptp(uvc[:, 1]))
    half = max(24.0, 0.5 * crop_mult * ext)
    x0, y0 = cc[0] - half, cc[1] - half
    # 先裁（带边界填充）再放大，再在放大图上画线，线条清晰
    pad = int(np.ceil(half)) + 2
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    xi, yi = int(round(x0)) + pad, int(round(y0)) + pad
    side = int(round(2 * half))
    crop = big[max(yi, 0):yi + side, max(xi, 0):xi + side]
    tile = cv2.resize(crop, (size_px, size_px), interpolation=cv2.INTER_CUBIC)
    sx = size_px / float(crop.shape[1])
    sy = size_px / float(crop.shape[0])

    def T(p):
        return (int(round((p[0] - round(x0)) * sx)), int(round((p[1] - round(y0)) * sy)))

    for oc, osz, oR in others:
        ouv = proj(box_corners(oc, osz, oR), K, D)
        for a, b in EDGES:
            cv2.line(tile, T(ouv[a]), T(ouv[b]), (140, 140, 140), 1, cv2.LINE_AA)
    for a, b in EDGES:
        th = 2 if (a, b) in FRONT_FACE_EDGES or (b, a) in FRONT_FACE_EDGES else 1
        cv2.line(tile, T(uvc[a]), T(uvc[b]), (0, 255, 255), th, cv2.LINE_AA)
    o = T(uva[0])
    if vel_dir is not None:
        cv2.arrowedLine(tile, o, T(uv[13]), (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(tile, 'v', T(uv[13]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    for k, col, name in ((1, (0, 0, 255), 'x'), (2, (0, 255, 0), 'y'), (3, (255, 0, 0), 'z')):
        cv2.arrowedLine(tile, o, T(uva[k]), col, 3, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(tile, name, T(uva[k]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    cv2.circle(tile, o, 3, (255, 255, 255), -1, cv2.LINE_AA)
    put_lines(tile, lines)
    return tile


def make_sheet(tiles, header_lines, out_path):
    while len(tiles) < COLS * ROWS:
        tiles.append(np.full((TILE, TILE, 3), 30, np.uint8))
    rows = [np.concatenate(tiles[r * COLS:(r + 1) * COLS], 1) for r in range(ROWS)]
    grid = np.concatenate(rows, 0)
    hdr = np.full((26 * len(header_lines) + 6, grid.shape[1], 3), 20, np.uint8)
    for i, t in enumerate(header_lines):
        s = 0.6
        while s > 0.35 and cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 1)[0][0] > grid.shape[1] - 16:
            s -= 0.02
        cv2.putText(hdr, t, (8, 26 * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
    sheet = np.concatenate([hdr, grid], 0)
    assert sheet.shape[1] <= 1600 and sheet.shape[0] <= 1100, sheet.shape
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return sheet.shape


# --------------------------------------------------------------------------- #
# 仿真
# --------------------------------------------------------------------------- #
def short_seq(seq):
    p = seq.split('/')
    scene = p[0].replace('_Capture', '').replace('Demonstration', 'Demo').replace('Simple', 'Simp')
    return '%s/%s/%s' % (scene, p[2], p[4])


def load_raw_boxes(seq_dir, stem):
    with open(os.path.join(seq_dir, 'boxes_rgb', stem + '.pkl'), 'rb') as f:
        raw = pickle.load(f)
    out = {}
    for row in raw:
        if not isinstance(row[0], str):
            continue
        out[row[0]] = np.array(row[1:], dtype=np.float64).reshape(8, 3)
    return out


def sim_candidates(max_z):
    cands = {c: [] for c in CLASSES}
    seen = set()
    for split in ('train', 'test'):
        with open(os.path.join(SIM_CACHE, split, 'index.pkl'), 'rb') as f:
            idx = pickle.load(f)
        for i in idx['valid_idx']:
            m = idx['metas'][int(i)]
            if m['crop_wh'] is not None:
                continue
            W, H = m['raw_wh']
            for j, (n, b) in enumerate(zip(m['names'], m['boxes9d'])):
                b = b.astype(np.float64)
                if b[2] > max_z:
                    continue
                key = (m['seq'], m['frame'], j)
                if key in seen:
                    continue
                seen.add(key)
                Rm = R.from_euler('xyz', b[6:9]).as_matrix()
                uv = proj(box_corners(b[:3], b[3:6], Rm), K_SIM, D_SIM)
                cu = proj(b[:3][None], K_SIM, D_SIM)[0]
                ext = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
                half = 0.8 * ext
                crop_in = (cu[0] - half >= 0 and cu[1] - half >= 0 and cu[0] + half < W and cu[1] + half < H)
                inside8 = bool(((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).all())
                v, az, el = view_class(b[:3], Rm)
                cands[n].append(dict(seq=m['seq'], frame=m['frame'], j=j, z=float(b[2]), box=b, view=v, az=az, el=el,
                                     inside8=inside8, crop_in=crop_in, ext=float(ext), split=split))
    for c in cands:
        cands[c].sort(key=lambda d: d['z'])
    return cands


def sim_frame_pack(c):
    """读原图 + 原始角点，现算 9 参数（与缓存断言一致），找下一帧算速度方向。"""
    seq_dir = os.path.join(SIM_ROOT, c['seq'])
    stem = os.path.splitext(c['frame'])[0]
    img = cv2.imread(os.path.join(seq_dir, 'images_rgb', c['frame']), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[1] == 1280
    raw = load_raw_boxes(seq_dir, stem)
    # 找到与缓存框对应的原始行：中心最接近
    best, bname = None, None
    for name, cs in raw.items():
        if class_of(name) is None:
            continue
        p9 = corners_to_9params(cs).astype(np.float64)
        d = np.linalg.norm(p9[:3] - c['box'][:3])
        if best is None or d < best[0]:
            best, bname = (d, p9, cs), name
    d, p9, cs = best
    assert d < 1e-3, ('原始角点与缓存框中心不一致', c['seq'], c['frame'], d)
    assert np.allclose(p9[3:6], c['box'][3:6], atol=1e-3), (p9[3:6], c['box'][3:6])
    ang = np.degrees((R.from_euler('xyz', p9[6:9]).inv() * R.from_euler('xyz', c['box'][6:9])).magnitude())
    assert ang < 0.1, ('旋转与缓存不一致', ang)
    others = []
    for name, ocs in raw.items():
        if name == bname or class_of(name) is None:
            continue
        o9 = corners_to_9params(ocs).astype(np.float64)
        if o9[2] > 0.3:
            others.append((o9[:3], o9[3:6], R.from_euler('xyz', o9[6:9]).as_matrix()))
    # 下一帧
    frames = sorted([f for f in os.listdir(os.path.join(seq_dir, 'images_rgb')) if f.endswith('.png')], key=frame_sort_key)
    k = frames.index(c['frame'])
    vel_dir, speed, dt = None, None, None
    if k + 1 < len(frames):
        nstem = os.path.splitext(frames[k + 1])[0]
        try:
            nraw = load_raw_boxes(seq_dir, nstem)
        except Exception:
            nraw = {}
        if bname in nraw:
            dp = nraw[bname].mean(0) - cs.mean(0)
            dt = float(nstem) - float(stem)
            speed = float(np.linalg.norm(dp) / dt) if dt > 0 else None
            if np.linalg.norm(dp) > 0.005:
                vel_dir = dp / np.linalg.norm(dp)
    return img, p9, others, bname, vel_dir, speed, dt


def run_sim(args):
    cands = sim_candidates(6.0)
    outs = []
    for cls in CLASSES:
        # 距离阈值从 --max-z 起逐级放宽，直到裁窗全在画内的候选覆盖 >= per 条不同序列（Matrice-600-Pro 最近只有 3.8 m）
        for zmax in [args.max_z, 4.0, 4.5, 5.0, 6.0]:
            cs = [c for c in cands[cls] if c['z'] < zmax]
            good = [c for c in cs if c['crop_in']] or [c for c in cs if c['inside8']] or cs
            if len(set(c['seq'] for c in good)) >= args.per or zmax >= 6.0:
                break
        picks = pick_diverse(good, args.per)
        if len(picks) < args.per:                       # 序列不够就放宽到不要求整裁窗在画面内
            used = set(p['seq'] for p in picks)
            for c in cs:
                if c['seq'] not in used and len(picks) < args.per:
                    picks.append(c)
                    used.add(c['seq'])
        vc = {v: sum(1 for c in cs if c['view'] == v) for v in VIEW_ORDER}
        print('[sim] %-16s Z<%.1f 候选 %5d（裁窗全在画内 %d，序列 %d） 视角分布 %s' %
              (cls, zmax, len(cs), sum(c['crop_in'] for c in cs), len(set(c['seq'] for c in cs)), vc))
        for p in picks:
            print('       选 %-52s %-11s Z=%.2f view=%-6s az=%+6.1f el=%+6.1f' %
                  (short_seq(p['seq']), p['frame'], p['z'], p['view'], p['az'], p['el']))
        if args.stats:
            continue
        tiles = []
        for p in picks:
            img, p9, others, bname, vel_dir, speed, dt = sim_frame_pack(p)
            Rm = R.from_euler('xyz', p9[6:9]).as_matrix()
            lines = ['%s  [%s]' % (cls, p['split']),
                     short_seq(p['seq']),
                     'f=%s  Z=%.2fm  %s  az=%+.0f el=%+.0f' % (os.path.splitext(p['frame'])[0], p['z'], p['view'],
                                                             p['az'], p['el']),
                     'lwh=%.2fx%.2fx%.2f  %s' % (p9[3], p9[4], p9[5],
                                                 ('v=%.2fm/s' % speed) if speed is not None else 'v: n/a')]
            if speed is not None and vel_dir is None:
                lines[-1] += ' (~0, no arrow)'
            tiles.append(render_tile(img, K_SIM, D_SIM, p9[:3], p9[3:6], Rm, p9[:3], vel_dir, lines, others))
        out = os.path.join(OUT_DIR, 'sim_%s.jpg' % cls)
        hdr = ['SIM %s   (Z<%.1fm, 6 different sequences, raw 1280x720 PNG, true K fx=fy=640 cx=639.5 cy=359.5)'
               % (cls, zmax),
               'yellow = 3D box (thick edges = +x face)   red/green/blue = body x/y/z   '
               'magenta = velocity (box-centre motion to next frame)   grey = other drones in frame']
        shp = make_sheet(tiles, hdr, out)
        print('   -> %s %s' % (out, shp))
        outs.append((cls, out, len(tiles)))
    return outs


# --------------------------------------------------------------------------- #
# MAV6D
# --------------------------------------------------------------------------- #
_MAV_POS = {}


def mav_seq_positions(lab_dir):
    """一条序列所有标签的 (有序帧号列表, {帧号: t})，用于算逐帧位移（运动模糊代理）。"""
    if lab_dir not in _MAV_POS:
        stems = sorted([int(os.path.splitext(f)[0]) for f in os.listdir(lab_dir)
                        if f.endswith('.txt') and os.path.splitext(f)[0].isdigit()])
        pos = {}
        for s in stems:
            p = os.path.join(lab_dir, '%d.txt' % s)
            if os.path.getsize(p):
                pos[s] = read_truth_Rt(p)[1]
        _MAV_POS[lab_dir] = (stems, pos)
    return _MAV_POS[lab_dir]


def mav_next_disp(cls, scene, seq, stem):
    """到下一帧的位移向量（相机系，m/帧）；没有下一帧返回 None。"""
    stems, pos = mav_seq_positions(os.path.join(MAV_ROOT, cls, 'labels', scene, seq))
    s = int(stem)
    k = stems.index(s)
    if k + 1 < len(stems) and s in pos and stems[k + 1] in pos:
        return pos[stems[k + 1]] - pos[s]
    return None


def mav_candidates(max_z, blur_px=15.0):
    cands = {'phantom4': [], 'mavic2': []}
    seen = set()
    W, H = 1920, 1080
    for split in ('train', 'val', 'test'):
        p = os.path.join(MAV_CACHE, split, 'index.pkl')
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            idx = pickle.load(f)
        for i in idx['valid_idx']:
            m = idx['metas'][int(i)]
            b = m['boxes9d'][0].astype(np.float64)
            if b[2] > max_z or (m['seq'], m['frame']) in seen:
                continue
            seen.add((m['seq'], m['frame']))
            Rm = R.from_euler('xyz', b[6:9]).as_matrix()
            uv = proj(box_corners(b[:3] + Rm @ MAV_BOX_OFF, OB_SIZE, Rm), K_CALIB, D_CALIB)
            cu = proj(b[:3][None], K_CALIB, D_CALIB)[0]
            ext = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
            half = 0.8 * ext
            crop_in = (cu[0] - half >= 0 and cu[1] - half >= 0 and cu[0] + half < W and cu[1] + half < H)
            v, az, el = view_class(b[:3], Rm)
            cls, scene, seq = m['seq'].split('/')
            dp = mav_next_disp(cls, scene, seq, os.path.splitext(m['frame'])[0])
            # 运动模糊代理：逐帧位移投影到像素（mavic2 飞得快，很多帧糊成一团）；超过 blur_px 的排到后面
            px = float(K_CALIB[0, 0] * np.linalg.norm(dp) / b[2]) if dp is not None else 0.0
            cands[m['cls']].append(dict(seq=m['seq'], frame=m['frame'], z=float(b[2]), box=b, view=v, az=az, el=el,
                                        crop_in=crop_in, ext=float(ext), split=split, motion_px=px))
    for c in cands:
        cands[c].sort(key=lambda d: (d['motion_px'] > blur_px, d['z']))
    return cands


def mav_frame_pack(c):
    cls, scene, seq = c['seq'].split('/')
    stem = os.path.splitext(c['frame'])[0]
    img = cv2.imread(os.path.join(MAV_ROOT, cls, 'JPEGImages', scene, seq, c['frame']), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[1] == 1920
    lab_dir = os.path.join(MAV_ROOT, cls, 'labels', scene, seq)
    Rm, t = read_truth_Rt(os.path.join(lab_dir, stem + '.txt'))
    assert np.linalg.norm(t - c['box'][:3]) < 1e-4
    vel_dir, speed = None, None
    # 下一帧：phantom4 帧名 1,2,3...；mavic2 帧名是纳秒时间戳，都按数值排序取紧邻的下一个标签文件
    dp = mav_next_disp(cls, scene, seq, stem)
    if dp is not None:
        speed = float(np.linalg.norm(dp))          # 每帧位移（m/帧），MAV6D 帧率未知
        if speed > 0.005:
            vel_dir = dp / speed
    return img, Rm, t, vel_dir, speed


def run_mav(args):
    cands = mav_candidates(args.max_z_mav)
    outs = []
    for cls in ('phantom4', 'mavic2'):
        cs = cands[cls]
        good = [c for c in cs if c['crop_in']] or cs
        picks = pick_diverse(good, args.per)
        vc = {v: sum(1 for c in cs if c['view'] == v) for v in VIEW_ORDER}
        print('[mav6d] %-9s Z<%.1f 候选 %5d（裁窗全在画内 %d，序列 %d） 视角分布 %s' %
              (cls, args.max_z_mav, len(cs), sum(c['crop_in'] for c in cs), len(set(c['seq'] for c in cs)), vc))
        for p in picks:
            print('       选 %-24s %-24s Z=%.2f view=%-6s az=%+6.1f el=%+6.1f motion=%.0fpx/frame' %
                  (p['seq'], p['frame'], p['z'], p['view'], p['az'], p['el'], p['motion_px']))
        if args.stats:
            continue
        tiles = []
        for p in picks:
            img, Rm, t, vel_dir, speed = mav_frame_pack(p)
            lines = ['MAV6D %s  [%s]' % (cls, p['split']),
                     '%s  f=%s' % (p['seq'], os.path.splitext(p['frame'])[0]),
                     'Z=%.2fm  %s  az=%+.0f el=%+.0f' % (p['z'], p['view'], p['az'], p['el']),
                     'box=official p4 range  %s' % (('d=%.3fm/frame' % speed) if speed is not None else 'v: n/a')]
            if speed is not None and vel_dir is None:
                lines[-1] += ' (~0, no arrow)'
            tiles.append(render_tile(img, K_CALIB, D_CALIB, t + Rm @ MAV_BOX_OFF, OB_SIZE, Rm, t, vel_dir, lines))
        out = os.path.join(OUT_DIR, 'mav6d_%s.jpg' % cls)
        hdr = ['MAV6D %s   (Z<%.1fm, 6 different sequences, raw distorted 1920x1080 JPEG, cv2.projectPoints with '
               'official K + dist, sharp frames preferred)' % (cls, args.max_z_mav),
               'yellow = official phantom4 box (thick edges = +x face)   red/green/blue = MAV body x/y/z at VICON '
               'origin t   magenta = motion to next frame']
        shp = make_sheet(tiles, hdr, out)
        print('   -> %s %s' % (out, shp))
        outs.append((cls, out, len(tiles)))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true', help='只打印候选统计与选帧，不出图')
    ap.add_argument('--per', type=int, default=6)
    ap.add_argument('--max-z', type=float, default=3.5, help='仿真目标距离上限（m）')
    ap.add_argument('--max-z-mav', type=float, default=3.5, help='MAV6D 目标距离上限（m），不够再放宽')
    ap.add_argument('--only', default='', choices=['', 'sim', 'mav'])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    outs = []
    if args.only in ('', 'sim'):
        outs += [('sim', c, p, n) for c, p, n in run_sim(args)]
    if args.only in ('', 'mav'):
        outs += [('mav6d', c, p, n) for c, p, n in run_mav(args)]
    for o in outs:
        print('OUT', *o)


if __name__ == '__main__':
    main()
