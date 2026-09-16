# -*- coding: utf-8 -*-
"""Empirical checks of the simulation label conventions traced in UavRecorder.cpp / build_mm_cache.py.

Read-only on the dataset. Writes a text summary to
E:/Open3DUAVDet/output/camnorm/annot_audit/sim_code/.

Checks
  A. Corner order / handedness: c1-c0, c3-c0, c4-c0 orthogonal, det sign in the OpenCV frame.
  B. Heading: label body-x (c1-c0, mapped to UE world) vs. flight direction (finite-difference
     velocity of the box center in UE world), per drone type, moving frames only.
  C. Box vs. LiDAR points tagged as drone (same time stamp): fraction inside box, extent of the
     tagged points in the box-local frame; and the centroid offset for LiDAR stamp offsets 0 / +6
     (tests the same-tick claim of the recorder against build_mm_cache's --lidar-offset default).
  D. Cache consistency: boxes9d in E:/mmcache/indoor8cn/test/index.pkl vs. corners_to_9params of
     the raw pickle for the same seq/frame.
"""
import glob
import os
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, 'E:/Open3DUAVDet/tools')
sys.path.insert(0, 'E:/Open3DUAVDet')
from build_mm_cache import corners_to_9params, CARLA_TO_OPENCV, OPENCV_TO_CARLA  # noqa: E402

ROOT = 'D:/data_collect'
OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/sim_code'
os.makedirs(OUT, exist_ok=True)
DT = 1.0 / 15.0
CLASSES = ['DJI-avata2', 'DJI-phantom4', 'drone-unk3', 'm210-rtk', 'DJI-mavic-mini', 'Matrice-600-Pro', 'matrix-300-RTK']


def stamp_key(p):
    return float(os.path.splitext(os.path.basename(p))[0])


def cls_of(name):
    for c in CLASSES:
        if c in name:
            return c
    return '?'


def cv_to_world(p_cv, E):
    P = np.c_[p_cv, np.ones(len(p_cv))]
    return (E @ (OPENCV_TO_CARLA @ P.T)).T[:, :3]


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def load_combo(combo_dir):
    info = pickle.load(open(os.path.join(combo_dir, 'im_info.pkl'), 'rb'))
    E = np.array(info['rgb']['extrinsic'], dtype=np.float64)
    files = sorted(glob.glob(os.path.join(combo_dir, 'boxes_rgb', '*.pkl')), key=stamp_key)
    frames = []
    for f in files:
        raw = pickle.load(open(f, 'rb'))
        rows = {}
        for row in raw:
            c = np.array(row[1:], dtype=np.float64).reshape(8, 3)
            rows[row[0]] = c
        frames.append((stamp_key(f), rows))
    return E, frames


def check_A_B(combo_dirs, log):
    dets, dots = [], []
    head = {}     # cls -> list of signed yaw diffs (deg)
    tilt = {}     # cls -> list of |pitch| of body x in world (deg)
    for cd in combo_dirs:
        E, frames = load_combo(cd)
        names = sorted(set(n for _, rows in frames for n in rows))
        for n in names:
            cls = cls_of(n)
            prev = None
            for t, rows in frames:
                if n not in rows:
                    prev = None
                    continue
                c = rows[n]
                ex, ey, ez = c[1] - c[0], c[3] - c[0], c[4] - c[0]
                Rm = np.stack([ex / np.linalg.norm(ex), ey / np.linalg.norm(ey), ez / np.linalg.norm(ez)], 1)
                dets.append(np.linalg.det(Rm))
                dots.append(max(abs(ex @ ey), abs(ey @ ez), abs(ex @ ez)) / (np.linalg.norm(ex) * np.linalg.norm(ey)))
                cw = cv_to_world(c, E)
                center = cw.mean(0)
                xw = cw[1] - cw[0]
                if prev is not None and abs(t - prev[0] - DT) < 1e-3:
                    v = (center - prev[1]) / DT
                    if np.linalg.norm(v[:2]) > 0.2:       # moving horizontally > 0.2 m/s
                        yaw_x = np.degrees(np.arctan2(xw[1], xw[0]))
                        yaw_v = np.degrees(np.arctan2(v[1], v[0]))
                        head.setdefault(cls, []).append(wrap(yaw_x - yaw_v))
                        tilt.setdefault(cls, []).append(np.degrees(np.arcsin(xw[2] / np.linalg.norm(xw))))
                prev = (t, center)
    dets = np.array(dets)
    log('A. corners: n=%d  det(min/max)=%.4f/%.4f  max|cos| between edges=%.2e' % (len(dets), dets.min(), dets.max(), max(dots)))
    log('B. heading: signed yaw(label body-x in UE world) - yaw(velocity), moving frames only')
    for cls in sorted(head):
        a = np.array(head[cls])
        p = np.percentile(a, [5, 25, 50, 75, 95])
        log('   %-16s n=%5d  median=%7.2f  p5/p25/p75/p95=%7.2f/%7.2f/%7.2f/%7.2f  |d|<10deg: %.1f%%   pitch(body x) median=%.2f' %
            (cls, len(a), p[2], p[0], p[1], p[3], p[4], 100.0 * np.mean(np.abs(a) < 10), np.median(tilt[cls])))


def check_C(combo_dir, log, every=10, offsets=(0, 6)):
    E, frames = load_combo(combo_dir)
    lr = pickle.load(open(os.path.join(combo_dir, 'lidar_radar_info.pkl'), 'rb'))
    L = np.array(lr['lidars'][0]['extrinsic'], dtype=np.float64)
    stamps = [t for t, _ in frames]
    name = sorted(set(n for _, rows in frames for n in rows))[0]
    res = {o: {'inside': [], 'cent_off': [], 'local': []} for o in offsets}
    npy0 = None
    for i in range(0, len(frames), every):
        t, rows = frames[i]
        if name not in rows:
            continue
        c = rows[name]
        p9 = corners_to_9params(c)
        center = c.mean(0)
        ax = np.stack([c[1] - c[0], c[3] - c[0], c[4] - c[0]], 1)
        lwh = np.linalg.norm(ax, axis=0)
        ax = ax / lwh
        for o in offsets:
            j = i + o
            if j >= len(frames):
                continue
            npy = os.path.join(combo_dir, 'lidar_1', '%.4f.npy' % stamps[j])
            if not os.path.exists(npy):
                continue
            P = np.load(npy)
            if npy0 is None:
                npy0 = (npy, P.shape, P.dtype)
            tag = P[:, 4]
            xyz = P[tag == 1][:, :3].astype(np.float64)
            if len(xyz) < 5:
                continue
            world = (L @ np.c_[xyz, np.ones(len(xyz))].T).T
            cv = (CARLA_TO_OPENCV @ (np.linalg.inv(E) @ world.T)).T[:, :3]
            loc = (cv - center) @ ax                         # box-local coords (m), axes = c1-c0, c3-c0, c4-c0
            inside = np.all(np.abs(loc) <= lwh / 2 + 0.01, axis=1)
            res[o]['inside'].append(inside.mean())
            res[o]['cent_off'].append(np.linalg.norm(cv.mean(0) - center))
            res[o]['local'].append(loc)
    log('C. LiDAR (tag==1) vs box, %s  drone=%s  npy=%s' % (os.path.basename(combo_dir), name, npy0))
    for o in offsets:
        r = res[o]
        if not r['inside']:
            log('   offset %+d: no data' % o)
            continue
        loc = np.concatenate(r['local'])
        lo, hi = np.percentile(loc, 1, axis=0), np.percentile(loc, 99, axis=0)
        log('   lidar stamp offset %+d frames: n_frames=%d  inside-box fraction mean=%.3f  centroid-to-boxcenter (m) median=%.3f max=%.3f'
            % (o, len(r['inside']), np.mean(r['inside']), np.median(r['cent_off']), np.max(r['cent_off'])))
        log('      tagged-point extent in box frame p1..p99 (m): x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]  (box half-size %.3f %.3f %.3f)'
            % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], *(lwh / 2)))


def check_D(log):
    idx = pickle.load(open('E:/mmcache/indoor8cn/test/index.pkl', 'rb'))
    log('D. cache index: format=%s intrinsic_mode=%s lidar_offset=%s every=%s' % (idx.get('format'), idx.get('intrinsic_mode'), idx.get('lidar_offset'), idx.get('every')))
    n = 0
    worst = 0.0
    for m in idx['metas']:
        if m is None:
            continue
        seq = m['seq'].replace('/', os.sep)
        f = os.path.join(ROOT, m['seq'], 'boxes_rgb', os.path.splitext(m['frame'])[0] + '.pkl')
        if not os.path.exists(f):
            continue
        raw = pickle.load(open(f, 'rb'))
        recon = {cls_of(row[0]): corners_to_9params(np.array(row[1:], dtype=np.float64).reshape(8, 3)) for row in raw}
        for name, b in zip(m['names'], m['boxes9d']):
            if name not in recon:
                continue
            r = recon[name]
            d = np.abs(r[:6] - b[:6]).max()
            ang = (R.from_euler('xyz', r[6:9]).inv() * R.from_euler('xyz', b[6:9])).magnitude()
            worst = max(worst, d, ang)
            n += 1
        if n >= 300:
            break
    log('   %d cached boxes re-derived from raw corners: max |diff| over x,y,z,l,w,h and rotation = %.2e' % (n, worst))
    # body z up? column 2 of R (body z in cv frame) should have negative y (cv y is down)
    up = []
    fwd_pitch = []
    for m in idx['metas'][:2000]:
        if m is None:
            continue
        for b in m['boxes9d']:
            Rm = R.from_euler('xyz', b[6:9]).as_matrix()
            up.append(Rm[1, 2])
            fwd_pitch.append(Rm[1, 0])
    up = np.array(up)
    log('   body z axis cv-y component: mean=%.3f, fraction negative (=up)=%.3f ; body x cv-y comp |mean|=%.3f' % (up.mean(), (up < 0).mean(), abs(np.mean(fwd_pitch))))


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    single = sorted(glob.glob(os.path.join(ROOT, 'Demonstration_Simple_Capture', 'carla_data', '00001', 'clear_day', '*')))
    single = [d for d in single if os.path.isdir(d)]
    multi = sorted(glob.glob(os.path.join(ROOT, 'Warehouse_Capture', 'carla_data', '00003', 'clear_day', '*')))
    multi = [d for d in multi if os.path.isdir(d)][:4]
    log('combos: %d single + %d multi' % (len(single), len(multi)))
    check_A_B(single + multi, log)
    for cd in single:
        if 'phantom4' in cd or 'Matrice-600' in cd:
            check_C(cd, log)
    check_D(log)
    with open(os.path.join(OUT, 'check_sim_label_conventions.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
