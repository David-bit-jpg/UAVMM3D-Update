# -*- coding: utf-8 -*-
"""机体 x 轴与飞行方向：用 VICON 系里相邻帧的位移（每序列按时间戳排序）表达到机体系，看水平速度在机体 xy 平面的方位角分布。
四旋翼常前飞，若 x 指机头，则方位角峰在 0°；若机头在 -x，则峰在 180°；若机头在 ±y，则峰在 ±90°。
同时给出 mavic2 / phantom4 两机型的框中心偏移在机体系的方向，作对照。"""
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = 'E:/MAV6D'
OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/mav_code'
lines = []
for cls in ('phantom4', 'mavic2'):
    seqs = sorted(glob.glob(os.path.join(ROOT, cls, 'labels', '*', '*')))
    az_all, sp_all = [], []
    n_seq = 0
    for sd in seqs:
        files = glob.glob(os.path.join(sd, '*.txt'))
        recs = []
        for f in files:
            if os.path.getsize(f) == 0:
                continue
            v = np.array(list(map(float, open(f).read().strip().split())))
            if len(v) < 16:
                continue
            recs.append((v[8], v[9:12], v[12:16]))
        if len(recs) < 5:
            continue
        recs.sort(key=lambda r: r[0])
        ts = np.array([r[0] for r in recs]) * 1e-9
        p = np.array([r[1] for r in recs])
        q = np.array([r[2] for r in recs])
        dt = np.diff(ts)
        ok = (dt > 1e-3) & (dt < 0.5)
        vel = np.diff(p, axis=0) / dt[:, None]
        Rm = R.from_quat(q[:-1]).as_matrix()          # body -> vicon
        vb = np.einsum('nji,nj->ni', Rm, vel)          # vicon -> body (R^T v)
        sp = np.linalg.norm(vb[:, :2], axis=1)
        m = ok & (sp > 0.3)                             # 只统计明显水平运动
        az_all.append(np.degrees(np.arctan2(vb[m, 1], vb[m, 0])))
        sp_all.append(sp[m])
        n_seq += 1
    az = np.concatenate(az_all)
    h, _ = np.histogram(az, bins=np.arange(-180, 181, 30))
    s = ('[%s] %d sequences, %d moving samples (|v_xy|>0.3 m/s): azimuth of body-frame velocity, 30-deg bins from -180: %s\n'
         '   fraction |az|<45 (moving toward +x) %.2f, |az-180|<45 (toward -x) %.2f, toward +y %.2f, toward -y %.2f'
         % (cls, n_seq, len(az), (100 * h / len(az)).round(1).tolist(),
            np.mean(np.abs(az) < 45), np.mean(np.abs(np.abs(az) - 180) < 45),
            np.mean(np.abs(az - 90) < 45), np.mean(np.abs(az + 90) < 45)))
    print(s)
    lines.append(s)
open(os.path.join(OUT, 'velocity_heading.txt'), 'w').write('\n'.join(lines))
