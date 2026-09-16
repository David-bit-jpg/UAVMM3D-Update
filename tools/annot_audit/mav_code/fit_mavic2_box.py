# -*- coding: utf-8 -*-
"""从 labels_yolo6d 的 9 个关键点反推官方给 mavic2（以及 phantom4 复核）用的角点范围 [min_x,max_x,min_y,max_y,min_z,max_z]：
已知每帧位姿（labels txt）与 util.py 的畸变投影，对 6 个未知量做最小二乘。"""
import os
import glob
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = 'E:/MAV6D'
OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/mav_code'
D = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
CAM2VIC = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                    [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                    [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                    [0, 0, 0, 1]])


def corners9(e):
    mnx, mxx, mny, mxy, mnz, mxz = e
    return np.array([[(mnx + mxx) / 2, (mny + mxy) / 2, (mnz + mxz) / 2],
                     [mnx, mny, mnz], [mnx, mny, mxz], [mnx, mxy, mnz], [mnx, mxy, mxz],
                     [mxx, mny, mnz], [mxx, mny, mxz], [mxx, mxy, mnz], [mxx, mxy, mxz]])


def project(pts_mav, up):
    r = R.from_quat(up[3:7])
    cv_ = r.apply(pts_mav) + up[:3]
    pc = CAM2VIC[:3, :3] @ cv_.T + CAM2VIC[:3, 3:4]
    xp = pc[0] / pc[2]
    yp = pc[1] / pc[2]
    r2 = xp * xp + yp * yp
    xd = xp * (1 + D[0] * r2 + D[1] * r2 ** 2 + D[4] * r2 ** 3) + 2 * D[2] * xp * yp + D[3] * (r2 + 2 * xp * xp)
    yd = yp * (1 + D[0] * r2 + D[1] * r2 ** 2 + D[4] * r2 ** 3) + D[2] * (r2 + 2 * yp * yp) + 2 * D[3] * xp * yp
    return np.stack([1979.4 * xd + 976.8189, 1979.1 * yd + 533.9717], 1)


lines = []
for cls in ('mavic2', 'phantom4'):
    files = sorted(glob.glob(os.path.join(ROOT, cls, 'labels_yolo6d', '01', '*', '*.txt')))[::40][:60]
    poses, kps = [], []
    for f in files:
        lab = os.path.join(ROOT, cls, 'labels', *f.split(os.sep)[-2:]) if os.sep in f else None
        lab = f.replace('labels_yolo6d', 'labels')
        if not os.path.exists(lab) or os.path.getsize(lab) == 0 or os.path.getsize(f) == 0:
            continue
        v = np.array(list(map(float, open(lab).read().strip().split())))
        y = np.array(list(map(float, open(f).read().strip().split())))
        if len(v) < 16 or len(y) < 21:
            continue
        poses.append(v[9:16])
        kps.append(y[1:19].reshape(9, 2) * np.array([1920., 1080.]))
    poses, kps = np.array(poses), np.array(kps)

    def resid(e):
        c = corners9(e)
        return np.concatenate([(project(c, p) - k).ravel() for p, k in zip(poses, kps)])

    sol = least_squares(resid, np.array([-0.18, 0.16, -0.16, 0.18, -0.17, 0.06]))
    e = sol.x
    rms = np.sqrt(np.mean(sol.fun ** 2))
    s = ('[%s] %d frames: fitted extents x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]  size %.3f x %.3f x %.3f  centre offset (%.3f,%.3f,%.3f)  rms %.3f px'
         % (cls, len(poses), e[0], e[1], e[2], e[3], e[4], e[5], e[1] - e[0], e[3] - e[2], e[5] - e[4],
            (e[0] + e[1]) / 2, (e[2] + e[3]) / 2, (e[4] + e[5]) / 2, rms))
    print(s)
    lines.append(s)
open(os.path.join(OUT, 'fit_box_from_yolo6d.txt'), 'w').write('\n'.join(lines))
