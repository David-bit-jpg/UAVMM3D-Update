# -*- coding: utf-8 -*-
"""MAV6D 机体轴高清近景：每机型取最近的若干帧（不同序列，train+val+test 都找），
在 1920x1080 原始 JPEG 上画 细的机体轴（红 x / 绿 y / 蓝 z，长 0.2 m）+ 官方角点框，裁剪放大成单张 800x800，
供人工判断红 x 是否指向云台 / 机头（Phantom 4：云台挂在机头下方，前臂 LED 红、后臂 LED 绿；
Mavic 2：云台在机头，前臂 LED 红）。
"""
import os
import pickle
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

ROOT = 'E:/MAV6D'
CACHE = 'E:/mmcache/mav6d_cn'
OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/mav_code/closeup'
os.makedirs(OUT, exist_ok=True)
K_CALIB = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.1, 533.9717], [0.0, 0.0, 1.0]])
D_CALIB = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
mnx, mxx, mny, mxy, mnz, mxz = -0.18, 0.16, -0.16, 0.18, -0.17, 0.06
C8 = np.array([[mnx, mny, mnz], [mnx, mny, mxz], [mnx, mxy, mnz], [mnx, mxy, mxz],
               [mxx, mny, mnz], [mxx, mny, mxz], [mxx, mxy, mnz], [mxx, mxy, mxz]])
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def proj(pts):
    uv, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_CALIB, D_CALIB)
    return uv.reshape(-1, 2)


cands = {'phantom4': [], 'mavic2': []}
for split in ('train', 'val', 'test'):
    idx = pickle.load(open(os.path.join(CACHE, split, 'index.pkl'), 'rb'))
    for i in idx['valid_idx']:
        m = idx['metas'][i]
        b = m['boxes9d'][0].astype(np.float64)
        cands[m['cls']].append((float(b[2]), split, m['seq'], m['frame'], b))
per = 6
for cls in cands:
    cands[cls].sort(key=lambda x: x[0])
    seen = set()
    k = 0
    for z, split, seqk, frame, b in cands[cls]:
        if seqk in seen:
            continue
        seen.add(seqk)
        c, scene, seq = seqk.split('/')
        img = cv2.imread(os.path.join(ROOT, c, 'JPEGImages', scene, seq, frame))
        Rm = R.from_euler('xyz', b[6:9]).as_matrix()
        t = b[:3]
        c8 = C8 @ Rm.T + t
        L = 0.2
        uv = proj(np.vstack([c8, t, t + Rm[:, 0] * L, t + Rm[:, 1] * L, t + Rm[:, 2] * L]))
        ext = max(np.ptp(uv[:8, 0]), np.ptp(uv[:8, 1]))
        h = int(0.75 * ext) + 20
        cu, cv_ = int(uv[8, 0]), int(uv[8, 1])
        pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT)
        crop = pad[cv_:cv_ + 2 * h, cu:cu + 2 * h]
        s = 800.0 / (2 * h)
        big = cv2.resize(crop, (800, 800), interpolation=cv2.INTER_CUBIC)
        clean = big.copy()
        uv2 = (uv - np.array([cu - h, cv_ - h])) * s
        for a, bb in EDGES:
            cv2.line(big, tuple(np.int32(uv2[a])), tuple(np.int32(uv2[bb])), (0, 255, 255), 1, cv2.LINE_AA)
        o = tuple(np.int32(uv2[8]))
        for kk, col, name in ((9, (0, 0, 255), 'x'), (10, (0, 255, 0), 'y'), (11, (255, 0, 0), 'z')):
            cv2.arrowedLine(big, o, tuple(np.int32(uv2[kk])), col, 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(big, name, tuple(np.int32(uv2[kk]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        cv2.putText(big, '%s %s %s/%s %s z=%.2f' % (cls, split, scene, seq, frame, z), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        both = np.concatenate([clean, big], 1)
        cv2.imwrite(os.path.join(OUT, '%s_%02d_%s_%s.jpg' % (cls, k, seq, os.path.splitext(frame)[0])), both,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(cls, k, split, seqk, frame, 'z=%.2f' % z, 'yaw(cam) x-axis dir', np.round(Rm[:, 0], 2))
        k += 1
        if k == per:
            break
