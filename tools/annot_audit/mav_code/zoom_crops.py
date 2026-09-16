# -*- coding: utf-8 -*-
"""对指定帧做 2 倍放大的紧裁剪（只含无人机），左：原图，右：细线机体轴（x 红 / y 绿 / z 蓝，0.15 m）+ '-x' 方向白色短线。"""
import os
import pickle
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

ROOT = 'E:/MAV6D'
CACHE = 'E:/mmcache/mav6d_cn'
OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/mav_code/zoom'
os.makedirs(OUT, exist_ok=True)
K = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.1, 533.9717], [0.0, 0.0, 1.0]])
D = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
WANT = {('mavic2/03/0302', '1649918855330990314.jpg'), ('mavic2/03/0305', '1649918920046311617.jpg'),
        ('mavic2/02/0211', '1649918398450536966.jpg'), ('mavic2/01/0104', '1649918006019544601.jpg'),
        ('mavic2/02/0201', '1649918252598290205.jpg'), ('mavic2/01/0105', '1649918009057963848.jpg'),
        ('phantom4/02/0206', '61.jpg'), ('phantom4/01/0102', '684.jpg')}


def proj(pts):
    uv, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(-1, 2)


for split in ('train', 'val', 'test'):
    idx = pickle.load(open(os.path.join(CACHE, split, 'index.pkl'), 'rb'))
    for i in idx['valid_idx']:
        m = idx['metas'][i]
        if (m['seq'], m['frame']) not in WANT:
            continue
        c, scene, seq = m['seq'].split('/')
        img = cv2.imread(os.path.join(ROOT, c, 'JPEGImages', scene, seq, m['frame']))
        b = m['boxes9d'][0].astype(np.float64)
        Rm = R.from_euler('xyz', b[6:9]).as_matrix()
        t = b[:3]
        L = 0.15
        uv = proj(np.vstack([t, t + Rm[:, 0] * L, t + Rm[:, 1] * L, t + Rm[:, 2] * L, t - Rm[:, 0] * L]))
        ext = 1979.0 * 0.36 / t[2]
        h = int(0.6 * ext)
        cu, cv_ = int(uv[0, 0]), int(uv[0, 1])
        pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT)
        crop = pad[cv_:cv_ + 2 * h, cu:cu + 2 * h]
        s = 2.0
        big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        clean = big.copy()
        uv2 = (uv - np.array([cu - h, cv_ - h])) * s
        o = tuple(np.int32(uv2[0]))
        for kk, col, name in ((1, (0, 0, 255), 'x'), (2, (0, 255, 0), 'y'), (3, (255, 0, 0), 'z')):
            cv2.arrowedLine(big, o, tuple(np.int32(uv2[kk])), col, 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(big, name, tuple(np.int32(uv2[kk]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        cv2.line(big, o, tuple(np.int32(uv2[4])), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(big, '-x', tuple(np.int32(uv2[4]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        both = np.concatenate([clean, big], 1)
        cv2.putText(both, '%s %s z=%.2f  x_cam=%s' % (m['seq'], m['frame'], t[2], np.round(Rm[:, 0], 2)), (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, '%s_%s_%s.jpg' % (c, seq, os.path.splitext(m['frame'])[0])), both, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print('wrote', c, seq, m['frame'], both.shape)
