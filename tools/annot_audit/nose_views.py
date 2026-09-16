# -*- coding: utf-8 -*-
"""仿真网格机头 vs 标签 x 轴：按「相机在标签机体系里的方位角」挑近距离原图，裁大看机头。

az = atan2(v_y, v_x)，v = 相机方向（机体系，x 前 y 左 z 上）。标签若正确：az≈180 看到机尾，az≈+90 看到机体左侧。
每个机型取 az≈180 / +90 / -90 三类各 2 张最近的，原始 PNG 裁块放大到 360x360，只画标签 x（红）与 y（绿）短箭头。
    python tools/annot_audit/nose_views.py
"""
import os
import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/nose_views'
os.makedirs(OUT, exist_ok=True)
K = np.array([[640, 0, 639.5], [0, 640, 359.5], [0, 0, 1.0]])
cands = {}
for split in ('train', 'test'):
    idx = pickle.load(open('E:/mmcache/indoor8jpg/%s/index.pkl' % split, 'rb'))
    for i, m in enumerate(idx['metas']):
        if not m or len(m['boxes9d']) != 1:
            continue
        b = np.asarray(m['boxes9d'][0], float)
        Rm = R.from_euler('xyz', b[6:9]).as_matrix()
        v = Rm.T @ (-b[:3]); v /= np.linalg.norm(v)
        az = np.degrees(np.arctan2(v[1], v[0])); el = np.degrees(np.arcsin(v[2]))
        uv = K @ b[:3]; uv = uv[:2] / uv[2]
        if abs(el) > 20 or not (150 < uv[0] < 1130 and 150 < uv[1] < 570):
            continue
        cands.setdefault(m['names'][0], []).append((b[2] / max(b[3:5]), az, el, m['seq'], m['frame'], b))
tiles = []
for name in sorted(cands):
    row = []
    for target in (180, 90, -90):
        c = [x for x in cands[name] if abs((x[1] - target + 180) % 360 - 180) < 20]
        c.sort(key=lambda x: x[0])
        seen = set()
        for rel, az, el, seq, frame, b in c:
            if seq in seen:
                continue
            seen.add(seq)
            img = cv2.imread(os.path.join('D:/data_collect', seq, 'images_rgb', frame), cv2.IMREAD_COLOR)
            Rm = R.from_euler('xyz', b[6:9]).as_matrix()
            pts = np.array([b[:3], b[:3] + Rm[:, 0] * b[3] * 0.35, b[:3] + Rm[:, 1] * b[4] * 0.35])
            uv = (K @ pts.T).T; uv = uv[:, :2] / uv[:, 2:3]
            half = int(max(40, 1.1 * 640 * max(b[3:5]) / b[2] / 2))
            cu, cvv = int(uv[0, 0]), int(uv[0, 1])
            pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
            crop = pad[cvv:cvv + 2 * half, cu:cu + 2 * half]
            s = 360.0 / (2 * half)
            crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_CUBIC)
            o = (180, 180)
            for k, col in ((1, (0, 0, 255)), (2, (0, 255, 0))):
                p = (int(180 + (uv[k, 0] - uv[0, 0]) * s), int(180 + (uv[k, 1] - uv[0, 1]) * s))
                cv2.arrowedLine(crop, o, p, col, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.rectangle(crop, (0, 0), (360, 22), (0, 0, 0), -1)
            cv2.putText(crop, '%s az%+.0f el%+.0f Z%.1f' % (name[:14], az, el, b[2]), (4, 16), 0, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            row.append(crop)
            if len(seen) == 2:
                break
        while len(row) % 2:
            row.append(np.zeros((360, 360, 3), np.uint8))
    tiles.append(np.hstack(row))
    cv2.imwrite(os.path.join(OUT, '%s.jpg' % name), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(name, len(cands[name]), 'row width', np.hstack(row).shape)
