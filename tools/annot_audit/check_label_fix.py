# -*- coding: utf-8 -*-
"""仿真标签修正（index_nose / index_nosescale）上线前检查。
a) 视角覆盖：相机在机体系的方位角分布，修正前/后，以及加上水平翻转（az -> -az）后的并集
b) 正面视角画新 x 轴（每机型 2 张，新 az≈0 且离得近），供目视确认机头
    python tools/annot_audit/check_label_fix.py
"""
import os
import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/label_fix'
os.makedirs(OUT, exist_ok=True)
K = np.array([[640, 0, 639.5], [0, 640, 359.5], [0, 0, 1.0]])


def az_el(b):
    Rm = R.from_euler('xyz', b[6:9]).as_matrix()
    v = Rm.T @ (-b[:3]); v /= np.linalg.norm(v)
    return np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arcsin(v[2]))


for split in ('train',):
    old = pickle.load(open('E:/mmcache/indoor8jpg/%s/index.pkl' % split, 'rb'))
    new = pickle.load(open('E:/mmcache/indoor8jpg/%s/index_nose.pkl' % split, 'rb'))
    ns = pickle.load(open('E:/mmcache/indoor8jpg/%s/index_nosescale.pkl' % split, 'rb'))
    A0, A1, rows = [], [], []
    for i in old['valid_idx']:
        i = int(i)
        for j, (bo, bn) in enumerate(zip(old['metas'][i]['boxes9d'], new['metas'][i]['boxes9d'])):
            a0, _ = az_el(np.asarray(bo, float)); a1, e1 = az_el(np.asarray(bn, float))
            A0.append(a0); A1.append(a1)
            rows.append((i, j, a1, e1))
    A0, A1 = np.array(A0), np.array(A1)
    bins = np.arange(-180, 181, 30)
    h0 = np.histogram(A0, bins)[0] / len(A0); h1 = np.histogram(A1, bins)[0] / len(A1)
    hf = (np.histogram(A1, bins)[0] + np.histogram(-A1, bins)[0]) / (2 * len(A1))
    print('方位角 30° 分箱（从 -180 起；0 = 相机在机头正前方，+90 = 机体左侧）')
    print('  修正前       ', np.round(h0, 3))
    print('  机头修正后   ', np.round(h1, 3))
    print('  +水平翻转 50%', np.round(hf, 3))
    for nm, A in (('修正前', A0), ('修正后', A1)):
        print('  %s: 正面|az|<45 %.3f  背面|az|>135 %.3f  左侧(45,135) %.3f  右侧(-135,-45) %.3f' % (
            nm, np.mean(np.abs(A) < 45), np.mean(np.abs(A) > 135), np.mean((A > 45) & (A < 135)), np.mean((A < -45) & (A > -135))))
    # 距离分布（尺度修正后）
    z0 = np.array([np.asarray(b)[2] for m in old['metas'] if m for b in m['boxes9d']])
    z2 = np.array([np.asarray(b)[2] for m in ns['metas'] if m for b in m['boxes9d']])
    print('  深度 Z 中位：原 %.2f m，尺度修正后 %.2f m（MAV6D test 3.38 m）；p10/p90 修正后 %.2f/%.2f' % (
        np.median(z0), np.median(z2), np.percentile(z2, 10), np.percentile(z2, 90)))
    lwh = {}
    for m in ns['metas']:
        if not m:
            continue
        for b, n in zip(m['boxes9d'], m['names']):
            lwh.setdefault(n, np.asarray(b)[3:6])
    print('  尺度修正后各机型 l,w,h:', {k: np.round(v, 3).tolist() for k, v in sorted(lwh.items())})

    # b) 正面视角画新机体轴
    tiles = []
    by_name = {}
    for i, j, a1, e1 in rows:
        m = new['metas'][i]
        if len(m['boxes9d']) != 1 or abs(a1) > 20 or abs(e1) > 25:
            continue
        b = np.asarray(m['boxes9d'][0], float)
        uv = K @ b[:3]; uv = uv[:2] / uv[2]
        if not (150 < uv[0] < 1130 and 150 < uv[1] < 570):
            continue
        by_name.setdefault(m['names'][0], []).append((b[2] / max(b[3:5]), i, a1, e1, b))
    for name in sorted(by_name):
        c = sorted(by_name[name], key=lambda x: x[0])
        seen, got = set(), 0
        for _, i, a1, e1, b in c:
            m = new['metas'][i]
            if m['seq'] in seen:
                continue
            seen.add(m['seq'])
            img = cv2.imread(os.path.join('D:/data_collect', m['seq'], 'images_rgb', m['frame']))
            Rm = R.from_euler('xyz', b[6:9]).as_matrix()
            L = max(b[3:6])
            pts = np.array([b[:3], b[:3] + Rm[:, 0] * L * 0.6, b[:3] + Rm[:, 1] * L * 0.4, b[:3] + Rm[:, 2] * L * 0.4])
            uv = (K @ pts.T).T; uv = uv[:, :2] / uv[:, 2:3]
            half = int(max(50, 640 * 0.8 * max(b[3:5]) / b[2]))
            pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
            cu, cv_ = int(uv[0, 0]), int(uv[0, 1])
            crop = cv2.resize(pad[cv_:cv_ + 2 * half, cu:cu + 2 * half], (380, 380), interpolation=cv2.INTER_CUBIC)
            s = 380 / (2 * half)
            for k, col in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):
                p = (int(190 + (uv[k, 0] - uv[0, 0]) * s), int(190 + (uv[k, 1] - uv[0, 1]) * s))
                cv2.arrowedLine(crop, (190, 190), p, col, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.rectangle(crop, (0, 0), (380, 22), (0, 0, 0), -1)
            cv2.putText(crop, '%s new az%+.0f el%+.0f' % (name[:15], a1, e1), (4, 16), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
            got += 1
            if got == 2:
                break
    while len(tiles) % 4:
        tiles.append(np.zeros((380, 380, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(OUT, 'front_views_new_axes.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('  正面视角图 ->', os.path.join(OUT, 'front_views_new_axes.jpg'), '（红 = 新 x，应指向相机/机头）')
