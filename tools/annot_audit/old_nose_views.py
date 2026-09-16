# -*- coding: utf-8 -*-
"""旧 CARLA 数据（E:/data_collect/Town*/No-MotionBlur，只读）是否也有机头 90° 问题。
从 near15 子集挑单机近距离帧，用 build_mm_cache.corners_to_9params 从原始角点算标签，
按「相机在标签机体系的方位角」分 az≈180（标签正后方）/ ≈+90 / ≈-90，每机型各 2 张原图裁块。
    python tools/annot_audit/old_nose_views.py
"""
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, 'E:/Open3DUAVDet/tools')
import build_mm_cache as BMC   # noqa: E402
from uavdet3d.utils import camera_geometry as cg   # noqa: E402

ROOT = 'E:/data_collect'
OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/old_carla'
os.makedirs(OUT, exist_ok=True)
rows = []
for lst in ('near_train.txt', 'near_test.txt'):
    for ln in open('E:/Open3DUAVDet/tools/cfgs/subsets/near15/' + lst, encoding='utf-8'):
        if ln.startswith('#') or not ln.strip():
            continue
        seq, frame, n, zmin, zmax = ln.split()
        if int(n) != 1 or not seq.split('/')[0].startswith(('Town', 'No-Motion')):
            continue
        rows.append((float(zmin), seq, frame))
rows.sort()
cands = {}
Kcache = {}
for z, seq, frame in rows:
    base = os.path.join(ROOT, seq)
    if seq not in Kcache:
        info = pickle.load(open(os.path.join(base, 'im_info.pkl'), 'rb'))
        K = np.array(info['rgb']['intrinsic'], float)
        img0 = None
        W = int(round(2 * K[0, 2] - 4)) if abs(K[1, 1] / K[0, 0] - 1.01) < 1e-3 else int(round(2 * K[0, 2]))
        Kcache[seq] = K
    K = Kcache[seq]
    pk = os.path.join(base, 'boxes_rgb', os.path.splitext(frame)[0] + '.pkl')
    if not os.path.exists(pk):
        continue
    raw = pickle.load(open(pk, 'rb'))
    rr = [r for r in raw if isinstance(r[0], str) and BMC.class_of(r[0]) is not None]
    if len(rr) != 1:
        continue
    c = np.array(rr[0][1:], dtype=np.float64).reshape(8, 3)
    b = BMC.corners_to_9params(c)
    Rm = R.from_euler('xyz', b[6:9]).as_matrix()
    v = Rm.T @ (-b[:3]); v /= np.linalg.norm(v)
    az = np.degrees(np.arctan2(v[1], v[0])); el = np.degrees(np.arcsin(v[2]))
    if abs(el) > 30:
        continue
    name = BMC.class_of(rr[0][0])
    for t in (180, 90, -90):
        if abs((az - t + 180) % 360 - 180) < 25:
            cands.setdefault((name, t), []).append((b[2] / max(b[3:5]), seq, frame, b, az, el))
names = sorted({k[0] for k in cands})
sheet_rows = []
for name in names:
    tiles = []
    for t in (180, 90, -90):
        got, seen = 0, set()
        for rel, seq, frame, b, az, el in sorted(cands.get((name, t), []), key=lambda x: x[0]):
            if seq in seen:
                continue
            img = cv2.imread(os.path.join(ROOT, seq, 'images_rgb', frame))
            if img is None:
                continue
            H, W = img.shape[:2]
            K, _ = cg.legacy_sim_intrinsic_fix(Kcache[seq], W, H)
            uv = K @ b[:3]; uv = uv[:2] / uv[2]
            half = int(max(24, 0.8 * K[0, 0] * max(b[3:5]) / b[2]))
            if not (half < uv[0] < W - half and half < uv[1] < H - half):
                continue
            seen.add(seq)
            crop = cv2.resize(img[int(uv[1]) - half:int(uv[1]) + half, int(uv[0]) - half:int(uv[0]) + half], (300, 300), interpolation=cv2.INTER_CUBIC)
            Rm = R.from_euler('xyz', b[6:9]).as_matrix()
            s = 300 / (2 * half)
            for k, col in ((0, (0, 0, 255)), (1, (0, 255, 0))):
                tip = K @ (b[:3] + Rm[:, k] * max(b[3:5]) * 0.5); tip = tip[:2] / tip[2]
                cv2.arrowedLine(crop, (150, 150), (int(150 + (tip[0] - uv[0]) * s), int(150 + (tip[1] - uv[1]) * s)), col, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.rectangle(crop, (0, 0), (300, 20), (0, 0, 0), -1)
            cv2.putText(crop, '%s az%+.0f Z%.1f %dpx' % (name[:12], az, b[2], 2 * half), (3, 14), 0, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop); got += 1
            if got == 2:
                break
        while got < 2:
            tiles.append(np.zeros((300, 300, 3), np.uint8)); got += 1
    sheet_rows.append(np.hstack(tiles))
    print(name, {t: len(cands.get((name, t), [])) for t in (180, 90, -90)})
cv2.imwrite(os.path.join(OUT, 'old_nose_views.jpg'), np.vstack(sheet_rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print('->', os.path.join(OUT, 'old_nose_views.jpg'))
