# -*- coding: utf-8 -*-
"""PREREG_xpaste.md（sha256 f82012af）背景库：真实训练集每段时间中位背景（无标签），按 P0_s3 真实训练帧最高分检测（≥0.5）取窗，
只留背景差分"可用"的窗口，最多 8000 个 → xpaste/bank_bg.npy（uint8 N×128×128×3）。"""
import pickle, sys
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, '@@ROOT@@'); sys.path.insert(0, G)
import diag_bgs as db
from scipy.spatial.transform import Rotation as R
from uavdet3d.utils import pose_eval as pe

fr = db.Frames('train')
recs = pickle.load(open(G + '/irloc_train_recs/train_P0_s3.pkl', 'rb'))['records']
bank, nall = [], 0
for r in recs:
    if len(r['pred']) == 0:
        continue
    pi = int(np.argmax(r['conf']))
    if r['conf'][pi] < 0.5:
        continue
    g = r['pred'][pi]
    if g[2] <= 1e-3:
        continue
    img, bg, K = fr.get(r['seq_id'], r['frame_id'])
    c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
    q = (K @ c.T).T
    if (q[:, 2] <= 1e-3).any():
        continue
    uv = q[:, :2] / q[:, 2:3]; u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max(); w = u1 - u0
    if not (6.0 <= w <= 170.0):
        continue
    cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    v = db.sw.view_img(db.ce.window(img, cx, cy, db.ce.KW * w), db.S0, 0, 0, 1.0, w, False)
    bv = db.sw.view_img(db.ce.window(bg, cx, cy, db.ce.KW * w), db.S0, 0, 0, 1.0, w, False)
    nall += 1
    if db.bgs_mask(v, bv)[1]:
        bank.append(bv)
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(bank), min(8000, len(bank)), replace=False))
np.save(G + '/xpaste/bank_bg.npy', np.stack([bank[i] for i in sel]))
print('窗口 %d，可用 %d，存 %d' % (nall, len(bank), len(sel)))
