# -*- coding: utf-8 -*-
"""11-08 wf21/strict：单帧背景库 v2（复核发现 bank_bg_sf.npy 选窗用的检测记录 irloc_train_recs/train_P0_s3.pkl
是带热图时序背景 causal100 推理的 → 选窗位置本身用了同段前 100 帧；且误检多，背景窗里残留真机）。
规则与 xpaste_bank_sf.py 逐字相同（同帧平移 1.8 倍框宽、最终视野约 √3·w 见方完全在图内、随机方向取第一个可行、最多 8000 张、rng 种子 0），
只把检测记录换成**单帧检测器**：全严格位置检测器 wsfnb（WSFNB_s0..2 三集成 + 翻转，推理不开热图时序背景）在真实 train 全部帧上的检测
（gproto/wf21/strict/recs/wsfnb_train.pkl）。只用最高分检测（≥0.5）的框，不读标签。→ xpaste/bank_bg_sf2.npy"""
import pickle, sys
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, '@@ROOT@@'); sys.path.insert(0, G)
import diag_bgs as db
from scipy.spatial.transform import Rotation as R
from uavdet3d.utils import pose_eval as pe

fr = db.Frames('train')
H, W = fr.arr.shape[1:3]
recs = pickle.load(open(G + '/gproto/wf21/strict/recs/wsfnb_train.pkl', 'rb'))['records']
rng = np.random.default_rng(0)
DIRS = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
HALF = 0.5 * float(db.S0)          # 最终视野半边长 / w（约 0.87）
bank, nall, nskip, meta = [], 0, 0, []
for r in recs:
    if len(r['pred']) == 0:
        continue
    pi = int(np.argmax(r['conf']))
    if r['conf'][pi] < 0.5:
        continue
    g = r['pred'][pi]
    if g[2] <= 1e-3:
        continue
    i = fr.key[(r['seq_id'], r['frame_id'])]
    img = np.ascontiguousarray(fr.arr[i]); K = np.asarray(fr.metas[i]['K_in'], np.float64)
    c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
    q = (K @ c.T).T
    if (q[:, 2] <= 1e-3).any():
        continue
    uv = q[:, :2] / q[:, 2:3]; u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max(); w = u1 - u0
    if not (6.0 <= w <= 170.0):
        continue
    nall += 1
    cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    ok = None
    for k in rng.permutation(len(DIRS)):
        a, b = DIRS[k]
        x, y = cx + 1.8 * w * a, cy + 1.8 * w * b
        if x - HALF * w >= 0 and x + HALF * w <= W - 1 and y - HALF * w >= 0 and y + HALF * w <= H - 1:
            ok = (x, y); break
    if ok is None:
        nskip += 1
        continue
    bank.append(db.sw.view_img(db.ce.window(img, ok[0], ok[1], db.ce.KW * w), db.S0, 0, 0, 1.0, w, False))
    meta.append((str(r['seq_id']), str(r['frame_id']), float(ok[0]), float(ok[1]), float(w)))
sel = np.sort(rng.choice(len(bank), min(8000, len(bank)), replace=False))
np.save(G + '/xpaste/bank_bg_sf2.npy', np.stack([bank[i] for i in sel]))
pickle.dump([meta[i] for i in sel], open(G + '/gproto/wf21/strict/bank_bg_sf2_meta.pkl', 'wb'))   # 窗口位置（供真机残留诊断，诊断时才读真值）
print('单帧背景库 v2：合格检测 %d，放不下平移窗跳过 %d，入库 %d，存 %d' % (nall, nskip, len(bank), len(sel)), flush=True)
