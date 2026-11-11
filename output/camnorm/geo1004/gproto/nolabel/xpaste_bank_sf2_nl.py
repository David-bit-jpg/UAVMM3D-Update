# -*- coding: utf-8 -*-
"""geo1004/xpaste_bank_sf2.py 的无标签链副本（11-10）：规则逐字相同（同帧平移 1.8 倍框宽、视野约 √3·w 见方完全在图内、随机方向取第一个可行、最多 8000 张、rng 种子 0），
只改：选窗检测 = 新检测器 <arm> 在真实 train 全部帧的单帧检测（nolabel/<arm>/recs/<arm>_train.pkl）；图像按去真值帧表取（不打开 index.pkl、不用 diag_bgs.Frames）；
不写真机残留诊断用的元数据之外的任何东西。→ xpaste/bank_bg_sf2_<arm>.npy + nolabel/<arm>/bank_bg_sf2_meta.pkl
用法：python xpaste_bank_sf2_nl.py <arm>"""
import pickle, sys, os
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'; NL = G + '/gproto/nolabel'
sys.path.insert(0, '@@ROOT@@'); sys.path.insert(0, G)
from scipy.spatial.transform import Rotation as R
from uavdet3d.utils import pose_eval as pe
import cropw_extract as ce
import silw_train as sw
ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
arm = sys.argv[1]
out = G + '/xpaste/bank_bg_sf2_%s.npy' % arm
if os.path.exists(out):
    print('exists', out); sys.exit(0)
T = pickle.load(open(NL + '/train_frames_nogt.pkl', 'rb'))
key = {k: i for i, k in enumerate(T['keys'])}
arr = np.load('@@MMCACHE@@/mav6d_cn_obj/train/rgb.npy', mmap_mode='r')
H, W = arr.shape[1:3]
recs = pickle.load(open('%s/%s/recs/%s_train.pkl' % (NL, arm, arm), 'rb'))['records']
rng = np.random.default_rng(0)
DIRS = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
HALF = 0.5 * float(S0)
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
    i = key[(str(r['seq_id']), str(r['frame_id']))]
    img = np.ascontiguousarray(arr[i]); K = np.asarray(T['K_in'][i], np.float64)
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
    bank.append(sw.view_img(ce.window(img, ok[0], ok[1], ce.KW * w), S0, 0, 0, 1.0, w, False))
    meta.append((str(r['seq_id']), str(r['frame_id']), float(ok[0]), float(ok[1]), float(w)))
sel = np.sort(rng.choice(len(bank), min(8000, len(bank)), replace=False))
np.save(out + '.tmp.npy', np.stack([bank[i] for i in sel])); os.replace(out + '.tmp.npy', out)
pickle.dump([meta[i] for i in sel], open('%s/%s/bank_bg_sf2_meta.pkl' % (NL, arm), 'wb'))
print('单帧背景库（%s）：合格检测 %d，放不下平移窗跳过 %d，入库 %d，存 %d' % (arm, nall, nskip, len(bank), len(sel)), flush=True)
