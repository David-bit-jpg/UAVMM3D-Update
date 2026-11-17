# -*- coding: utf-8 -*-
"""N1：第一代第二段（红外剪影蒸馏测距网络 V2Pp_s0..4）在验证集教师检测上的逐检测修正量 δ̂（与 strict_deltas.py / make_st2plus_dump.py 同网络同规则）。
输入两份教师验证集检测：
  nb  = 本目录 real_val_preds_st2nb.pkl（ST2 五集成 + 翻转，无时序背景）→ 最终无标签链 nl075 的教师
  bg  = geo1004/final/ens_val.pkl（ST2 五集成 + 翻转 + 时序背景 causal100）→ 11-07 wa 的教师（st2plus）
只推理，不训练；显存上限 0.15。输出 val_deltas.pkl。"""
import os, sys, pickle
import numpy as np
import torch
torch.cuda.set_per_process_memory_fraction(0.15, 0)
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'; N1 = G + '/gproto/review1113/N1'
sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, G); sys.path.insert(0, G + '/finalads')
import unc_dump as UD
XF = UD.XF
MEM = [('V2Pp', j) for j in range(5)]
imgs = XF.dae.ImagesVal()
out = {}
d = pickle.load(open(N1 + '/real_val_preds_st2nb.pkl', 'rb'))
recs = [dict(seq_id=f['seq'], frame_id=f['frame'], pred=np.asarray(f['pred'], np.float64).reshape(-1, 9),
             conf=np.asarray(f['conf'], np.float64).reshape(-1), gt=np.zeros((0, 9))) for f in d['frames']]
with torch.no_grad():
    out['nb'] = dict(keys=[(r['seq_id'], r['frame_id']) for r in recs], unc=UD.deltas_full(recs, imgs, MEM))
print('nb', len(recs), flush=True)
vr = pickle.load(open(G + '/final/ens_val.pkl', 'rb'))['records']
vr = [dict(r, gt=np.zeros((0, 9))) for r in vr]
with torch.no_grad():
    out['bg'] = dict(keys=[(r['seq_id'], r['frame_id']) for r in vr], unc=UD.deltas_full(vr, imgs, MEM))
print('bg', len(vr), flush=True)
pickle.dump(out, open(N1 + '/val_deltas.pkl', 'wb'))
