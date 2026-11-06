# -*- coding: utf-8 -*-
"""PREREG_st2plus.md（sha256 156e8d81）第 1 步：ST2 五集成在真实训练集上的预测 → 第二段 V2Pp_s0..4 平均修正 + 因果全局滚动中位 ×0.75 → 新伪标签来源 dump。
不读真实标注（dump 里的 gt 字段原样保留，只给 make_pseudo_index 打印诊断）。输出 output/camnorm/st/real_train_preds_st2plus.pkl"""
import copy, pickle, sys
import numpy as np
ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT); sys.path.insert(0, G)
import softkd_analyze as ska
import dcalign_eval as dae
from anchor_eval import deploy
from st2combo_analyze import causal_global
from xpaste_final import deltas
from v2_feats import _TrainImages

SRC = ROOT + '/output/camnorm/st/real_train_preds_st2.pkl'
OUT = ROOT + '/output/camnorm/st/real_train_preds_st2plus.pkl'
MEM = [('V2Pp', j) for j in range(5)]
d = pickle.load(open(SRC, 'rb'))
recs = [dict(seq_id=f['seq'], frame_id=f['frame'], pred=np.asarray(f['pred'], np.float64).reshape(-1, 9), conf=np.asarray(f['conf'], np.float64).reshape(-1), gt=np.zeros((0, 9))) for f in d['frames']]
pr = deltas(recs, _TrainImages(), MEM)
vrecs = pickle.load(open(G + '/final/ens_val.pkl', 'rb'))['records']
c0 = ska.cval(deltas(vrecs, dae.ImagesVal(), MEM))
cc = causal_global(recs, pr, c0)
new = deploy(recs, pr, [x * 0.75 for x in cc])
out = copy.deepcopy(d)
nchg = 0
for f, r0, r1 in zip(out['frames'], recs, new):
    if len(r1['pred']):
        f['pred'] = r1['pred'].astype(np.asarray(f['pred']).dtype).reshape(np.asarray(f['pred']).shape)
        nchg += int(np.any(np.abs(r1['pred'][:, 2] - r0['pred'][:, 2]) > 1e-9))
out['info'] = dict(d['info'], pipeline=d['info']['pipeline'] + '；+ 第二段 V2Pp_s0..4 平均修正 + 因果全局滚动中位 ×0.75（PREREG_st2plus）', c0_val=c0)
pickle.dump(out, open(OUT, 'wb'))
z0 = np.array([r['pred'][0, 2] for r in recs if len(r['pred'])]); z1 = np.array([r['pred'][0, 2] for r in new if len(r['pred'])])
print('帧 %d，改了距离的帧 %d；距离倍数中位 %.3f（四分位 %.3f–%.3f）；验证先验 c0 = %.3f' % (len(recs), nchg, np.median(z1 / z0), np.percentile(z1 / z0, 25), np.percentile(z1 / z0, 75), c0))
