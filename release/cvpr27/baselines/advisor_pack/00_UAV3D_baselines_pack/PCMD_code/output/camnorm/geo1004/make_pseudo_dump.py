# -*- coding: utf-8 -*-
"""通用：检测器集成在真实训练集上的预测 dump → 第二段 V2Pp_s0..4 平均修正 + 因果全局滚动中位 × λ → 新伪标签来源 dump（不读真实标注）。
用法：python make_pseudo_dump.py <输入 dump> <输出 dump> <λ> <验证集同管线记录 pkl（给预热先验）>"""
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

SRC, OUT, LAM, VREC = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
MEM = [('V2Pp', j) for j in range(5)]
d = pickle.load(open(SRC, 'rb'))
recs = [dict(seq_id=f['seq'], frame_id=f['frame'], pred=np.asarray(f['pred'], np.float64).reshape(-1, 9),
             conf=np.asarray(f['conf'], np.float64).reshape(-1), gt=np.zeros((0, 9))) for f in d['frames']]
pr = deltas(recs, _TrainImages(), MEM)
vrecs = pickle.load(open(VREC, 'rb'))['records']
c0 = ska.cval(deltas(vrecs, dae.ImagesVal(), MEM))
new = deploy(recs, pr, [x * LAM for x in causal_global(recs, pr, c0)])
out = copy.deepcopy(d)
for f, r1 in zip(out['frames'], new):
    if len(r1['pred']):
        f['pred'] = r1['pred'].astype(np.asarray(f['pred']).dtype).reshape(np.asarray(f['pred']).shape)
out['info'] = dict(d['info'], pipeline=d['info']['pipeline'] + '；+ 第二段 V2Pp_s0..4 平均修正 + 因果全局滚动中位 ×%.2f' % LAM, c0_val=c0)
pickle.dump(out, open(OUT, 'wb'))
z0 = np.array([r['pred'][0, 2] for r in recs if len(r['pred'])])
z1 = np.array([r['pred'][0, 2] for r in new if len(r['pred'])])
print('帧 %d；距离倍数中位 %.3f（四分位 %.3f–%.3f）；验证先验 c0 = %.3f' % (len(recs), np.median(z1 / z0), np.percentile(z1 / z0, 25), np.percentile(z1 / z0, 75), c0), flush=True)
