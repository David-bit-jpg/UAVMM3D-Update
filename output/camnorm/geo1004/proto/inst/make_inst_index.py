# -*- coding: utf-8 -*-
"""实例原型修正伪标签距离：每片段最小二乘（伪标签距离 σ 5% + 实例剪影相对尺度 σ 1.2%，匹配质量 ≥0.8，参数按验证集定），
沿伪标签视线缩放位置，尺寸/朝向不动 → index_pseudo_st2inst.pkl。真值只用来打印修正前后误差。"""
import pickle, glob, copy, os
import numpy as np
exec(open('inst_solve.py', encoding='utf-8').read().split("def evaluate")[0])
R = '@@MMCACHE@@/mav6d_cn_obj/train/'
P0 = pickle.load(open(R + os.environ.get('INST_IDX', 'index_pseudo_st2plus.pkl'), 'rb')); P = copy.deepcopy(P0)
S = sum((pickle.load(open(f, 'rb')) for f in sorted(glob.glob(os.environ.get('INST_PAIRS', 'pairs2_train') + '_w*.pkl'))), [])
E0, E1, ch, n = [], [], [], 0
for s, ld, lg, pairs, ids in S:
    x = solve(ld, pairs, 0.05, 0.012, 0.8)
    for t, i in enumerate(ids):
        b = np.asarray(P['metas'][i]['boxes9d'], np.float64).copy(); f = np.exp(x[t] - ld[t]); b[0, :3] *= f
        P['metas'][i]['boxes9d'] = b.astype(np.float32); ch.append(abs(np.log(f))); n += 1
    ok = np.isfinite(lg)
    e0, e1 = (ld - lg)[ok], (x - lg)[ok]
    E0.append(e0 - np.median(e0)); E1.append(e1 - np.median(e1))
E0, E1 = np.concatenate(E0), np.concatenate(E1)
P['pseudo'] = dict(P0['pseudo'], depth='instance_prototype_silhouette_ls(sd=0.05,si=0.012,q>=0.8)')
pickle.dump(P, open(R + os.environ.get('INST_OUTIDX', 'index_pseudo_st2inst.pkl'), 'wb'))
print('修正 %d / %d 帧，距离改动 中位 %.2f%%；伪标签片段内逐帧误差 |中位| %.2f%% → %.2f%%，std %.2f%% → %.2f%%（真值只评测）' % (
    n, len(P0['valid_idx']), 100*np.median(ch), 100*np.median(np.abs(E0)), 100*np.median(np.abs(E1)), 100*np.std(E0), 100*np.std(E1)))
