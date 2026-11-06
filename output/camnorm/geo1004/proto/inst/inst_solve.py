# -*- coding: utf-8 -*-
"""读 pairs_<split>.pkl，每片段最小二乘融合检测器绝对距离 + 实例原型相对尺度，评测片段内去偏后的逐帧距离误差（真值只评测）。"""
import pickle, sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
def solve(ld, pairs, sd, si, qmin):
    n = len(ld); rows, cols, vals, b, w = [], [], [], [], []
    for t in range(n):
        rows.append(len(b)); cols.append(t); vals.append(1 / sd); b.append(ld[t] / sd)
    for (a, c, ls, q) in pairs:
        if q < qmin: continue
        r = len(b); rows += [r, r]; cols += [a, c]; vals += [1 / si, -1 / si]; b.append(ls / si)
    A = coo_matrix((vals, (rows, cols)), shape=(len(b), n)).tocsr()
    return lsqr(A, np.array(b), atol=1e-10, btol=1e-10, iter_lim=5000)[0]
def evaluate(S, sd, si, qmin):
    E0, E1, seqs = [], [], []
    for s, ld, lg, pairs in S:
        x = solve(ld, pairs, sd, si, qmin)
        e0 = ld - lg; e1 = x - lg
        E0.append(e0 - np.median(e0)); E1.append(e1 - np.median(e1))
    E0, E1 = np.concatenate(E0), np.concatenate(E1)
    return 100*np.median(np.abs(E0)), 100*np.median(np.abs(E1)), 100*np.std(E0), 100*np.std(E1)
for sp in ('val', 'test'):
    S = pickle.load(open('pairs_%s.pkl' % sp, 'rb'))
    print('== %s  %d 片段 %d 帧' % (sp, len(S), sum(len(x[1]) for x in S)))
    print('   预设 σ检测器 5%%  σ实例 1.2%%  匹配≥0.5：逐帧误差|中位| %.2f%% → %.2f%%   std %.2f%% → %.2f%%' % evaluate(S, 0.05, 0.012, 0.5))
    for sd, si, qm in ((0.05, 0.006, 0.5), (0.05, 0.02, 0.5), (0.05, 0.012, 0.3), (0.05, 0.012, 0.7), (0.03, 0.012, 0.5), (0.08, 0.012, 0.5)):
        print('   敏感性 σd %.0f%% σi %.1f%% 匹配≥%.1f：|中位| %.2f%% → %.2f%%   std %.2f%% → %.2f%%' % ((100*sd, 100*si, qm) + evaluate(S, sd, si, qm)))
