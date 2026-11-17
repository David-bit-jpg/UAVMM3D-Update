# -*- coding: utf-8 -*-
"""[11-13] 正文用的紧凑版 Fig. 3：两行，每行 3 格，单栏 3.33 in，高约 1.6 in。
复用 make_fig_qual.py 的 panel()，样例取自同一个 selection.json（挑选规则不变）：
MAV6D：成功 / 典型 / 位置失败；AntiUAV3D：成功 / 典型 / Matrice 400 位置失败。图例放在顶部一行。"""
import json, pickle, sys, os
sys.argv = [sys.argv[0]]
import make_fig_qual as M
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
PDF = 'D:/CVPR27_UAV/figures/fig_qual_main.pdf'; PNG = M.OUT.rsplit('/', 1)[0] + '/fig_qual_main.png'
PICK = {'mav': [1, 3, 4], 'anti': [1, 3, 4]}
TAG = {'mav': 'MAV6D', 'anti': 'AntiUAV3D'}
metas = {}
for ds, p in M.CACHE.items():
    I = pickle.load(open(p, 'rb')); metas[ds] = {(m['seq'], m['frame']): m for m in I['metas']}
FW = 3.33; pw = (FW - 0.02) / 3; phh = pw * 0.68; leg = 0.11
FH = leg + 2 * phh + 0.01
fig = plt.figure(figsize=(FW, FH)); L = iter('abcdef'); info = {}
for row, ds in enumerate(('mav', 'anti')):
    for c, k in enumerate(PICK[ds]):
        r = dict(M.SEL[ds]['rows'][k])
        ax = fig.add_axes([(0.003 + c * (pw + 0.006)) / FW, (FH - leg - (row + 1) * phh + 0.006) / FH, (pw - 0.003) / FW, (phh - 0.012) / FH])
        info['%s_%d' % (ds, k)] = M.panel(ax, ds, r, metas[ds][(r['seq'], r['frame'])], next(L))
fig.legend(handles=[Line2D([], [], color=M.C_PRED, lw=1.0, label='prediction (PST)'), Line2D([], [], color=M.C_GT, lw=1.0, label='ground truth')],
           loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=5.0, handlelength=1.8, borderaxespad=0.05)
fig.savefig(PDF, dpi=400); fig.savefig(PNG, dpi=400)
print('wrote', PDF, fig.get_size_inches(), [ (ds, [M.SEL[ds]['rows'][k]['kind'] for k in PICK[ds]]) for ds in PICK])
