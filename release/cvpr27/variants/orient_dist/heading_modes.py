# -*- coding: utf-8 -*-
"""[review1113 orient_dist C1 supplement] where the unfolded heading error HEAD = |atan2(E10, E00)|, E = Rg^T Rp, falls, on each arm's TP frames:
share in [0,45) (right quadrant), [45,135) (90 deg off), [135,180] (front/back flipped). Same loading / TP rule as orient_dist.py. CPU only."""
import os, json, runpy
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
OUT = os.path.dirname(os.path.abspath(__file__))
src = open(OUT + '/orient_dist.py', encoding='utf-8').read().split("OM = ('AOE90'")[0]   # helpers + SETS only (no analysis)
G = {'__file__': OUT + '/orient_dist.py', '__name__': 'od'}; exec(compile(src, 'orient_dist_head', 'exec'), G)
res = {}
for tgt, cfg in G['SETS'].items():
    M = np.load(cfg['meta']); g = M['gt']; K = M['K']; cls = np.array([str(x) for x in M['cls']]); c = G['S'].CFG[cfg['rules']]
    subs = {'all': np.ones(len(g), bool)}; subs.update({a: cls == code for a, code in cfg['air'].items()})
    for arm, fn in cfg['arms'].items():
        F = G['per_frame'](G['load'](fn, M), g, K, c['rel'])
        for sub, m in subs.items():
            h = F['hd'][F['tp'] & m]
            res.setdefault(tgt, {}).setdefault(sub, {})[arm] = dict(n_tp=int(len(h)), lt45=float(np.mean(h < 45)), mid=float(np.mean((h >= 45) & (h < 135))),
                                                                   ge135=float(np.mean(h >= 135)))
            print(tgt, sub, arm, res[tgt][sub][arm], flush=True)
json.dump(res, open(OUT + '/heading_modes.json', 'w'), indent=1)
