# -*- coding: utf-8 -*-
"""11-09 诊断（真值只诊断）：最终链上两种距离的逐帧误差，按颜色类（暗≈Mavic 2 / 亮≈Phantom 4）。
z_fin = 红外教的测距网络修正后的距离；z_kp = 最终朝向学生角点 PnP 的距离。各自去掉整体偏置（中位数）后报稳健散布 1.4826·MAD，以及两者误差的相关系数。
配置同 chain_eval_m（环境变量 FE_*，自动设 FE_ZKP=1）。用法：python diag_corner.py val test"""
import os, sys, json
os.environ['FE_ZKP'] = '1'
sys.path.insert(0, '@@ROOT@@/output/camnorm/geo1004/gproto/wf23/mirror')
import numpy as np
import chain_eval_m as CE
rob = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))
out = {}
for sp in sys.argv[1:]:
    recs, K = CE.build(sp)
    cc = np.asarray(K['col_c'])
    E = {0: [], 1: []}
    for r in recs:
        if not r.get('z_kp') or not len(r['pred']) or r['lum'] is None:
            continue
        p = np.asarray(r['pred']).reshape(-1, 9)[int(np.argmax(r['conf']))]
        g = np.asarray(r['gt']).reshape(-1, 9)
        if not len(g):
            continue
        g = g[np.argmin(np.linalg.norm(g[:, :3] - p[:3], axis=1))]
        if np.linalg.norm(g[:3] - p[:3]) > 1.0 or p[2] <= 0:
            continue
        k = int(np.argmin(np.abs(cc - r['lum'])))
        E[k].append((np.log(p[2] / g[2]), np.log(r['z_kp'] / g[2])))
    res = {}
    for k, v in E.items():
        a = np.array(v)
        ef, ek = a[:, 0] - np.median(a[:, 0]), a[:, 1] - np.median(a[:, 1])
        res[('dark' if k == 0 else 'bright')] = dict(n=len(a), ir_rob=rob(a[:, 0]), kp_rob=rob(a[:, 1]), corr=float(np.corrcoef(ef, ek)[0, 1]))
    out[sp] = res
    print(sp, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open('@@ROOT@@/output/camnorm/geo1004/gproto/wf24/diag_corner_%s.json' % os.environ.get('FE_OUT', 'x'), 'w'), indent=1)
