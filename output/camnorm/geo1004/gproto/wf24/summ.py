# -*- coding: utf-8 -*-
"""汇总：读 wf23/mirror/fe_<名>_w<ww>_<split>_P.pkl，按论文口径打分（综合分 / 平均精度 / 位置误差 / 朝向误差 / 航向中位）。用法 python summ.py w15 名1 名2 ..."""
import sys, pickle, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/gproto/wf23/mirror')
import chain_eval_m as CE
ww = sys.argv[1]
for n in sys.argv[2:]:
    row = []
    for sp in ('val', 'test'):
        f = '%s/gproto/wf23/mirror/fe_%s_%s_%s_P.pkl' % (G, n, ww, sp) if ww != '-' else '%s/gproto/wf23/mirror/fe_%s_%s_P.pkl' % (G, n, sp)
        D = pickle.load(open(f, 'rb')); P, valid = D['P'], D['valid']
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
        m = CE.PT.metrics(P, M, valid); am = CE.ang_metrics(P, M, valid)
        pe = float(np.nanmedian(np.linalg.norm(P['pt'] - M['gt'][:, :3], axis=1)[valid]))
        row.append('%s 综合 %.2f AP %.2f 位置 %.3f(中位 %.3f) 朝向 %.2f 航向中位 %.2f' % (sp, m['UDS'], m['AP'], m['ATE'], pe, m['AOE'], np.nanmedian(am['hs'])))
    print('%-14s | %s' % (n, ' | '.join(row)), flush=True)
