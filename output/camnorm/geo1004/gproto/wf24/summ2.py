# -*- coding: utf-8 -*-
"""论文表用：读 fe_<名>_<ww>_<split>_P.pkl（ww='-' 表示无 w 后缀），报 UDS / AP / ATE / ASE(cm) / AOE / 航向中位 / 位置中位，以及测试集近 / 中 / 远（<3 / 3–4 / ≥4 m）的 AP 与 UDS。用法 python summ2.py <ww> 名1 名2 ..."""
import sys, pickle, os, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/gproto/wf23/mirror')
import chain_eval_m as CE
ww = sys.argv[1]
OUT = {}
for n in sys.argv[2:]:
    OUT[n] = {}
    for sp in ('val', 'test'):
        f = '%s/gproto/wf23/mirror/fe_%s_%s_%s_P.pkl' % (G, n, ww, sp) if ww != '-' else '%s/gproto/wf23/mirror/fe_%s_%s_P.pkl' % (G, n, sp)
        D = pickle.load(open(f, 'rb')); P, valid = D['P'], D['valid']
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
        m = CE.PT.metrics(P, M, valid); am = CE.ang_metrics(P, M, valid)
        pe = float(np.nanmedian(np.linalg.norm(P['pt'] - M['gt'][:, :3], axis=1)[valid]))
        row = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], ASE=m['ASE'], AOE=m['AOE'], yaw_med=float(np.nanmedian(am['hs'])), pos_med=pe, mDR=m['mDR'])
        d = np.linalg.norm(M['gt'][:, :3], axis=1)
        for nm, lo, hi in CE.PT.BINS:
            sel = valid[(d[valid] >= lo) & (d[valid] < hi)]
            mb = CE.PT.metrics(P, M, sel)
            row['AP_' + nm] = mb['AP']; row['UDS_' + nm] = mb['UDS']
        OUT[n][sp] = row
        print('%-12s %s UDS %.2f AP %.2f ATE %.3f(中位 %.3f) ASE %.2f AOE %.2f 航向中位 %.2f 检出 %.1f | %s' % (
            n, sp, row['UDS'], row['AP'], row['ATE'], pe, row['ASE'], row['AOE'], row['yaw_med'], row['mDR'],
            ' '.join('%s AP %.2f UDS %.2f' % (nm, row['AP_' + nm], row['UDS_' + nm]) for nm, _, _ in CE.PT.BINS)), flush=True)
json.dump(OUT, open(G + '/gproto/wf24/summ2_%s.json' % '_'.join(sys.argv[2:])[:80], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
