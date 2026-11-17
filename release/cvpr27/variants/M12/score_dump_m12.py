# -*- coding: utf-8 -*-
"""[M12] score a raw detector dump (eval_dump_nl records: pred (n,9), conf) with the SAME metric code as nolabel/fuse_eval_nl.py
(cmpeval/paper_table.metrics on cmpeval/<split>_meta.npz, top-confidence box per frame, no re-ranking). Ground truth is read only here.
usage: python score_dump_m12.py <dump.pkl> <split> <out.json>"""
import os, sys, json, pickle
import numpy as np
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'
sys.path.insert(0, G + '/cmpeval')
import paper_table as PT  # noqa
dump, sp, out = sys.argv[1:4]
recs = pickle.load(open(dump, 'rb'))['records']
M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
valid = np.where(M['n_gt'] > 0)[0]; N = len(M['seq'])
P = {'pt': np.full((N, 3), np.nan), 'plwh': np.full((N, 3), np.nan), 'peul': np.full((N, 3), np.nan), 'conf': np.full(N, np.nan)}
nf = 0
for r in recs:
    p = np.asarray(r['pred'], np.float64).reshape(-1, 9); c = np.asarray(r['conf'], np.float64).reshape(-1)
    i = idx.get((str(r['seq_id']), str(r['frame_id'])))
    if i is None or not len(c):
        continue
    j = int(np.argmax(c)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = p[j, :3], p[j, 3:6], p[j, 6:9], c[j]; nf += 1
m = PT.metrics(P, M, valid)
row = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], ASE=m['ASE'], AOE=m['AOE'], mDR=m['mDR'], n_frames_with_box=nf, n_valid=int(len(valid)))
d = np.linalg.norm(M['gt'][:, :3], axis=1)
for nm, lo, hi in PT.BINS:
    sel = valid[(d[valid] >= lo) & (d[valid] < hi)]; mb = PT.metrics(P, M, sel); row['AP_' + nm] = mb['AP']; row['UDS_' + nm] = mb['UDS']
json.dump(row, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('AP %.2f | ATE %.3f m | ASE %.2f cm | AOE %.2f | mDR %.1f%% | UDS %.2f  (frames with box %d / valid %d)' % (row['AP'], row['ATE'], row['ASE'], row['AOE'], row['mDR'], row['UDS'], nf, len(valid)))
