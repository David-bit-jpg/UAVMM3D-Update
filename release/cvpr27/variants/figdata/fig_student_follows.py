# -*- coding: utf-8 -*-
"""Figure data (b): the self-trained detector follows its pseudo labels (held-out MAV6D validation sequences; post-hoc, CPU, no inference).
Reuses review1113/N1clean/n1c_analyze.py exactly (same pseudo-label rebuild with the TRAIN constant c_bar, same 1 m filter); writes the 803 frames."""
import os, sys, pickle
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav'; N = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1clean'
REAL = 'E:/mmcache/mav6d_cn_obj'; OUT = 'D:/CVPR27_UAV/figures/data'
LAM, CONF, GUARD = 0.75, 0.5, float(np.log(1.4))
P = lambda f: pickle.load(open(f, 'rb'))


def avg(U):   # = srcx/lib/avg_inf.py
    return {k: float(np.mean([u[k]['d'] for u in U])) for k in U[0] if all(k in u for u in U)}


def pl_table(recs, dh, cbar):   # = make_pseudo_x.py rule (as in n1c_analyze.py)
    T = {}
    for r in recs:
        if r['box'] is None or r['conf'] < CONF:
            continue
        p = np.asarray(r['box'], np.float64).copy(); d = dh.get((r['seq'], r['frame']))
        if d is not None and abs(d) <= GUARD:
            p[:3] *= np.exp(d - LAM * cbar)
        T[(r['seq'], r['frame'])] = p
    return T


D0 = P(X + '/recs/det0.pkl')['recs']; D1 = P(X + '/recs/det1.pkl')['recs']
Utr = P(X + '/ref/inf_st2_det0_train.pkl')
dtr = np.array([Utr[(r['seq'], r['frame'])]['d'] for r in D0['train'] if (r['seq'], r['frame']) in Utr])
cbar = float(np.median(dtr[np.abs(dtr) <= GUARD]))
Uva = avg([P('%s/st2_s%d_det0_val.pkl' % (N, j)) for j in (5, 6)])
Tva = pl_table(D0['val'], Uva, cbar)
I = P(REAL + '/val/index.pkl')
GT = {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64)[0] for m in I['metas'] if m is not None and len(m['boxes9d'])}
S1 = {(r['seq'], r['frame']): (None if r['box'] is None else np.asarray(r['box'], np.float64)) for r in D1['val']}
S0 = {(r['seq'], r['frame']): (None if r['box'] is None else np.asarray(r['box'], np.float64)) for r in D0['val']}
rows = []
nan = float('nan')
w1 = lambda p, g: p is not None and bool(np.linalg.norm(p[:3] - g[:3]) < 1.0)
for k, t in Tva.items():            # all 835 confident frames; flags reproduce n1c_analyze.collect's filters
    g = GT.get(k); s = S1.get(k); s0 = S0[k]
    if g is None or not np.isfinite(g[2]):
        continue
    rows.append(dict(frame_id='%s/%s' % k, seq=k[0], frame=k[1], aircraft=k[0].split('/')[0],
                     pl_dist_m=t[2], st_dist_m=s[2] if s is not None else nan, start_dist_m=s0[2], gt_dist_m=g[2],
                     pl_range_m=np.linalg.norm(t[:3]), st_range_m=np.linalg.norm(s[:3]) if s is not None else nan, start_range_m=np.linalg.norm(s0[:3]), gt_range_m=np.linalg.norm(g[:3]),
                     in_copy_set_803=w1(t, g) and w1(s, g), in_start_ctrl_set_779=w1(s0, g) and w1(s, g)))
conf0 = {(r['seq'], r['frame']): r['conf'] for r in D0['val']}
for r in rows:
    r['det0_conf'] = float(conf0[(r['seq'], r['frame'])])
da = pd.DataFrame(rows); da.to_csv(OUT + '/student_follows_pl_mav6d.csv', index=False, float_format='%.6g')
df = da[da.in_copy_set_803]; c = da[da.in_start_ctrl_set_779]; v = da[np.isfinite(da.st_dist_m)]
g = lambda a, b: 100 * np.median(np.abs(a / b - 1))
log = ['all rows %d (confident frames with GT; student box present %d): median |st/pl-1| %.2f%%, |st/start-1| %.2f%%' % (len(da), len(v), g(v.st_dist_m, v.pl_dist_m), g(v.st_dist_m, v.start_dist_m)),'c_bar (train) = %.5f; pseudo labels (conf>=0.5) %d; rows %d, sequences %d (%s)' % (cbar, len(Tva), len(df), df.seq.nunique(), df.aircraft.value_counts().to_dict()),
       'median |st/pl - 1|      all %d rows: %.2f%%' % (len(df), g(df.st_dist_m, df.pl_dist_m)),
       'median |st/start - 1|   in_start_ctrl_set_779 (%d rows, = N1clean control): %.2f%%;  in_copy_set_803: %.2f%%' % (len(c), g(c.st_dist_m, c.start_dist_m), g(df.st_dist_m, df.start_dist_m)),
       'median |st/gt - 1| %.2f%%  |pl/gt - 1| %.2f%%  |start/gt - 1| %.2f%%' % (g(df.st_dist_m, df.gt_dist_m), g(df.pl_dist_m, df.gt_dist_m), g(df.start_dist_m, df.gt_dist_m)),
       'Pearson log(st/gt) vs log(pl/gt): %.3f' % np.corrcoef(np.log(df.st_dist_m / df.gt_dist_m), np.log(df.pl_dist_m / df.gt_dist_m))[0, 1]]
open(OUT + '/student_follows_pl_mav6d_check.log', 'w', encoding='utf-8').write('\n'.join(log) + '\n'); print('\n'.join(log))
