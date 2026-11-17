# -*- coding: utf-8 -*-
"""Figure data (b), TEST version (11-14): the self-trained detector follows its pseudo labels on MAV6D TEST frames.
Identical pseudo-label rebuild to fig_student_follows.py / N1clean/n1c_analyze.py (train c_bar, lambda 0.75, conf>=0.5, |delta_hat|<=log1.4 guard,
delta_hat = mean of stage-2 refiners s5/s6 on det0 TEST crops from figdata/testgap/). Label-free headline: ALL confident test frames, no GT filter.
GT is read only afterwards for the post-hoc correlation and for the plotting column / 1 m flags (never used to select the headline set)."""
import os, pickle
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav'; TG = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/figdata/testgap'
REAL = 'E:/mmcache/mav6d_cn_obj'; OUT = 'D:/CVPR27_UAV/figures/data'
LAM, CONF, GUARD = 0.75, 0.5, float(np.log(1.4))
P = lambda f: pickle.load(open(f, 'rb'))


def avg(U):
    return {k: float(np.mean([u[k]['d'] for u in U])) for k in U[0] if all(k in u for u in U)}


def pl_table(recs, dh, cbar):
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
Ute = avg([P('%s/st2_s%d_det0_test.pkl' % (TG, j)) for j in (5, 6)])
Tte = pl_table(D0['test'], Ute, cbar)
S1 = {(r['seq'], r['frame']): (None if r['box'] is None else np.asarray(r['box'], np.float64)) for r in D1['test']}
S0 = {(r['seq'], r['frame']): (None if r['box'] is None else np.asarray(r['box'], np.float64)) for r in D0['test']}
conf0 = {(r['seq'], r['frame']): r['conf'] for r in D0['test']}
nan = float('nan')
rows = []
for k, t in Tte.items():                      # label-free set: every confident det0 test frame
    s = S1.get(k); s0 = S0[k]
    rows.append(dict(frame_id='%s/%s' % k, seq=k[0], frame=k[1], aircraft=k[0].split('/')[0],
                     pl_dist_m=t[2], st_dist_m=s[2] if s is not None else nan, start_dist_m=s0[2],
                     pl_range_m=np.linalg.norm(t[:3]), st_range_m=np.linalg.norm(s[:3]) if s is not None else nan, start_range_m=np.linalg.norm(s0[:3]),
                     pl_refined=bool(k in Ute and abs(Ute[k]) <= GUARD), det0_conf=float(conf0[k])))
da = pd.DataFrame(rows)
g = lambda a, b: 100 * np.median(np.abs(np.asarray(a) / np.asarray(b) - 1))
v = da[np.isfinite(da.st_dist_m)].reset_index(drop=True)
h_pl, h_st = g(v.st_dist_m, v.pl_dist_m), g(v.st_dist_m, v.start_dist_m)
# bootstrap over sequences (label-free)
rng = np.random.default_rng(0); seqs = v.seq.unique(); idx = {q: np.where(v.seq.values == q)[0] for q in seqs}
a_pl = np.abs(v.st_dist_m / v.pl_dist_m - 1).values; a_st = np.abs(v.st_dist_m / v.start_dist_m - 1).values
B = []
for _ in range(2000):
    ii = np.concatenate([idx[q] for q in rng.choice(seqs, len(seqs))])
    B.append((100 * np.median(a_pl[ii]), 100 * np.median(a_st[ii]), 100 * (np.median(a_st[ii]) - np.median(a_pl[ii]))))
B = np.array(B); ci = lambda c: (np.percentile(B[:, c], 2.5), np.percentile(B[:, c], 97.5))
nseq_win = sum(np.median(a_pl[idx[q]]) < np.median(a_st[idx[q]]) for q in seqs)
log = ['MAV6D TEST, label-free. c_bar (train) = %.5f; det0 test frames %d, confident (conf>=0.5) %d, refined by guard %d; det1 box present %d; sequences %d (%s)'
       % (cbar, len(D0['test']), len(da), int(da.pl_refined.sum()), len(v), v.seq.nunique(), v.groupby('aircraft').seq.nunique().to_dict()),
       'frames per aircraft: %s' % v.aircraft.value_counts().to_dict(),
       'median |st/pl - 1|    = %.2f%%  [seq-bootstrap 95%% CI %.2f, %.2f]' % ((h_pl,) + ci(0)),
       'median |st/start - 1| = %.2f%%  [seq-bootstrap 95%% CI %.2f, %.2f]' % ((h_st,) + ci(1)),
       'difference (start - pl) %.2f pts [CI %.2f, %.2f]; sequences where st is closer to pl than to start (per-seq median): %d/%d' % ((h_st - h_pl,) + ci(2) + (nseq_win, len(seqs)))]
# ---- post hoc: GT read only here ----
I = P(REAL + '/test/index.pkl')
GT = {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64)[0] for m in I['metas'] if m is not None and len(m['boxes9d'])}
gt = [GT.get((r.seq, r.frame)) for r in da.itertuples()]
da['gt_dist_m'] = [x[2] if x is not None else nan for x in gt]
da['gt_range_m'] = [np.linalg.norm(x[:3]) if x is not None else nan for x in gt]
w1 = lambda a, b: bool(np.isfinite(a).all() and np.isfinite(b).all() and np.linalg.norm(a - b) < 1.0)
pls = [Tte[(r.seq, r.frame)] for r in da.itertuples()]
sts = [S1.get((r.seq, r.frame)) for r in da.itertuples()]
da['posthoc_in_copy_set'] = [x is not None and s is not None and w1(p[:3], x[:3]) and w1(s[:3], x[:3]) for p, s, x in zip(pls, sts, gt)]
cols = ['frame_id', 'seq', 'frame', 'aircraft', 'pl_dist_m', 'st_dist_m', 'start_dist_m', 'gt_dist_m', 'pl_range_m', 'st_range_m', 'start_range_m', 'gt_range_m',
        'pl_refined', 'posthoc_in_copy_set', 'det0_conf']
da[cols].to_csv(OUT + '/student_follows_pl_mav6d_test.csv', index=False, float_format='%.6g')
for nm, d in (('all confident frames with GT and det1 box', da[np.isfinite(da.gt_dist_m) & np.isfinite(da.st_dist_m)]), ('posthoc 1 m set (old 803-style filter)', da[da.posthoc_in_copy_set])):
    log.append('POST HOC [%s] n=%d seq=%d: Pearson log(st/gt) vs log(pl/gt) = %.3f; median |rel| to GT: st %.2f%% pl %.2f%% start %.2f%%; |st/pl-1| %.2f%% |st/start-1| %.2f%%' % (
        nm, len(d), d.seq.nunique(), np.corrcoef(np.log(d.st_dist_m / d.gt_dist_m), np.log(d.pl_dist_m / d.gt_dist_m))[0, 1],
        g(d.st_dist_m, d.gt_dist_m), g(d.pl_dist_m, d.gt_dist_m), g(d.start_dist_m, d.gt_dist_m), g(d.st_dist_m, d.pl_dist_m), g(d.st_dist_m, d.start_dist_m)))
open(OUT + '/student_follows_pl_mav6d_test_check.log', 'w', encoding='utf-8').write('\n'.join(log) + '\n'); print('\n'.join(log))
