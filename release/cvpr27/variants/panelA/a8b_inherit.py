# -*- coding: utf-8 -*-
"""A8 supplement: how much of the pseudo label's per-frame correction does the student inherit on held-out val?
e_student = a + b0 * e_det0 + b1 * c,  c = log(z_PL / z_det0) = applied label correction (delta_hat - lam c_bar, or 0 outside the guard).
b1 = 1: the student reproduces the label's frame-by-frame correction; b1 = 0: it ignores it. Also the same regression after per-sequence demeaning.
No 1 m filter; per airframe; paired sequence bootstrap (B = 2000, seed 0). GT only for evaluation."""
import json, sys
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA')
from pa_common import *  # noqa
rng = np.random.default_rng(0)
GT = gt_index('train')  # unused; keeps the helper import symmetric
GTva = gt_index('val'); D0 = recs(X, 'det0'); cb = cbar_of(D0['recs']['train'], {k: v['d'] for k, v in P(X + '/ref/inf_st2_det0_train.pkl').items()})
Uva = avg([P('%s/N1clean/st2_s%d_det0_val.pkl' % (R13, j)) for j in (5, 6)])
PL = pseudo_table(D0['recs']['val'], Uva, cb)
z0 = {(r['seq'], r['frame']): float(r['box'][2]) for r in D0['recs']['val'] if r['box'] is not None}
z1 = {(r['seq'], r['frame']): float(r['box'][2]) for r in recs(X, 'det1')['recs']['val'] if r['box'] is not None}
ks = [k for k in PL if k in GTva and k in z1 and np.isfinite(GTva[k][2])]
es = np.array([np.log(z1[k] / GTva[k][2]) for k in ks]); e0 = np.array([np.log(z0[k] / GTva[k][2]) for k in ks])
c = np.array([np.log(PL[k][2] / z0[k]) for k in ks]); sq = np.array([k[0] for k in ks]); air = np.array([k[0].split('/')[0] for k in ks])


def dm(x, s):
    y = x.copy()
    for u in np.unique(s):
        m = s == u; y[m] = x[m] - x[m].mean()
    return y


def fit(ii):
    A = np.c_[np.ones(len(ii)), e0[ii], c[ii]]; b = np.linalg.lstsq(A, es[ii], rcond=None)[0]
    lab = np.repeat(np.arange(len(ii)), 1)  # placeholder
    return b[1], b[2]


def fit_dm(ii, lab):
    y, a0, a1 = dm(es[ii], lab), dm(e0[ii], lab), dm(c[ii], lab)
    b = np.linalg.lstsq(np.c_[a0, a1], y, rcond=None)[0]
    return b[0], b[1]


res = dict(c_bar=cb, n=len(ks))
for grp in ('all', 'phantom4', 'mavic2'):
    sel = np.where((air == grp) | (grp == 'all'))[0]
    us = np.unique(sq[sel]); idx = {u: sel[sq[sel] == u] for u in us}
    p = fit(sel); pd = fit_dm(sel, sq[sel]); bb, bd = [], []
    for _ in range(2000):
        pick = rng.choice(us, len(us)); ii = np.concatenate([idx[u] for u in pick])
        lab = np.concatenate([np.full(len(idx[u]), t) for t, u in enumerate(pick)])
        bb.append(fit(ii)); bd.append(fit_dm(ii, lab))
    bb, bd = np.array(bb), np.array(bd)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res[grp] = dict(n=int(len(sel)), n_seq=int(len(us)), b_det0=float(p[0]), b_det0_ci=q(bb[:, 0]), b_corr=float(p[1]), b_corr_ci=q(bb[:, 1]),
                    dm_b_det0=float(pd[0]), dm_b_det0_ci=q(bd[:, 0]), dm_b_corr=float(pd[1]), dm_b_corr_ci=q(bd[:, 1]),
                    sd_correction=float(np.std(c[sel])), frac_corrected=float(np.mean(c[sel] != 0)))
    print(grp, json.dumps(res[grp], default=lambda x: round(float(x), 3)), flush=True)
json.dump(res, open(PA + '/a8b_inherit.json', 'w'), indent=1, default=float)
