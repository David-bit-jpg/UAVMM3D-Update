# -*- coding: utf-8 -*-
"""A1 supplement: bias vs per-frame spread of each stage (train kept frames), and like-for-like frame sets (PL vs fused stage-3 target on the
gate-kept crops). Spread = 1.4826 * MAD(e - median e). Also the counterfactual 'PL with c_bar on the kept set' for both arms, and the
test-side +4.6% chain (from clean/diag/ate_diag.json). GT only for evaluation."""
import json, sys
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA')
from pa_common import *  # noqa


def sp(e):
    e = e[np.isfinite(e)]
    b = np.median(e)
    return dict(n=int(len(e)), bias=float(b), spread=float(1.4826 * np.median(np.abs(e - b))), med_abs_rel=float(np.median(np.abs(np.expm1(e)))))


GT = gt_index('train'); D0 = recs(X, 'det0')['recs']['train']
kept = [(r['seq'], r['frame']) for r in D0 if r['box'] is not None and r['conf'] >= CONF and (r['seq'], r['frame']) in GT]
z0 = {(r['seq'], r['frame']): float(r['box'][2]) for r in D0 if r['box'] is not None}
e_of = lambda Z, ks: np.array([np.log(Z[k] / GT[k][2]) if k in Z else np.nan for k in ks])
out = {}
for nm, ch in (('PST', X), ('RGB-only', XN)):
    U2 = {k: v['d'] for k, v in P(ch + '/ref/inf_st2_det0_train.pkl').items()}
    cb = cbar_of(D0, U2); dk = np.array([U2[k] for k in kept if k in U2]); cbk = float(np.median(dk[np.abs(dk) <= GUARD]))
    PL = {k: v[2] for k, v in pseudo_table(D0, U2, cb).items()}; PLk = {k: v[2] for k, v in pseudo_table(D0, U2, cbk).items()}
    meta_ = P(ch + '/ref/real_d0_meta.pkl'); dp = np.load(ch + '/ref/dp_fused.npy')
    T3 = {(m['seq'], m['frame']): m['z_det'] * np.exp(dp[i]) for i, m in enumerate(meta_) if np.isfinite(dp[i])}
    gk = [k for k in kept if k in T3]
    D1 = {(r['seq'], r['frame']): float(r['box'][2]) for r in recs(ch, 'det1')['recs']['train'] if r['box'] is not None}
    out[nm] = {'det0 (kept)': sp(e_of(z0, kept)), 'pseudo label (kept)': sp(e_of(PL, kept)), 'pseudo label, c_bar on kept set (counterfactual)': sp(e_of(PLk, kept)),
               'student det1 (kept)': sp(e_of(D1, kept)),
               'pseudo label (gate-kept crops)': sp(e_of(PL, gk)), 'fused stage-3 target (gate-kept crops)': sp(e_of(T3, gk)), 'det0 (gate-kept crops)': sp(e_of(z0, gk)),
               'c_bar': cb, 'c_bar_kept': cbk}
    for k, v in out[nm].items():
        if isinstance(v, dict):
            print('%-9s %-50s n %5d bias %+6.1f%% spread %5.1f%% med|rel| %5.1f%%' % (nm, k, v['n'], 100 * v['bias'], 100 * v['spread'], 100 * v['med_abs_rel']))
# same-frame comparison PST vs RGB-only on the PST gate-kept crops intersect RGB gate-kept crops
json.dump(out, open(PA + '/a1b_spread.json', 'w', encoding='utf-8'), indent=1, default=float)
A = json.load(open(R13 + '/clean/diag/ate_diag.json', encoding='utf-8'))
print(list(A.keys())[:20])
