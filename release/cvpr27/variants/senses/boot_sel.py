# -*- coding: utf-8 -*-
"""Sequence-cluster bootstrap (2000, rng 0) of the difference in T6 pseudo-label error between matched-count selections (agree_rows.pkl)."""
import pickle, json, numpy as np
D = pickle.load(open('agree_rows.pkl', 'rb')); rows, seqall = D['rows'], D['seq']
N = len(rows); npass = sum(r['d_IL'] <= 10 for r in rows)
seq = np.array([seqall[r['i']] for r in rows]); e = np.array([r.get('geo_T6', np.nan) for r in rows]); y = np.array([r.get('yaw_T6', np.nan) for r in rows])
def keep(key):
    s = np.array([r[key] for r in rows]); m = np.zeros(N, bool); m[np.argsort(s, kind='stable')[:npass]] = True; return m
M = {k: keep(k) for k in ('d_IL', 'd_RR', 'd_TR2')}
us = np.unique(seq); idx = {u: np.where(seq == u)[0] for u in us}
rng = np.random.default_rng(0); out = {}
for a, b in (('d_IL', 'd_RR'), ('d_IL', 'd_TR2')):
    for nm, v in (('geo', e), ('yaw', y)):
        base = np.nanmean(v[M[a]]) - np.nanmean(v[M[b]]); bs = []
        for _ in range(2000):
            ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
            bs.append(np.nanmean(v[ii][M[a][ii]]) - np.nanmean(v[ii][M[b][ii]]))
        out['%s-%s_%s' % (a, b, nm)] = dict(diff=float(base), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
out['n_seq'] = int(len(us)); json.dump(out, open('boot_sel.json', 'w'), indent=1); print(json.dumps(out, indent=1))
