# -*- coding: utf-8 -*-
"""Review 11-13 (CPU, read-only): which agreement check picks better pseudo-orientations at a MATCHED frame count?
Same teacher readings (wf21/orient_uda/teach/teach_train.npz), same group fusion and PnP as pseudo_sets.py (t6il = final S6I labels).
Label = T6 (IRLiD s0,s1 + IR s0,s1 + LiD s0,s1) SA-TM -> PnP, exactly as pseudo_sets.py L46-62.
Agreement scores compared:
  d_IL   : IR-distilled pair vs LiDAR pair (the method's check)
  d_RR   : colour-only pair RGB s0,s1 vs colour-only pair RGB s2,s3 (seed-only diversity control)
  d_TR2  : T6 label vs colour-only pair RGB s0,s1 (the dropped second check)
  d_IRL_IR / d_IRL_LiD : IR+LiDAR pair vs IR pair / vs LiDAR pair
Sealed MAV6D train GT is read only for diagnosis (same as pseudo_sets.py --diag); nothing here feeds training or selection.
Output: agree_diag.json next to this script."""
import sys
sys.dont_write_bytecode = True
import os, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
OU = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/wf21/orient_uda'
sys.path.insert(0, OU)
import pseudo_build as PB   # noqa: E402
C1, S2, KFO = PB.C1, PB.S2, PB.KFO
OUT = os.path.dirname(os.path.abspath(__file__))

Z = dict(np.load(OU + '/teach/teach_train.npz', allow_pickle=False))
J = json.load(open(OU + '/teach/teach_train.json', encoding='utf-8'))
ix = {n: i for i, n in enumerate(J['member_names'])}
G = {'RGBa': ['RGB_s0', 'RGB_s1'], 'RGBb': ['RGB_s2', 'RGB_s3'], 'IRL2': ['IRLiD_s0', 'IRLiD_s1'], 'IR': ['IR_s0', 'IR_s1'], 'LiD': ['LiD_s0', 'LiD_s1']}
G = {k: [ix[n] for n in v] for k, v in G.items()}
G['T6'] = G['IRL2'] + G['IR'] + G['LiD']
G['T4'] = G['IR'] + G['LiD']
ks = float(J.get('KP_SCALE') or 64.0)
frames, _ = C1.load_dump_nogt('%s/tags/%s/real_ens5flip_%s.pkl' % (PB.REPRO, PB.TAG, PB.TAG))
import repl_common as RC   # noqa: E402
fr_gt, _ = RC.load_frames('%s/tags/%s/real_ens5flip_%s.pkl' % (PB.REPRO, PB.TAG, PB.TAG), '%s/tags/%s/sealed_gt_%s.pkl' % (PB.REPRO, PB.TAG, PB.TAG), with_gt=True)
keys = [(str(s), str(f)) for s, f in zip(Z['seq'], Z['frame'])]
rows = []
for i, k in enumerate(keys):
    fr = frames[k]
    t = C1.hook_top(Z, i, fr)
    if t < 0:
        continue
    lwh = np.asarray(fr['pred'][t, 3:6], np.float64)
    ctx = C1.make_ctx(Z, i, lwh, ks)
    Rdep = KFO.rot_from_eul(fr['pred'][t, 6:9])
    out = {}
    for g, mem in G.items():
        cell = ctx.cell_for(mem)
        o, fb, plan, (xs, Rs) = KFO.fuse_subset(ctx, mem, cell, ['C0', 'SATM'], Rdep)
        vec = KFO.satm_vec(xs, plan) if plan is not None else KFO.mean_vec(xs)
        out[g] = dict(R=o['SATM'], fb=bool(fb['SATM']), vec=vec)
    lab = {}
    for g in ('T6', 'T4', 'IRL2', 'RGBa'):
        rt = KFO.pnp(KFO.f32(out[g]['vec']), ctx.g, return_rt=True)
        lab[g] = None if (rt is None or out[g]['fb']) else rt[0]
    if lab['T6'] is None:
        continue
    r = dict(i=i, d_IL=PB.sym(out['IR']['R'], out['LiD']['R']), d_RR=PB.sym(out['RGBa']['R'], out['RGBb']['R']),
             d_TR2=PB.sym(lab['T6'], out['RGBa']['R']), d_IRL_IR=PB.sym(out['IRL2']['R'], out['IR']['R']),
             d_IRL_LiD=PB.sym(out['IRL2']['R'], out['LiD']['R']))
    g = np.asarray(fr_gt[k]['gt'], np.float64).reshape(-1, 9)
    tp = np.asarray(fr['pred'][t, :3], np.float64)
    if len(g):
        g = g[np.argmin(np.linalg.norm(g[:, :3] - tp, axis=1))]
        if np.linalg.norm(g[:3] - tp) <= 1.0:
            Rg = Rot.from_euler('xyz', g[6:9]).as_matrix()
            for name, Rl in lab.items():
                if Rl is not None:
                    r['geo_' + name] = PB.sym(Rl, Rg); r['yaw_' + name] = PB.sym_head(Rl, Rg); r['tilt_' + name] = PB.tilt(Rl, Rg)
            for name in ('IR', 'LiD', 'RGBb'):
                r['geo_' + name] = PB.sym(out[name]['R'], Rg); r['yaw_' + name] = PB.sym_head(out[name]['R'], Rg)
    rows.append(r)
    if i % 500 == 0:
        print(i, flush=True)

N = len(rows)
sc = [r for r in rows if 'geo_T6' in r]
A = lambda key, rr: np.array([r.get(key, np.nan) for r in rr], np.float64)
res = dict(n_rows=N, n_scored=len(sc))
dIL_all = A('d_IL', rows)
n_pass = int((dIL_all <= 10).sum())
res['n_pass_dIL10'] = n_pass


def stats(mask, rr, lab='T6'):
    e, y, tl = A('geo_' + lab, rr)[mask], A('yaw_' + lab, rr)[mask], A('tilt_' + lab, rr)[mask]
    ok = np.isfinite(e)
    e, y, tl = e[ok], y[ok], tl[ok]
    return dict(n=int(ok.sum()), geo_mean=float(e.mean()), geo_med=float(np.median(e)), yaw_mean=float(y.mean()), yaw_med=float(np.median(y)),
                tilt_med=float(np.median(tl)), frac_geo_gt30=float((e > 30).mean()), frac_yaw_gt20=float((y > 20).mean()))


allm = np.ones(len(sc), bool)
res['labels_all_frames'] = {lab: stats(np.isfinite(A('geo_' + lab, sc)), sc, lab) for lab in ('T6', 'T4', 'IRL2', 'RGBa', 'IR', 'LiD', 'RGBb')}
# matched count: in the full row set, keep the n_pass frames with the smallest score; then score the T6 label on scored ones
sel = {}
for key in ('d_IL', 'd_RR', 'd_TR2', 'd_IRL_IR', 'd_IRL_LiD'):
    s = A(key, rows)
    order = np.argsort(s, kind='stable')
    keep = np.zeros(N, bool); keep[order[:n_pass]] = True
    kept_idx = set(np.array([r['i'] for r in rows])[keep].tolist())
    m = np.array([r['i'] in kept_idx for r in sc])
    st = stats(m, sc); st['threshold_at_matched_count'] = float(s[order[n_pass - 1]])
    st['frac_at_10deg'] = float((s <= 10).mean())
    st['spearman_score_vs_T6geo'] = float(__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(A(key, sc), A('geo_T6', sc)).correlation)
    st['rest'] = stats(~m, sc)
    sel[key] = st
res['matched_count_selection_T6_label'] = sel
rng = np.random.default_rng(0)
rnd = []
for _ in range(200):
    keep = rng.choice(len(sc), int(round(n_pass * len(sc) / N)), replace=False)
    m = np.zeros(len(sc), bool); m[keep] = True
    e = A('geo_T6', sc)[m]; rnd.append(np.nanmean(e))
res['random_equal_count_geo_mean'] = dict(mean=float(np.mean(rnd)), p2_5=float(np.percentile(rnd, 2.5)), p97_5=float(np.percentile(rnd, 97.5)))
# union / intersection of the two checks at the same count
s1, s2 = A('d_IL', rows), A('d_RR', rows)
k1 = set(np.argsort(s1)[:n_pass]); k2 = set(np.argsort(s2)[:n_pass])
res['overlap_dIL_dRR_matched'] = len(k1 & k2) / n_pass
json.dump(res, open(OUT + '/agree_diag.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(res, indent=1))
import pickle
pickle.dump(dict(rows=rows, seq=[str(s) for s in Z['seq']]), open(OUT + '/agree_rows.pkl', 'wb'))
