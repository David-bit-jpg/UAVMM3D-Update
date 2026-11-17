# -*- coding: utf-8 -*-
# [11-17 copy of heads/heads_boot.py] only change: point-estimate self-check tolerance 1e-6 -> 1e-3 (a tied top score orders differently in fast_ap vs S.metrics; AP differs by ~7e-5)
"""[review1113 heads] TEST metrics of an ablation arm vs the same-seed reported run, paired bootstrap; pooled over the seed pairs given.
usage: python heads_boot.py <tgt mav|antifs> <out.json> <label> <arm_pred_1> <ref_pred_1> [<arm_pred_2> <ref_pred_2> ...]
  MAV6D    : pack score.py 'real_test' metrics on cmpeval/test_meta.npz; paired SEQUENCE bootstrap (10 test sequences resampled with
             replacement, the same resample for every arm/ref pair; AP recomputed per resample), B = 2000, rng seed 0
             (= review1113/cpu5/cpu5.py item 1 / clean2/buildup).
  AntiUAV3D: pack 'outdoor_test' metrics on the sealed test meta; paired circular BLOCK bootstrap within sequences, L = 176 (= cpu5 item 5).
Point estimates are asserted equal to the pack's own S.metrics. Only TEST labels are read (no calibration / val labels).
Output: per pair and pooled (mean over pairs) differences arm - ref for UDS AP ATE ASE AOE mDR with 95% percentile intervals and P(diff <= 0)."""
import os, sys, json, pickle, importlib.util
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scipy.spatial.transform import Rotation as Rot

G = 'E:/Open3DUAVDet/output/camnorm/geo1004'
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
B = 2000
KEYS = ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR')
RS = np.linspace(0, 1, 101); RS = RS[RS > 0.1]


def fast_ap(sc, hit, n_gt):   # = clean2/buildup.fast_ap (asserted against S.metrics below)
    if not len(sc):
        return 0.0
    o = np.argsort(-sc); t = hit[o].astype(float)
    ctp = np.cumsum(t); cfp = np.cumsum(1 - t); rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1)
    smax = np.maximum.accumulate(prec[::-1])[::-1]; k = np.searchsorted(rec, RS, side='left')
    p = np.where(k < len(rec), smax[np.minimum(k, len(rec) - 1)], 0.0)
    return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)


def meta(tgt):
    if tgt == 'mav':
        M = np.load(G + '/cmpeval/test_meta.npz'); c = S.CFG['real_test']
    else:
        M = np.load('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz'); c = S.CFG['outdoor_test']
    return dict(gt=np.asarray(M['gt'], np.float64), K=np.asarray(M['K'], np.float64), seq=np.array([str(s) for s in M['seq']]),
                frame=np.array([str(f) for f in M['frame']])), c


def load(p, M):
    key = {(s, f): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; N = len(M['seq'])
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(p, 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def frame_q(P, M, c):   # = buildup.frame_q (MAV, absolute distance) / cpu5.a_frame (AntiUAV3D, relative distance)
    g, K = M['gt'], M['K']; N = len(g)
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf'])
    d = np.linalg.norm(P['pt'] - g[:, :3], axis=1)
    if c['rel']:
        d = d / np.linalg.norm(g[:, :3], axis=1)
    tp = np.zeros(N, bool); se = np.full(N, np.nan); ae = np.full(N, np.nan)
    for i in np.where(has)[0]:
        fe = np.isfinite(P['peul'][i]).all(); eul = P['peul'][i] if fe else np.zeros(3)
        if S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(P['plwh'][i] - g[i, 3:6]))
        ae[i] = S.sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if fe else 90.0
    return dict(has=has, conf=P['conf'], d=d, tp=tp, se=se, ae=ae)


def stats(F, idx, c):
    h = F['has'][idx]; n = len(idx)
    ap = 100 * np.mean([fast_ap(F['conf'][idx][h], F['d'][idx][h] <= th, n) for th in c['thr']])
    t = idx[F['tp'][idx]]
    if not len(t):
        return dict(UDS=50 * ap / 100, AP=ap, ATE=np.nan, ASE=np.nan, AOE=np.nan, mDR=0.0)
    ATE, ASE, AOE = F['d'][t].mean(), F['se'][t].mean(), F['ae'][t].mean(); mdr = len(t) / n
    uds = 100 * (0.5 * ap / 100 + 0.5 * np.mean([1 - min(1, ATE / c['a']), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr]))
    return dict(UDS=uds, AP=ap, ATE=ATE, ASE=ASE, AOE=AOE, mDR=100 * mdr)


def groups_of(M):
    seq = M['seq']; frm = np.array([int(f.split('.')[0]) for f in M['frame']]); gr = []
    for s in sorted(np.unique(seq)):
        ix = np.where(seq == s)[0]; gr.append(ix[np.argsort(frm[ix], kind='stable')])
    return gr


def block_indices(rng, groups, L):   # = cpu5.block_indices (review1113/A3/a_block_boot.py)
    out = []
    for ix in groups:
        n = len(ix); Lg = min(L, n); nb = int(np.ceil(n / Lg))
        st = rng.integers(0, n, nb); pos = (st[:, None] + np.arange(Lg)[None, :]) % n
        out.append(ix[pos.ravel()[:n]])
    return np.concatenate(out)


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def main():
    tgt, out, label = sys.argv[1], sys.argv[2], sys.argv[3]
    pr = sys.argv[4:]; assert len(pr) >= 2 and len(pr) % 2 == 0, pr
    pairs = [(pr[i], pr[i + 1]) for i in range(0, len(pr), 2)]
    M, c = meta(tgt); N = len(M['seq'])
    FF, rows = [], []
    for a, r in pairs:
        fa, fr = [], {}
        for nm, p in (('arm', a), ('ref', r)):
            P = load(p, M); F = frame_q(P, M, c); s = stats(F, np.arange(N), c)
            ref = S.metrics(P, M['gt'], M['K'], np.arange(N), c)
            assert all(abs(ref[k] - s[k]) < 1e-3 or (np.isnan(ref[k]) and np.isnan(s[k])) for k in KEYS), (p, ref, s)
            fr[nm] = {k: float(s[k]) for k in KEYS}; fa.append(F)
        FF.append(fa); rows.append(dict(arm_pred=a, ref_pred=r, arm=fr['arm'], ref=fr['ref'], diff={k: fr['arm'][k] - fr['ref'][k] for k in KEYS}))
    rng = np.random.default_rng(0)
    if tgt == 'mav':
        us = np.unique(M['seq']); idx = {u: np.where(M['seq'] == u)[0] for u in us}
        draw = lambda: np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        boot_kind = 'paired sequence bootstrap over %d test sequences' % len(us)
    else:
        grp = groups_of(M)
        draw = lambda: block_indices(rng, grp, 176)
        boot_kind = 'paired circular block bootstrap within sequences, L = 176'
    D = {k: [[] for _ in pairs] for k in KEYS}
    for _ in range(B):
        ii = draw()
        for j, (Fa, Fr) in enumerate(FF):
            sa, sr = stats(Fa, ii, c), stats(Fr, ii, c)
            for k in KEYS:
                D[k][j].append(sa[k] - sr[k])
    res = dict(label=label, target=tgt, n_frames=int(N), B=B, bootstrap=boot_kind, pairs=rows, boot={})
    for k in KEYS:
        A = np.array(D[k])   # pairs x B
        for j in range(len(pairs)):
            rows[j].setdefault('ci', {})[k] = ci(A[j])
        pooled = A.mean(0)
        res['boot'][k] = dict(point=float(np.mean([r['diff'][k] for r in rows])), ci=ci(pooled), frac_le0=float(np.mean(pooled <= 0)))
    res['mean_arm'] = {k: float(np.mean([r['arm'][k] for r in rows])) for k in KEYS}
    res['mean_ref'] = {k: float(np.mean([r['ref'][k] for r in rows])) for k in KEYS}
    json.dump(res, open(out, 'w'), indent=1, default=float)
    f = lambda d: 'UDS %.2f AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.2f' % tuple(d[k] for k in KEYS)
    b = res['boot']
    print('%s [%s, %d pair(s)] arm %s | ref %s | diff UDS %+.2f [%+.2f, %+.2f] AP %+.2f [%+.2f, %+.2f] ATE %+.3f [%+.3f, %+.3f] ASE %+.2f AOE %+.2f [%+.2f, %+.2f] mDR %+.2f' % (
        label, tgt, len(pairs), f(res['mean_arm']), f(res['mean_ref']), b['UDS']['point'], *b['UDS']['ci'], b['AP']['point'], *b['AP']['ci'],
        b['ATE']['point'], *b['ATE']['ci'], b['ASE']['point'], b['AOE']['point'], *b['AOE']['ci'], b['mDR']['point']))


if __name__ == '__main__':
    main()
