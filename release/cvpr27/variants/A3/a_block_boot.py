# -*- coding: utf-8 -*-
"""A3-A: moving-block bootstrap (within-sequence, sequence strata kept) for AP / UDS on AntiUAV3D v2 test (antifs_test).
CPU only, existing predictions only. Per-frame quantities are precomputed with the pack scorer's own functions
(box2d / iou / sym from D:/UAV3D_baselines_pack/scripts/score.py); AP (vectorised nus_ap, checked equal) and UDS are recomputed on every resample."""
import os, json, pickle, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as Rot
OUT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx'
ARMS = {'PST': X + '/out2antifs/preds/pcmd_fused_test.pkl', 'Sim-only': X + '/out2antifs/preds/srconly_test.pkl',
        'RGB-only teachers': X + '/out2antifs_none/preds/pcmd_fused_test.pkl'}
M = np.load('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz')
c = S.CFG['outdoor_test']; g = M['gt']; K = M['K']; N = len(g)
seq = np.array([str(s) for s in M['seq']]); frm = np.array([int(str(f).split('.')[0]) for f in M['frame']])
key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
B = 2000


def load(fn):
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(fn, 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def per_frame(P):
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf'])
    d = np.linalg.norm(P['pt'] - g[:, :3], axis=1) / np.linalg.norm(g[:, :3], axis=1)
    tp = np.zeros(N, bool); se = np.full(N, np.nan); ae = np.full(N, np.nan)
    for i in np.where(has)[0]:
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        if S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(P['plwh'][i] - g[i, 3:6]))
        ae[i] = S.sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if np.isfinite(P['peul'][i]).all() else 90.0
    return dict(has=has, conf=P['conf'], d=d, tp=tp, se=se, ae=ae)


RS = np.linspace(0, 1, 101); RS = RS[RS > 0.1]


def fast_ap(sc, hit, n_gt):
    """== S.nus_ap (vectorised)."""
    if not len(sc):
        return 0.0
    o = np.argsort(-sc); t = hit[o].astype(float)
    ctp = np.cumsum(t); cfp = np.cumsum(1 - t); rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1)
    smax = np.maximum.accumulate(prec[::-1])[::-1]               # max precision over indices >= k (rec non-decreasing)
    k = np.searchsorted(rec, RS, side='left')                     # first index with rec >= r
    p = np.where(k < len(rec), smax[np.minimum(k, len(rec) - 1)], 0.0)
    return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)


def fast_metrics(F, idx):
    h = F['has'][idx]; n = len(idx)
    sc = F['conf'][idx][h]; d = F['d'][idx][h]
    ap = 100 * np.mean([fast_ap(sc, d <= th, n) for th in c['thr']])
    t = F['tp'][idx]
    if not t.any():
        return ap, 50 * ap / 100
    ATE = F['d'][idx][t].mean(); ASE = F['se'][idx][t].mean(); AOE = F['ae'][idx][t].mean(); mdr = t.sum() / n
    tps = [1 - min(1, ATE / c['a']), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr]
    return ap, 100 * (0.5 * ap / 100 + 0.5 * np.mean(tps))


def acf(x, maxlag=400):
    x = np.asarray(x, float); x = x - x.mean(); v = (x * x).mean()
    return np.array([1.0] + [(x[:-L] * x[L:]).mean() / v for L in range(1, maxlag + 1)])


def first_below(a, thr=0.1):
    w = np.where(a < thr)[0]
    return int(w[0]) if len(w) else None


def block_indices(rng, groups, L):
    """circular moving-block bootstrap within each sequence (stratum); each stratum keeps its own length."""
    out = []
    for ix in groups:
        n = len(ix); Lg = min(L, n); nb = int(np.ceil(n / Lg))
        st = rng.integers(0, n, nb)
        pos = (st[:, None] + np.arange(Lg)[None, :]) % n
        out.append(ix[pos.ravel()[:n]])
    return np.concatenate(out)


if __name__ == '__main__':
    FF = {}; point = {}
    for nm, fn in ARMS.items():
        P = load(fn); FF[nm] = per_frame(P)
        ref = S.metrics(P, g, K, np.arange(N), c); fa, fu = fast_metrics(FF[nm], np.arange(N))
        stored = json.load(open(fn + '.score_antifs_test.json'))['all']
        point[nm] = dict(AP=ref['AP'], UDS=ref['UDS'], fast_AP=fa, fast_UDS=fu, stored_AP=stored['AP'], stored_UDS=stored['UDS'],
                         ATE=ref['ATE'], ASE=ref['ASE'], AOE=ref['AOE'], mDR=ref['mDR'])
        assert abs(fa - ref['AP']) < 1e-9 and abs(fu - ref['UDS']) < 1e-9 and abs(ref['UDS'] - stored['UDS']) < 1e-9, (nm, fa, ref, stored)
    # sanity: fast metric == pack metric on a random block resample (duplicates)
    rng0 = np.random.default_rng(123)
    seqs = sorted(np.unique(seq)); groups = []
    for s in seqs:
        ix = np.where(seq == s)[0]; groups.append(ix[np.argsort(frm[ix], kind='stable')])
    idx0 = block_indices(rng0, groups, 50)
    for nm, fn in ARMS.items():
        ref = S.metrics(load(fn), g, K, idx0, c); fa, fu = fast_metrics(FF[nm], idx0)
        assert abs(fa - ref['AP']) < 1e-9 and abs(fu - ref['UDS']) < 1e-9, ('resample check', nm, fa, fu, ref)
    gaps = {s: np.unique(np.diff(np.sort(frm[seq == s])), return_counts=True) for s in seqs}
    big = groups[int(np.argmax([len(x) for x in groups]))]; ac = {}
    for nm, F in FF.items():
        o = big[F['has'][big]]; ot = big[F['tp'][big]]
        a_d = acf(F['d'][o]); a_o = acf(F['ae'][ot])
        ac[nm] = dict(dist_lag01=first_below(a_d), aoe_lag01=first_below(a_o), dist_lag1e=first_below(a_d, np.exp(-1)), aoe_lag1e=first_below(a_o, np.exp(-1)),
                      acf_dist_at_1_5_10_20_50_100_200=a_d[[1, 5, 10, 20, 50, 100, 200]].round(3).tolist(),
                      acf_aoe_at_1_5_10_20_50_100_200=a_o[[1, 5, 10, 20, 50, 100, 200]].round(3).tolist(),
                      n_pred=int(len(o)), n_tp=int(len(ot)))
    lags = [v for nm in ac for v in (ac[nm]['dist_lag01'], ac[nm]['aoe_lag01']) if v is not None]
    L_main = int(max(lags))
    res = dict(point=point, acf=ac, L_main=L_main, n_frames={s: int((seq == s).sum()) for s in seqs},
               frame_gaps={s: dict(zip(map(int, v[0]), map(int, v[1]))) for s, v in gaps.items()}, B=B, boot={})
    print(json.dumps({k: res[k] for k in ('point', 'acf', 'L_main', 'n_frames', 'frame_gaps')}, indent=1, default=lambda x: round(float(x), 4)), flush=True)
    pairs = [('PST', 'Sim-only'), ('PST', 'RGB-only teachers')]
    for L in sorted({L_main, 50, 100}):
        rng = np.random.default_rng(0)
        vals = {nm: np.zeros((B, 2)) for nm in FF}
        for b in range(B):
            idx = block_indices(rng, groups, L)
            for nm, F in FF.items():
                vals[nm][b] = fast_metrics(F, idx)
        r = {}
        for nm in FF:
            r[nm] = {m: [float(np.percentile(vals[nm][:, j], 2.5)), float(np.percentile(vals[nm][:, j], 97.5))] for j, m in enumerate(('AP', 'UDS'))}
            r[nm]['sd'] = vals[nm].std(0).tolist()
        for a, bb in pairs:
            dd = vals[a] - vals[bb]
            r['%s - %s' % (a, bb)] = {m: dict(point=point[a][m] - point[bb][m], ci=[float(np.percentile(dd[:, j], 2.5)), float(np.percentile(dd[:, j], 97.5))],
                                             frac_le0=float(np.mean(dd[:, j] <= 0))) for j, m in enumerate(('AP', 'UDS'))}
        res['boot'][str(L)] = r
        print('L', L, json.dumps(r, default=lambda x: round(float(x), 3)), flush=True)
    json.dump(res, open(OUT + '/a_block_boot.json', 'w'), indent=1, default=float)
