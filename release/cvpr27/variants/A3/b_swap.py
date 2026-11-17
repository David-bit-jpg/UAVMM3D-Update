# -*- coding: utf-8 -*-
"""A3-B step 2: swap sim-only height ratios into the reproduced deployed MAV6D run (review1113/height/mav_rk), height source only,
exactly as height_eval.py variant (a): h := r * mean(l, w); l, w, position, rotation, confidence unchanged.
Brightness class of each frame = the class the deployed chain gave it (read back from the deployed ratio 0.380 / 0.711).
Scoring: cmpeval/paper_table.metrics (= pack score.py real_test). Per-frame errors on the scorer's TP set (height_eval.per_frame / summ).
Sequence bootstrap (B = 2000, seed 0) of differences vs LiDAR r_k: ASE, mean |dh| (cm and % of GT h), UDS. CPU only."""
import os, sys, json, pickle, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as Rot
OUT = os.path.dirname(os.path.abspath(__file__))
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'
HE_DIR = G + '/gproto/review1113/height'
spec = importlib.util.spec_from_file_location('height_eval', HE_DIR + '/height_eval.py')
HE = importlib.util.module_from_spec(spec); spec.loader.exec_module(HE)
PT = HE.PT
SA = json.load(open(OUT + '/b_sim_aspect.json'))
B_BOOT = 2000
C_SBAR = HE.C_MAV
b0 = SA['b']['depth|real_col_c']; b1 = SA['b']['depth|sim_km2']
ALT = {   # name -> (r_dark, r_bright)
    's̄ constant aspect (no LiDAR, deployed IR/RGB-only rule)': (C_SBAR, C_SBAR),
    '(a) sim mean aspect, per object': (SA['a_per_object_mean'],) * 2,
    '(a) sim mean aspect, per model': (SA['a_per_model_mean'],) * 2,
    '(b) sim per-brightness-class, per object (real centres)': tuple(b0['r_dark_bright_per_object']),
    '(b) sim per-brightness-class, per model (real centres)': tuple(b0['r_dark_bright_per_model']),
    '(b) sim per-brightness-class, per model (sim k-means centres)': tuple(b1['r_dark_bright_per_model']),
}


def frame_q(P, M, valid):
    """per-frame has / conf / d / tp / se / ae / |dh|, same rules as PT.metrics."""
    g = M['gt']; K = M['K']; N = len(g)
    has = np.zeros(N, bool); tp = np.zeros(N, bool); se = np.full(N, np.nan); ae = np.full(N, np.nan)
    d = np.linalg.norm(P['pt'] - g[:, :3], axis=1)
    for i in valid:
        if not (np.isfinite(P['pt'][i]).all() and np.isfinite(P['conf'][i])):
            continue
        has[i] = True
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        if PT.iou(PT.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), PT.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(P['plwh'][i] - g[i, 3:6]))
        ae[i] = PT.sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if np.isfinite(P['peul'][i]).all() else 90.0
    dh = 100 * np.abs(P['plwh'][:, 2] - g[:, 5]); dhr = 100 * dh / (100 * g[:, 5])
    return dict(has=has, conf=P['conf'], d=d, tp=tp, se=se, ae=ae, dh=dh, dhr=dhr)


RS = np.linspace(0, 1, 101); RS = RS[RS > 0.1]


def fast_ap(sc, hit, n_gt):
    if not len(sc):
        return 0.0
    o = np.argsort(-sc); t = hit[o].astype(float)
    ctp = np.cumsum(t); cfp = np.cumsum(1 - t); rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1)
    smax = np.maximum.accumulate(prec[::-1])[::-1]; k = np.searchsorted(rec, RS, side='left')
    p = np.where(k < len(rec), smax[np.minimum(k, len(rec) - 1)], 0.0)
    return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)


def stats(F, idx):
    h = F['has'][idx]; n = len(idx)
    ap = 100 * np.mean([fast_ap(F['conf'][idx][h], F['d'][idx][h] <= th, n) for th in PT.THR])
    t = idx[F['tp'][idx]]
    ATE, ASE, AOE = F['d'][t].mean(), F['se'][t].mean(), F['ae'][t].mean(); mdr = len(t) / n
    uds = 100 * (0.5 * ap / 100 + 0.5 * np.mean([1 - min(1, ATE), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr]))
    return dict(UDS=uds, AP=ap, ASE=ASE, dh=F['dh'][t].mean(), dhr=F['dhr'][t].mean(), n_tp=int(len(t)))


if __name__ == '__main__':
    res = dict(alternatives={k: list(v) for k, v in ALT.items()}, splits={})
    for sp in ('val', 'test'):
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); valid = np.where(M['n_gt'] > 0)[0]
        gt = M['gt']; seq = np.array([str(s) for s in M['seq']])
        typ = np.where(np.isclose(gt[:, 5], 0.12, atol=0.01), 'Mavic 2', 'Phantom 4')
        R = pickle.load(open('%s/mav_rk/fe_deploy_fused_%s_P.pkl' % (HE_DIR, sp), 'rb'))['P']
        ratio = R['plwh'][:, 2] / R['plwh'][:, :2].mean(1)
        fin = np.isfinite(ratio)
        rk = (float(np.median(ratio[fin & (ratio < 0.55)])), float(np.median(ratio[fin & (ratio >= 0.55)])))
        cls = np.where(ratio < 0.55, 0, 1)
        off = fin & ~(np.isclose(ratio, rk[0], atol=1e-3) | np.isclose(ratio, rk[1], atol=1e-3))
        V = {'PST: LiDAR height ratio r_k': R}
        for nm, (rd, rb) in ALT.items():
            A = {k: v.copy() for k, v in R.items()}; A['plwh'][:, 2] = np.where(cls == 0, rd, rb) * A['plwh'][:, :2].mean(1); V[nm] = A
        vm = np.zeros(len(gt), bool); vm[valid] = True
        groups = {'all': vm, 'Mavic 2': vm & (typ == 'Mavic 2'), 'Phantom 4': vm & (typ == 'Phantom 4')}
        FF, rows = {}, {}
        for nm, P in V.items():
            m = PT.metrics(P, M, valid); F = frame_q(P, M, valid); FF[nm] = F
            chk = stats(F, valid); assert abs(chk['UDS'] - m['UDS']) < 1e-9 and abs(chk['ASE'] - m['ASE']) < 1e-9, (nm, chk, m)
            row = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], ASE=m['ASE'], AOE=m['AOE'], mDR=m['mDR'])
            for gname, gm in groups.items():
                ix = np.where(gm)[0]; s = stats(F, ix)
                row[gname] = dict(ASE=s['ASE'], dh=s['dh'], dhr=s['dhr'], n_tp=s['n_tp'], UDS_sub=s['UDS'])
            rows[nm] = row
        # sequence bootstrap of differences vs r_k (same resampled sequences for both arms)
        rng = np.random.default_rng(0); boot = {}
        for gname, gm in groups.items():
            us = np.unique(seq[gm]); idx = {u: np.where(gm & (seq == u))[0] for u in us}
            acc = {nm: {k: [] for k in ('ASE', 'dh', 'dhr', 'UDS')} for nm in ALT}
            for _ in range(B_BOOT):
                ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
                s0 = stats(FF['PST: LiDAR height ratio r_k'], ii)
                for nm in ALT:
                    s1 = stats(FF[nm], ii)
                    for k in acc[nm]:
                        acc[nm][k].append(s1[k] - s0[k])
            boot[gname] = {nm: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in a.items()} for nm, a in acc.items()}
            boot[gname]['n_seq'] = int(len(us))
        res['splits'][sp] = dict(rk_from_deployed=rk, n_off_class_frames=int(off.sum()), rows=rows, boot=boot,
                                 class_by_type={t: float(np.mean(cls[vm & (typ == t) & fin])) for t in ('Mavic 2', 'Phantom 4')})
        print(sp, json.dumps(res['splits'][sp], default=lambda x: round(float(x), 3), ensure_ascii=False)[:3000], flush=True)
    json.dump(res, open(OUT + '/b_swap.json', 'w', encoding='utf-8'), indent=1, default=float, ensure_ascii=False)
