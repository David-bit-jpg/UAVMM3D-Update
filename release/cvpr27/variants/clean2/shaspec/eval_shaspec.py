# -*- coding: utf-8 -*-
"""[clean2 item 4] Evaluate ShaSpec rotations read on the clean det1 crops (shaspec_clean.py) against the clean PST orientation students. CPU only.
Frame set: the TP frames of the deployed clean PST (pcmd_fused; scorer rule top box, 2D IoU >= 0.1), identical for every arm.
Per frame: AOE term = quadrotor-symmetric geodesic (pack sym), yaw = min_k |yaw(R_gt^T R Rz(90k))| (tab_orient definition). A missing PnP counts 90 deg
in AOE (scorer rule) and is excluded from yaw. Also: all frames with a box (cmp_eval-style average over every frame), and the full scorer with the
ShaSpec rotation put into the PST boxes (UDS/AOE on its own TP set). Paired sequence bootstrap (B = 2000, seed 0) of ShaSpec - PST on the TP set."""
import os, sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'; X = G + '/gproto/srcx/in2mav'
sys.path.insert(0, G + '/gproto/review1113/clean2/buildup')
import buildup as BU  # noqa
S = BU.S


def per_frame(E, M, sel):
    g = M['gt']; ae = np.full(len(g), np.nan); hs = np.full(len(g), np.nan)
    for i in sel:
        Rg = Rot.from_euler('xyz', g[i, 6:9]).as_matrix()
        if np.isfinite(E[i]).all():
            Rp = Rot.from_euler('xyz', E[i]).as_matrix(); ae[i] = S.sym(Rp, Rg); hs[i] = min(BU.yawerr(Rp @ BU.RZ[q], Rg) for q in range(4))
        else:
            ae[i] = 90.0
    return ae, hs


def summ(ae, hs, sel):
    return dict(n=int(len(sel)), AOE=float(np.mean(ae[sel])), yaw_med=float(np.nanmedian(hs[sel])), yaw_mean=float(np.nanmean(hs[sel])),
                geo_med=float(np.median(ae[sel])), lt15=float(100 * np.mean(ae[sel] < 15)))


if __name__ == '__main__':
    res = {}
    for sp in ('val', 'test'):
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); N = len(M['seq']); seq = np.array([str(s) for s in M['seq']])
        key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
        Pf = BU.load_list(pickle.load(open('%s/preds/pcmd_fused_%s.pkl' % (X, sp), 'rb')), M)
        tp = np.where(BU.frame_q(Pf, M)['tp'])[0]; has = np.where(np.isfinite(Pf['pt']).all(1))[0]
        arms = {'PST students (S_ab0+S_ab1, + mirror)': Pf['peul']}
        for tag in ('raw', 'adabn'):
            for s in (0, 1):
                fn = '%s/pf_s%d_%s_%s.npz' % (OUT, s, sp, tag)
                if not os.path.exists(fn):
                    continue
                Z = np.load(fn); E = np.full((N, 3), np.nan)
                for q, f, e in zip(Z['seq'], Z['frame'], Z['eul']):
                    i = key.get((str(q), str(f)))
                    if i is not None:
                        E[i] = e
                arms['ShaSpec s%d %s' % (s, tag)] = E
        out = dict(n_tp=int(len(tp)), n_frames=int(N), arms={})
        PFQ = {}
        for nm, E in arms.items():
            ae, hs = per_frame(E, M, has); PFQ[nm] = (ae, hs)
            row = dict(tp_set=summ(ae, hs, tp), all_frames=summ(ae, hs, has))
            if nm.startswith('ShaSpec'):
                P = {k: v.copy() for k, v in Pf.items()}; P['peul'] = E.copy()
                m = S.metrics(P, M['gt'], M['K'], np.arange(N), BU.CFG); row['scorer_in_PST_boxes'] = {k: float(v) for k, v in m.items()}
            out['arms'][nm] = row
        # seed means
        for tag in ('raw', 'adabn'):
            ks = [k for k in arms if k.startswith('ShaSpec') and k.endswith(tag)]
            if len(ks) == 2:
                out['arms']['ShaSpec mean s0,s1 %s' % tag] = {part: {m: float(np.mean([out['arms'][k][part][m] for k in ks])) for m in ('AOE', 'yaw_med', 'yaw_mean', 'lt15')}
                                                               for part in ('tp_set', 'all_frames', 'scorer_in_PST_boxes') if part != 'scorer_in_PST_boxes'}
                out['arms']['ShaSpec mean s0,s1 %s' % tag]['scorer_in_PST_boxes'] = {m: float(np.mean([out['arms'][k]['scorer_in_PST_boxes'][m] for k in ks])) for m in ('UDS', 'AOE')}
        # paired sequence bootstrap on the TP set: ShaSpec (seed mean of per-frame AOE terms) - PST
        us = np.unique(seq[tp]); idx = {u: tp[seq[tp] == u] for u in us}; rng = np.random.default_rng(0); boot = {}
        a0, h0 = PFQ['PST students (S_ab0+S_ab1, + mirror)']
        for tag in ('raw', 'adabn'):
            ks = [k for k in arms if k.startswith('ShaSpec') and k.endswith(tag)]
            if not ks:
                continue
            aS = np.mean([PFQ[k][0] for k in ks], 0)
            dA, dY = [], []
            for _ in range(2000):
                ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
                dA.append(aS[ii].mean() - a0[ii].mean())
                dY.append(np.mean([np.nanmedian(PFQ[k][1][ii]) for k in ks]) - np.nanmedian(h0[ii]))
            boot[tag] = dict(dAOE=[float(np.mean(aS[tp]) - np.mean(a0[tp])), float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))],
                             dyaw_med=[float(np.mean([np.nanmedian(PFQ[k][1][tp]) for k in ks]) - np.nanmedian(h0[tp])), float(np.percentile(dY, 2.5)), float(np.percentile(dY, 97.5))])
        out['boot_shaspec_minus_pst'] = boot
        res[sp] = out
        print(sp, json.dumps(out, default=lambda x: round(float(x), 3)), flush=True)
    json.dump(res, open(OUT + '/eval_shaspec.json', 'w', encoding='utf-8'), indent=1, default=float)
