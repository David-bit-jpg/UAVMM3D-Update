# -*- coding: utf-8 -*-
"""Height-source ablation (review1113/height). CPU only, reuses existing prediction files; labels used only for scoring.
MAV6D: R = deployed nl075 chain (h = r_k * mean(l, w), r_k per brightness class); A = R with h := c * mean(l, w), c = s_bar aspect;
       B = FE_HTR=c full rerun of the chain (height also in the corner-PnP box -> PnP distance AND PnP rotation change);
       B' = B position + R rotation (only the PnP distance sees the constant height).
AntiUAV3D v2 (out2antifs): R = deployed pcmd_fused; A = h := c * mean(l, w), c = s_bar aspect 0.4433 (the IR-only/RGB-only rule).
       Corner PnP in srcx uses the s_bar box (oc_*.pkl 'size'), not r_k -> variant (b) is identical to (a).
Per-axis errors on the scorer's TP set (top-confidence box, 2D IoU >= 0.1), plus the 10-18 definition (median |dh| over all frames with a prediction)."""
import os, sys, json, pickle, importlib.util
import numpy as np
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'
OUT = G + '/gproto/review1113/height'
sys.path.insert(0, G + '/cmpeval')
import paper_table as PT  # noqa
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
C_MAV = 0.5717326440954079            # s_bar (0.34967, 0.34855, 0.19960): 0.19960 / mean(0.34967, 0.34855)
RNG = np.random.default_rng(0); B_BOOT = 2000


def per_frame(P, gt, K, valid, box2d, iou):
    N = len(gt); tp = np.zeros(N, bool); has = np.zeros(N, bool)
    for i in valid:
        if not (np.isfinite(P['pt'][i]).all() and np.isfinite(P['conf'][i])):
            continue
        has[i] = True
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        tp[i] = iou(box2d(P['pt'][i], P['plwh'][i], eul, K[i]), box2d(gt[i, :3], gt[i, 3:6], gt[i, 6:9], K[i])) >= 0.1
    e = 100 * np.abs(P['plwh'] - gt[:, 3:6])                       # cm, |dl| |dw| |dh|
    return dict(tp=tp, has=has, e=e, se=e.mean(1), rel_h=100 * e[:, 2] / (100 * gt[:, 5]))


def summ(F, sel):
    t = sel & F['tp']; a = sel & F['has']
    return dict(n_tp=int(t.sum()), ASE=F['se'][t].mean(), dl=F['e'][t, 0].mean(), dw=F['e'][t, 1].mean(), dh=F['e'][t, 2].mean(),
                dh_med=float(np.median(F['e'][t, 2])), dh_rel=F['rel_h'][t].mean(), dh_rel_med=float(np.median(F['rel_h'][t])),
                dh_med_all=float(np.median(F['e'][a, 2])), dh_mean_all=F['e'][a, 2].mean())   # *_all = 10-18 definition (all frames w/ prediction)


def boot(F0, F1, seq, sel, keys=('se', 'dh')):
    us = np.unique(seq[sel]); idx = {u: np.where(sel & (seq == u))[0] for u in us}
    out = {k: [] for k in keys}
    for _ in range(B_BOOT):
        ii = np.concatenate([idx[u] for u in RNG.choice(us, len(us))])
        for k in keys:
            col = (lambda F: F['se']) if k == 'se' else (lambda F: F['e'][:, 2])
            m1 = col(F1)[ii][F1['tp'][ii]].mean(); m0 = col(F0)[ii][F0['tp'][ii]].mean()
            out[k].append(m1 - m0)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def mav():
    res = {}
    for sp in ('val', 'test'):
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); valid = np.where(M['n_gt'] > 0)[0]
        gt = M['gt']; seq = np.array([str(s) for s in M['seq']])
        typ = np.where(np.isclose(gt[:, 5], 0.12, atol=0.01), 'Mavic 2', 'Phantom 4')
        R = pickle.load(open('%s/mav_rk/fe_deploy_fused_%s_P.pkl' % (OUT, sp), 'rb'))['P']
        Bf = pickle.load(open('%s/mav_htr/fe_deploy_fused_%s_P.pkl' % (OUT, sp), 'rb'))['P']
        assert np.allclose(np.nan_to_num(R['plwh'][:, :2]), np.nan_to_num(Bf['plwh'][:, :2])), 'l, w must be unchanged'
        A = {k: v.copy() for k, v in R.items()}; A['plwh'][:, 2] = C_MAV * A['plwh'][:, :2].mean(1)
        A2 = {k: v.copy() for k, v in R.items()}; A2['plwh'][:, 2] = 0.573 * A2['plwh'][:, :2].mean(1)
        Bp = {k: v.copy() for k, v in Bf.items()}; Bp['peul'] = R['peul'].copy()
        V = {'R_rk': R, 'A_const_box': A, 'A_const_0573': A2, 'B_const_box+pnp': Bf, 'Bp_const_box+pnpdist_rotR': Bp}
        ratio = R['plwh'][:, 2] / R['plwh'][:, :2].mean(1)
        res[sp] = {'deployed_ratio_by_type': {t: dict(frac_dark_0380=float(np.mean(np.isclose(ratio[(typ == t)], 0.38, atol=0.01))),
                                                       frac_bright_0711=float(np.mean(np.isclose(ratio[(typ == t)], 0.7111, atol=0.01))))
                                              for t in ('Mavic 2', 'Phantom 4')}}
        FF = {}
        for nm, P in V.items():
            m = PT.metrics(P, M, valid); F = per_frame(P, gt, M['K'], valid, PT.box2d, PT.iou); FF[nm] = F
            row = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], ASE=m['ASE'], AOE=m['AOE'], mDR=m['mDR'])
            vm = np.zeros(len(gt), bool); vm[valid] = True
            row['all'] = summ(F, vm)
            for t in ('Mavic 2', 'Phantom 4'):
                row[t] = summ(F, vm & (typ == t))
            res[sp][nm] = row
        vm = np.zeros(len(gt), bool); vm[valid] = True
        res[sp]['boot'] = {}
        for nm in ('A_const_box', 'B_const_box+pnp', 'Bp_const_box+pnpdist_rotR'):
            res[sp]['boot'][nm] = {g: boot(FF['R_rk'], FF[nm], seq, vm & (typ == g if g != 'all' else True)) for g in ('all', 'Mavic 2', 'Phantom 4')}
        res[sp]['n_seq'] = {g: int(len(np.unique(seq[vm & (typ == g if g != 'all' else True)]))) for g in ('all', 'Mavic 2', 'Phantom 4')}
    return res


def anti():
    Mt = np.load('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz')
    c = S.CFG['outdoor_test']; key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(Mt['seq'], Mt['frame']))}
    gt = Mt['gt']; N = len(gt); typ = np.where(np.isclose(gt[:, 5], 0.15, atol=0.01), 'M4E', 'large (0.98x0.76x0.48)')
    X = G + '/gproto/srcx'
    dep = json.load(open(X + '/out2antifs/deploy_pcmd.json')); sb = np.asarray(dep['s_bar']); cst = sb[2] / sb[:2].mean()

    def load(fn):
        P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
        for r in pickle.load(open(fn, 'rb')):
            i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores']).reshape(-1)
            if i is None or not len(b):
                continue
            j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
        return P
    R = load(X + '/out2antifs/preds/pcmd_fused_test.pkl')
    A = {k: v.copy() for k, v in R.items()}; A['plwh'][:, 2] = cst * A['plwh'][:, :2].mean(1)
    V = {'R_rk': R, 'A_const': A, 'IR-only (out2antifs_ir)': load(X + '/out2antifs_ir/preds/pcmd_fused_test.pkl'),
         'RGB-only (out2antifs_none)': load(X + '/out2antifs_none/preds/pcmd_fused_test.pkl')}
    res = dict(const=float(cst), s_bar=sb.tolist(), r_k=dep['r_k'], gt_sizes={t: gt[typ == t][0, 3:6].tolist() for t in np.unique(typ)},
               n={t: int((typ == t).sum()) for t in np.unique(typ)})
    allv = np.arange(N); vm = np.ones(N, bool)
    for nm, P in V.items():
        m = S.metrics(P, gt, Mt['K'], allv, c); F = per_frame(P, gt, Mt['K'], allv, S.box2d, S.iou)
        row = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], ASE=m['ASE'], AOE=m['AOE'], mDR=m['mDR'], all=summ(F, vm))
        for t in np.unique(typ):
            row[t] = summ(F, vm & (typ == t))
        t = F['tp']
        row['pred_mean_lwh_tp'] = np.nanmean(P['plwh'][t], 0).tolist(); row['pred_ratio_med'] = float(np.nanmedian(P['plwh'][t, 2] / P['plwh'][t, :2].mean(1)))
        res[nm] = row
    return res


if __name__ == '__main__':
    out = dict(mav=mav(), anti=anti())
    json.dump(out, open(OUT + '/height_results.json', 'w', encoding='utf-8'), indent=1, default=float, ensure_ascii=False)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 3), ensure_ascii=False))
