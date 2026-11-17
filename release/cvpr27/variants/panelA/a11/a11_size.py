# -*- coding: utf-8 -*-
"""[panelA A11] Size prior: distance error a perfect size would give, size-bias sweep, and the two cancelling AntiUAV3D biases.
CPU only, existing predictions only; GT used only for evaluation (and for the oracle corrections, which are diagnostics, never deployed).
Distance decoding (C1_REPORT, object_encoder_mav6d): z_det = fx*span_x(s_bar, R_det)/w2d_hat. Per frame (input-px K_in from the recs):
  size_pure = log(span(s_bar, R_det) / span(s_gt, R_det))   at the GT centre: the pure size effect (only lwh changes)
  size_c1   = log(span(s_bar, R_det) / span(GT box, R_gt))   C1_REPORT definition (also absorbs the rotation mismatch)
  w2d       = log(span(GT box, R_gt) / w2d_hat)              width read error (negative = over-read)
Perfect size: t_final * exp(-size_pure) (same implied apparent width, GT size). Sweep: size = s_gt*(1+b) -> t * exp(-size_pure) * span(s_gt(1+b))/span(s_gt).
Positions scaled along the ray; predicted lwh/rotation/conf kept; re-scored with the pack metrics (score.py)."""
import os, json, pickle, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as Rot
os.environ['CUDA_VISIBLE_DEVICES'] = ''
OUT = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA'
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/'
P8 = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
DS = {
    'MAV6D': dict(meta='E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval/test_meta.npz', rules='real_test',
                  arms={'PST': ('in2mav', 'pcmd_fused_test', 'det1'), 'RGB-only': ('in2mav_none', 'pcmd_fused_test', 'det1'),
                        'Sim-only': ('in2mav', 'srconly_test', 'det0')}),
    'AntiUAV3D': dict(meta='D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz', rules='outdoor_test',
                      arms={'PST': ('out2antifs', 'pcmd_fused_test', 'det1'), 'RGB-only': ('out2antifs_none', 'pcmd_fused_test', 'det1'),
                            'Sim-only': ('out2antifs', 'srconly_test', 'det0')}),
}
BIAS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]


def span(t, lwh, eul, K):
    c = (P8 * lwh) @ Rot.from_euler('xyz', eul).as_matrix().T + t
    u = (K @ c.T).T; u = u[:, 0] / u[:, 2]
    return float(np.ptp(u))


def load(fn, M, key):
    N = len(M['seq'])
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(fn, 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def tp_mask(P, g, K):
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf']); tp = np.zeros(len(g), bool)
    for i in np.where(has)[0]:
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        tp[i] = S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) >= 0.1
    return tp


def scaled(P, f):
    Q = {k: v.copy() for k, v in P.items()}; ok = np.isfinite(f)
    Q['pt'][ok] = P['pt'][ok] * np.exp(f[ok])[:, None]
    return Q


def evaluate(P, g, K, c, sel_groups):
    """pack metrics on all frames + distance-error stats on the variant's own TP frames, per group"""
    tp = tp_mask(P, g, K); out = {}
    for gn, gsel in sel_groups.items():
        idx = np.where(gsel)[0]; m = S.metrics(P, g, K, idx, c)
        t = gsel & tp; dp = np.linalg.norm(P['pt'][t], axis=1); dg = np.linalg.norm(g[t, :3], axis=1); lr = np.log(dp / dg)
        out[gn] = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], n_tp=int(t.sum()), med_log=float(np.median(lr)) if t.any() else np.nan,
                       med_abs_rel=float(np.median(np.abs(dp / dg - 1))) if t.any() else np.nan,
                       mean_abs_rel=float(np.mean(np.abs(dp / dg - 1))) if t.any() else np.nan)
    return out


res = {}
for dn, D in DS.items():
    M = np.load(D['meta']); g = M['gt'].astype(np.float64); K = M['K']; N = len(g); c = S.CFG[D['rules']]
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    cls = np.array([str(x) for x in M['cls']]) if dn == 'AntiUAV3D' else np.array([str(s).split('/')[0] for s in M['seq']])
    groups = {'all': np.ones(N, bool)}
    for u in np.unique(cls):
        groups[u] = cls == u
    R = res[dn] = {'groups_n': {k: int(v.sum()) for k, v in groups.items()}, 'arms': {}}
    for an, (combo, pred, det) in D['arms'].items():
        P = load(X + combo + '/preds/' + pred + '.pkl', M, key)
        Dd = pickle.load(open(X + combo + '/recs/' + det + '.pkl', 'rb')); sbar = np.asarray(Dd['size'], np.float64)
        Rr = {(r['seq'], r['frame']): r for r in Dd['recs']['test'] if r['box'] is not None}
        sp, sc1, w2d = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan); sw = {b: np.full(N, np.nan) for b in BIAS}
        for i in np.where(np.isfinite(P['pt']).all(1))[0]:
            r = Rr.get((str(M['seq'][i]), str(M['frame'][i])))
            if r is None:
                continue
            Kin = np.asarray(r['K_in'], np.float64); t = g[i, :3]; Rd = np.asarray(r['box'], np.float64)[6:9]
            s_bar = span(t, sbar, Rd, Kin); s_gtR = span(t, g[i, 3:6], Rd, Kin); s_gt = span(t, g[i, 3:6], g[i, 6:9], Kin)
            sp[i] = np.log(s_bar / s_gtR); sc1[i] = np.log(s_bar / s_gt); w2d[i] = np.log(s_gt / r['w2d'])
            for b in BIAS:
                sw[b][i] = np.log(span(t, g[i, 3:6] * (1 + b), Rd, Kin) / s_gtR)
        tp0 = tp_mask(P, g, K)
        A = R['arms'][an] = dict(s_bar=sbar.tolist(), n_pred=int(np.isfinite(P['pt']).all(1).sum()),
                                 n_missing_rec=int((np.isfinite(P['pt']).all(1) & ~np.isfinite(sp)).sum()))
        # medians of the terms on the deployed TP frames (per group)
        dfin = np.log(np.linalg.norm(P['pt'], axis=1) / np.linalg.norm(g[:, :3], axis=1))
        A['terms_on_TP'] = {gn: dict(n=int((gs & tp0).sum()), size_pure=float(np.nanmedian(sp[gs & tp0])), size_c1=float(np.nanmedian(sc1[gs & tp0])),
                                     w2d=float(np.nanmedian(w2d[gs & tp0])), final=float(np.nanmedian(dfin[gs & tp0])),
                                     final_minus_size_c1=float(np.nanmedian((dfin - sc1)[gs & tp0])))
                            for gn, gs in groups.items() if (gs & tp0).any()}
        msp, mc1, mw = (float(np.nanmedian(v[tp0])) for v in (sp, sc1, w2d))
        A['global_medians_TP'] = dict(size_pure=msp, size_c1=mc1, w2d=mw)
        z = np.zeros(N)
        V = {'deployed': z,
             'perfect size (per-frame size_pure)': -sp,
             'remove size bias, per-frame (size_c1)': -sc1,
             'remove width over-read (global median w2d)': np.full(N, -mw),
             'remove both: per-frame size_c1 + global w2d': -sc1 - mw,
             'remove size bias, global median size_c1': np.full(N, -mc1),
             'remove both: global size_c1 + global w2d': np.full(N, -mc1 - mw),
             'remove own global bias (median final log, ref.)': np.full(N, -float(np.nanmedian(dfin[tp0])))}
        for b in BIAS:
            V['sweep b=%+d%%' % round(100 * b)] = -sp + sw[b]
        A['variants'] = {}
        for vn, f in V.items():
            f = np.where(np.isfinite(f), f, 0.0)   # frames without rec (none expected) keep their position
            A['variants'][vn] = evaluate(scaled(P, f), g, K, c, groups)
        print(dn, an, json.dumps({k: A[k] for k in ('s_bar', 'n_pred', 'n_missing_rec', 'global_medians_TP')}), flush=True)
        for vn, v in A['variants'].items():
            print('   %-48s UDS %.2f AP %.2f ATE %.4f medlog %+.3f med|rel| %.3f' % (vn, v['all']['UDS'], v['all']['AP'], v['all']['ATE'], v['all']['med_log'], v['all']['med_abs_rel']), flush=True)
json.dump(res, open(OUT + '/a11/a11_size.json', 'w'), indent=1, default=float)

# ---- figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
col = {'PST': '#1f77b4', 'RGB-only': '#ff7f0e', 'Sim-only': '#7f7f7f'}
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
for j, dn in enumerate(DS):
    for an in DS[dn]['arms']:
        Vv = res[dn]['arms'][an]['variants']
        xs = [100 * b for b in BIAS]
        ys = [100 * Vv['sweep b=%+d%%' % round(100 * b)]['all']['med_abs_rel'] for b in BIAS]
        ate = [Vv['sweep b=%+d%%' % round(100 * b)]['all']['ATE'] * (100 if dn == 'AntiUAV3D' else 1) for b in BIAS]
        dep = 100 * Vv['deployed']['all']['med_abs_rel']; depa = Vv['deployed']['all']['ATE'] * (100 if dn == 'AntiUAV3D' else 1)
        ax[0, j].plot(xs, ys, 'o-', color=col[an], label=an, ms=3); ax[0, j].axhline(dep, color=col[an], ls=':', lw=1)
        ax[1, j].plot(xs, ate, 'o-', color=col[an], label=an, ms=3); ax[1, j].axhline(depa, color=col[an], ls=':', lw=1)
    ax[0, j].set_title('%s test' % dn, fontsize=9)
    ax[0, j].set_ylabel('median |rel. distance error| (%)', fontsize=8)
    ax[1, j].set_ylabel('ATE (%s)' % ('% of range' if dn == 'AntiUAV3D' else 'm'), fontsize=8)
    ax[1, j].set_xlabel('size used for decoding: GT size + bias (%)', fontsize=8)
    for a in ax[:, j]:
        a.set_xticks([100 * b for b in BIAS]); a.tick_params(labelsize=7); a.grid(alpha=.3)
ax[0, 0].legend(fontsize=7, title='solid: GT size + bias; dotted: as deployed', title_fontsize=6)
fig.tight_layout(); fig.savefig(OUT + '/A11_size_sweep.png', dpi=200); fig.savefig(OUT + '/A11_size_sweep.pdf')
print('done')
