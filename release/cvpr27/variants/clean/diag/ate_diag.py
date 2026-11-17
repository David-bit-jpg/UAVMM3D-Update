# -*- coding: utf-8 -*-
"""[clean item 4] Why is the clean chain's test ATE (0.306 m) worse than sim-only (0.283 m)? Diagnostic only (CPU, existing predictions;
labels read only to score; nothing here feeds back into any chain).
Per arm: e = log(z_pred / z_gt) on the scorer's TP set (top box, 2D IoU >= 0.1); global bias b = median e; spread = 1.4826 * median |e - b|;
the same per brightness class (clean chain's class: refiner silhouette brightness, nearest colour centre) and per aircraft (GT type).
De-biased rescoring: t := t * exp(-b) (global b of that arm on that split; diagnostic oracle, not deployable) -> AP / ATE / UDS via
cmpeval/paper_table.metrics (= pack real_test). Also ATE on the frames that are TP in BOTH arms (removes the detection-selection effect).
Stages of the clean chain (all single-frame, from its own files): det0 = sim-only ensemble (srconly) -> det1 = self-trained ensemble (recs/det1.pkl)
-> + refiner (pcmd_nofuse: t_det1 * exp(d_hat - c_k)) -> + corner fusion (pcmd_fused, w = 0.190, b_k on val images).
70.16 run (development-derived nl075): raw ensemble (nl075_ens/ens_<sp>.pkl) -> w = 0 (M12 repro_full) -> fused (M12 repro_full)."""
import os, sys, json, pickle
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'; X = G + '/gproto/srcx/in2mav'; M12 = G + '/gproto/review1113/M12/M1/repro_full'
sys.path.insert(0, G + '/cmpeval')
_argv = sys.argv; sys.argv = sys.argv[:1]
import paper_table as PT  # noqa
sys.argv = _argv
DEP = json.load(open(X + '/deploy_pcmd.json')); CC = np.asarray(DEP['col_centres']); CK = {int(k): v for k, v in DEP['c_k'].items()}
BK = {int(k): v for k, v in DEP['b_k_val_images'].items()}; W = DEP['w']; GUARD = float(np.log(1.4))
TYPES = ('Mavic 2', 'Phantom 4')


def empty(N):
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan); return P


def from_list(L, M):
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; P = empty(len(M['seq']))
    for r in L:
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores']).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def recs_list(recs):
    return [dict(seq=r['seq'], frame=r['frame'], boxes9d=np.asarray(r['box'], np.float64).reshape(1, 9), scores=np.array([r['conf']])) for r in recs if r['box'] is not None]


def ens_list(sp):
    out = []
    for r in pickle.load(open('%s/nl075_ens/ens_%s.pkl' % (G, sp), 'rb'))['records']:
        if len(r['pred']):
            out.append(dict(seq=r['seq_id'], frame=r['frame_id'], boxes9d=np.asarray(r['pred']).reshape(-1, 9), scores=np.asarray(r['conf']).reshape(-1)))
    return out


def tp_of(P, M, valid):
    g = M['gt']; K = M['K']; tp = np.zeros(len(g), bool)
    for i in valid:
        if not (np.isfinite(P['pt'][i]).all() and np.isfinite(P['conf'][i])):
            continue
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        tp[i] = PT.iou(PT.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), PT.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) >= 0.1
    return tp


def bias_spread(e):
    e = e[np.isfinite(e)]
    if not len(e):
        return dict(n=0)
    b = float(np.median(e)); return dict(n=int(len(e)), bias=b, spread=float(1.4826 * np.median(np.abs(e - b))), mean_abs=float(np.mean(np.abs(e))))


def score(P, M, valid):
    m = PT.metrics(P, M, valid); return {k: float(m[k]) for k in ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR')}


def scaled(P, s):
    Q = {k: v.copy() for k, v in P.items()}; Q['pt'] = Q['pt'] * (s[:, None] if np.ndim(s) else s); return Q


if __name__ == '__main__':
    res = {}
    for sp in ('val', 'test'):
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); valid = np.where(M['n_gt'] > 0)[0]; gt = M['gt']; N = len(gt)
        typ = np.where(np.isclose(gt[:, 5], 0.12, atol=0.01), 'Mavic 2', 'Phantom 4')
        key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
        U = pickle.load(open('%s/ref/inf_fin_det1_%s.pkl' % (X, sp), 'rb'))
        cls = np.full(N, -1); dhat = np.full(N, np.nan)
        for k, u in U.items():
            i = key.get((str(k[0]), str(k[1])))
            if i is None:
                continue
            dhat[i] = u['d']
            if u['lum'] is not None:
                cls[i] = int(np.argmin(np.abs(CC - u['lum'])))
        D1 = pickle.load(open(X + '/recs/det1.pkl', 'rb'))['recs'][sp]
        arms = {
            'clean: sim-only (det0, 3 seeds + flip)': from_list(pickle.load(open('%s/preds/srconly_%s.pkl' % (X, sp), 'rb')), M),
            'clean: self-trained detector (det1, raw)': from_list(recs_list(D1), M),
            'clean: + depth refiner (no fusion)': from_list(pickle.load(open('%s/preds/pcmd_nofuse_%s.pkl' % (X, sp), 'rb')), M),
            'clean: PST (+ corner fusion)': from_list(pickle.load(open('%s/preds/pcmd_fused_%s.pkl' % (X, sp), 'rb')), M),
            '70.16: self-trained detector (raw ens)': from_list(ens_list(sp), M),
            '70.16: + depth refiner (w = 0)': pickle.load(open('%s/fe_no_fusion_w0_%s_P.pkl' % (M12, sp), 'rb'))['P'],
            '70.16: PST (+ corner fusion)': pickle.load(open('%s/fe_deploy_fused_%s_P.pkl' % (M12, sp), 'rb'))['P'],
        }
        R = {}; TP = {}
        for nm, P in arms.items():
            tp = tp_of(P, M, valid); TP[nm] = tp
            e = np.log(P['pt'][:, 2] / gt[:, 2])
            row = dict(score=score(P, M, valid), all=bias_spread(e[tp]))
            row['by_class'] = {str(c): bias_spread(e[tp & (cls == c)]) for c in (0, 1, -1)}
            row['by_type'] = {t: bias_spread(e[tp & (typ == t)]) for t in TYPES}
            b = row['all']['bias']
            row['debias_global'] = dict(scale=float(np.exp(-b)), **score(scaled(P, np.exp(-b)), M, valid))
            s = np.ones(N)
            for t in TYPES:
                s[typ == t] = np.exp(-row['by_type'][t]['bias'])
            row['debias_per_aircraft'] = score(scaled(P, s), M, valid)
            R[nm] = row
        base = 'clean: sim-only (det0, 3 seeds + flip)'; common = {}
        d0 = np.linalg.norm(arms[base]['pt'] - gt[:, :3], axis=1)
        for nm, P in arms.items():
            both = TP[nm] & TP[base]; d = np.linalg.norm(P['pt'] - gt[:, :3], axis=1); e = np.log(P['pt'][:, 2] / gt[:, 2]); only = TP[nm] & ~TP[base]
            common[nm] = dict(n=int(both.sum()), ATE=float(d[both].mean()), ATE_simonly_same_frames=float(d0[both].mean()), bias=float(np.median(e[both])),
                              spread=float(1.4826 * np.median(np.abs(e[both] - np.median(e[both])))),
                              n_only_this=int(only.sum()), ATE_only_this=float(d[only].mean()) if only.any() else None)
        tpf = TP['clean: PST (+ corner fusion)']; st = {}
        z = {nm: arms[nm]['pt'][:, 2] for nm in arms}
        pairs = [('self-training: det1 / det0', 'clean: self-trained detector (det1, raw)', 'clean: sim-only (det0, 3 seeds + flip)'),
                 ('refiner applied: nofuse / det1 (= d_hat - c_k)', 'clean: + depth refiner (no fusion)', 'clean: self-trained detector (det1, raw)'),
                 ('fusion: fused / nofuse', 'clean: PST (+ corner fusion)', 'clean: + depth refiner (no fusion)'),
                 ('70.16 self-training: raw ens / det0', '70.16: self-trained detector (raw ens)', 'clean: sim-only (det0, 3 seeds + flip)'),
                 ('70.16 refiner: w0 / raw', '70.16: + depth refiner (w = 0)', '70.16: self-trained detector (raw ens)'),
                 ('70.16 fusion: fused / w0', '70.16: PST (+ corner fusion)', '70.16: + depth refiner (w = 0)'),
                 ('clean det1 / 70.16 raw ens', 'clean: self-trained detector (det1, raw)', '70.16: self-trained detector (raw ens)')]
        for lab, a, b_ in pairs:
            v = np.log(z[a] / z[b_]); ok = tpf & np.isfinite(v)
            st[lab] = dict(median=float(np.median(v[ok])), by_type={t: float(np.median(v[ok & (typ == t)])) for t in TYPES})
        ok = tpf & np.isfinite(dhat)
        st['refiner raw d_hat (before - c_k)'] = dict(median=float(np.median(dhat[ok])), by_class={str(c): float(np.median(dhat[ok & (cls == c)])) for c in (0, 1) if (ok & (cls == c)).any()},
                                                      c_k=CK, frac_guarded=float(np.mean(np.abs(dhat[ok]) > GUARD)))
        zn, zf = z['clean: + depth refiner (no fusion)'], z['clean: PST (+ corner fusion)']
        lzc = (np.log(zf) - (1 - W) * np.log(zn)) / W; fused_mask = np.abs(np.log(zf / zn)) > 1e-9
        e_c = lzc - np.log(gt[:, 2]); okc = tpf & fused_mask & np.isfinite(e_c)
        st['corner depth (log z_L - b_k) vs GT'] = dict(bs=bias_spread(e_c[okc]), by_type={t: bias_spread(e_c[okc & (typ == t)]) for t in TYPES},
                                                         by_class={str(c): bias_spread(e_c[okc & (cls == c)]) for c in (0, 1)}, b_k=BK, w=W)
        e_n = np.log(zn / gt[:, 2]); okn = okc & np.isfinite(e_n)
        st['corr(e_refined, e_corner)'] = float(np.corrcoef(e_n[okn], e_c[okn])[0, 1])
        res[sp] = dict(arms=R, common_tp_with_simonly=common, stages=st, n_valid=int(len(valid)))
        print(sp, json.dumps({k: (v['score'], v['all'], v['debias_global']) for k, v in R.items()}, default=lambda x: round(float(x), 4)), flush=True)
        print(json.dumps(common, default=lambda x: round(float(x), 4)), flush=True); print(json.dumps(st, default=lambda x: round(float(x), 4)), flush=True)
    json.dump(res, open(OUT + '/ate_diag.json', 'w', encoding='utf-8'), indent=1, default=float, ensure_ascii=False)
