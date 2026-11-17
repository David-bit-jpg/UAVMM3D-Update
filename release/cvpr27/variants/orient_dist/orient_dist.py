# -*- coding: utf-8 -*-
"""[review1113 orient_dist] C1 orientation three ways + C2 AntiUAV3D test by distance band. CPU only, existing prediction pickles only.

Loading / TP rule = pack scorer (D:/UAV3D_baselines_pack/scripts/score.py via srcx/lib/score_x.py): per frame the top-scoring box;
TP = projected-2D IoU(pred box, GT box) >= 0.1. All orientation metrics are averaged over each arm's own TP frames (as pack AOE).
With Rp = Rot.from_euler('xyz', pred_eul), Rg = Rot.from_euler('xyz', gt_eul), RZ[q] = Rz(90 q deg), geo(A, B) = arccos((tr(A^T B) - 1) / 2):
  AOE90   = min_{q in 0..3} geo(Rp RZ[q], Rg)                (= pack sym, the reported AOE; asserted equal)
  AOE180  = min_{q in {0, 2}} geo(Rp RZ[q], Rg)              (2-fold: only a 180 deg turn about body z is free)
  HEAD    = |atan2(E[1,0], E[0,0])|, E = Rg^T Rp            (unfolded heading error in [0, 180]: yaw angle of the relative rotation
                                                             in the GT body frame, Z-Y-X decomposition; no symmetry, front/back counts)
  Frames with non-finite predicted Euler angles: pack gives AOE 90; we give AOE180 90 and HEAD 180 (counted; 0 occur here).
Bootstrap (paired: the same resample for both arms; B = 2000, default_rng(0)):
  MAV6D: whole-sequence bootstrap over the 10 test sequences (per aircraft: over that aircraft's sequences, Mavic 2 = 3, Phantom 4 = 7).
  AntiUAV3D: circular moving-block bootstrap within each sequence (strata kept, numeric frame order), L = 176 (= A3 L_main);
             AP / UDS recomputed with the A3 vectorised nus_ap (asserted == pack). Distance bands: frames of the band inside each resample."""
import os, sys, json, pickle, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as Rot
os.environ['CUDA_VISIBLE_DEVICES'] = ''
OUT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx'
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
B = 2000
SETS = {
    'mav': dict(meta='E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval/test_meta.npz', rules='real_test', stored='real_test',
                arms={'PST': X + '/in2mav/preds/pcmd_fused_test.pkl', 'RGB-only': X + '/in2mav_none/preds/pcmd_fused_test.pkl',
                      'Sim-only': X + '/in2mav/preds/srconly_test.pkl'},
                air={'Mavic 2': 'mavic2', 'Phantom 4': 'phantom4'}),
    'antifs': dict(meta='D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz', rules='outdoor_test', stored='antifs_test',
                   arms={'PST': X + '/out2antifs/preds/pcmd_fused_test.pkl', 'RGB-only': X + '/out2antifs_none/preds/pcmd_fused_test.pkl',
                         'Sim-only': X + '/out2antifs/preds/srconly_test.pkl'},
                   air={'Matrice 4E': 'M4E', 'Matrice 400': 'M400'}),
}
BANDS = ((0, 20), (20, 40), (40, 60), (60, 999))
RS = np.linspace(0, 1, 101); RS = RS[RS > 0.1]


def geo(A, Bm):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ Bm) - 1) / 2, -1, 1))))


def load(fn, M):
    N = len(M['seq']); key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(fn, 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def per_frame(P, g, K, rel):
    N = len(g); has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf'])
    d = np.linalg.norm(P['pt'] - g[:, :3], axis=1)
    if rel:
        d = d / np.linalg.norm(g[:, :3], axis=1)
    tp = np.zeros(N, bool); se = np.full(N, np.nan); a90 = np.full(N, np.nan); a180 = np.full(N, np.nan); hd = np.full(N, np.nan); nbad = 0
    for i in np.where(has)[0]:
        fin = np.isfinite(P['peul'][i]).all(); eul = P['peul'][i] if fin else np.zeros(3)
        if S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(P['plwh'][i] - g[i, 3:6]))
        if not fin:
            a90[i], a180[i], hd[i] = 90.0, 90.0, 180.0; nbad += 1; continue
        Rp = Rot.from_euler('xyz', eul).as_matrix(); Rg = Rot.from_euler('xyz', g[i, 6:9]).as_matrix()
        a90[i] = S.sym(Rp, Rg)
        a180[i] = min(geo(Rp @ RZ[q], Rg) for q in (0, 2))
        E = Rg.T @ Rp; hd[i] = abs(float(np.degrees(np.arctan2(E[1, 0], E[0, 0]))))
    return dict(has=has, conf=P['conf'], d=d, tp=tp, se=se, a90=a90, a180=a180, hd=hd, nbad=nbad)


def fast_ap(sc, hit, n_gt):   # == S.nus_ap (A3 vectorised form)
    if not len(sc):
        return 0.0
    o = np.argsort(-sc); t = hit[o].astype(float)
    ctp = np.cumsum(t); cfp = np.cumsum(1 - t); rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1)
    smax = np.maximum.accumulate(prec[::-1])[::-1]
    k = np.searchsorted(rec, RS, side='left')
    p = np.where(k < len(rec), smax[np.minimum(k, len(rec) - 1)], 0.0)
    return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)


def fm(F, idx, c, det=True):
    """metrics on the (resampled) frame index list idx. det=False skips AP (orientation-only bootstrap)."""
    n = len(idx); t = F['tp'][idx]; o = {}
    if det:
        h = F['has'][idx]; sc = F['conf'][idx][h]; d = F['d'][idx][h]
        o['AP'] = 100 * np.mean([fast_ap(sc, d <= th, n) for th in c['thr']])
    if not t.any():
        o.update(AOE90=np.nan, AOE180=np.nan, HEADmean=np.nan, HEADmed=np.nan, AOE90med=np.nan, AOE180med=np.nan, ATE=np.nan, mDR=0.0)
        if det:
            o['UDS'] = 50 * o['AP'] / 100
        return o
    a90 = F['a90'][idx][t]; a180 = F['a180'][idx][t]; hd = F['hd'][idx][t]
    o.update(AOE90=a90.mean(), AOE90med=np.median(a90), AOE180=a180.mean(), AOE180med=np.median(a180), HEADmean=hd.mean(), HEADmed=np.median(hd),
             ATE=F['d'][idx][t].mean(), mDR=100 * t.sum() / n)
    if det:
        ASE = F['se'][idx][t].mean()
        tps = [1 - min(1, o['ATE'] / c['a']), 1 - min(1, ASE / 10.0), 1 - min(1, o['AOE90'] / 57.2958), t.sum() / n]
        o['UDS'] = 100 * (0.5 * o['AP'] / 100 + 0.5 * np.mean(tps)); o['ASE'] = ASE
    return o


def block_indices(rng, groups, L):   # = A3 a_block_boot.block_indices
    out = []
    for ix in groups:
        n = len(ix); Lg = min(L, n); nb = int(np.ceil(n / Lg))
        st = rng.integers(0, n, nb)
        pos = (st[:, None] + np.arange(Lg)[None, :]) % n
        out.append(ix[pos.ravel()[:n]])
    return np.concatenate(out)


def seq_indices(rng, groups):
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[k] for k in pick])


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


OM = ('AOE90', 'AOE180', 'HEADmean', 'HEADmed')
res = {'C1': {}, 'C2': {}}
for tgt, cfg in SETS.items():
    M = np.load(cfg['meta']); g = M['gt']; K = M['K']; N = len(g); c = S.CFG[cfg['rules']]
    seq = np.array([str(s) for s in M['seq']]); cls = np.array([str(x) for x in M['cls']])
    frm = np.array([int(str(f).split('.')[0]) for f in M['frame']])
    FF = {}
    for arm, fn in cfg['arms'].items():
        P = load(fn, M); FF[arm] = F = per_frame(P, g, K, c['rel'])
        ref = S.metrics(P, g, K, np.arange(N), c); me = fm(F, np.arange(N), c)
        stored = json.load(open(fn + '.score_%s.json' % cfg['stored']))['all']
        for k in ('AP', 'UDS', 'ATE'):
            assert abs(me[k] - ref[k]) < 1e-9, (tgt, arm, k, me[k], ref[k])
        assert abs(me['AOE90'] - ref['AOE']) < 1e-9 and abs(ref['UDS'] - stored['UDS']) < 1e-9, (tgt, arm, me, ref, stored)
        print(tgt, arm, 'checked == pack / stored; non-finite euler on TP:', F['nbad'], flush=True)
    subs = {'all': np.arange(N)}
    subs.update({a: np.where(cls == code)[0] for a, code in cfg['air'].items()})
    res['C1'][tgt] = {}
    for sub, sel in subs.items():
        # resampling units inside this subset
        if tgt == 'mav':
            groups = [sel[seq[sel] == s] for s in sorted(np.unique(seq[sel]))]
            draw = lambda rng: seq_indices(rng, groups)
            how = 'sequence bootstrap over %d sequences' % len(groups)
        else:
            groups = []
            for s in sorted(np.unique(seq[sel])):
                ix = sel[seq[sel] == s]; groups.append(ix[np.argsort(frm[ix], kind='stable')])
            draw = lambda rng: block_indices(rng, groups, 176)
            how = 'block bootstrap L=176 within %d sequence(s), %s frames' % (len(groups), '/'.join(str(len(x)) for x in groups))
        R = {arm: fm(F, sel, c, det=False) for arm, F in FF.items()}
        for arm, F in FF.items():
            R[arm]['n_tp'] = int(F['tp'][sel].sum())
        R['n'] = int(len(sel)); R['resampling'] = how
        common = sel[FF['PST']['tp'][sel] & FF['RGB-only']['tp'][sel]]
        R['common_tp'] = dict(n=int(len(common)), **{'%s %s' % (arm, m): float(v) for arm in ('PST', 'RGB-only') for m, v in fm(FF[arm], common, c, det=False).items() if m in OM})
        rng = np.random.default_rng(0); bs = {p: {m: [] for m in OM} for p in ('PST - RGB-only', 'PST - Sim-only')}
        for b in range(B):
            idx = draw(rng); v = {arm: fm(F, idx, c, det=False) for arm, F in FF.items()}
            for m in OM:
                bs['PST - RGB-only'][m].append(v['PST'][m] - v['RGB-only'][m]); bs['PST - Sim-only'][m].append(v['PST'][m] - v['Sim-only'][m])
        for p, ob in (('PST - RGB-only', 'RGB-only'), ('PST - Sim-only', 'Sim-only')):
            R[p] = {m: dict(point=float(R['PST'][m] - R[ob][m]), ci=ci(bs[p][m]), frac_ge0=float(np.nanmean(np.array(bs[p][m]) >= 0))) for m in OM}
        res['C1'][tgt][sub] = R
        print('C1', tgt, sub, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in R.items()}, default=str), flush=True)
    if tgt != 'antifs':
        continue
    # ---------------- C2: distance bands (range |t_gt|, as the pack's rel=True bins)
    rngd = np.linalg.norm(g[:, :3], axis=1)
    groups = []
    for s in sorted(np.unique(seq)):
        ix = np.where(seq == s)[0]; groups.append(ix[np.argsort(frm[ix], kind='stable')])
    inband = {b: (rngd >= b[0]) & (rngd < b[1]) for b in BANDS}
    P0 = {arm: load(fn, M) for arm, fn in cfg['arms'].items()}
    for b in BANDS:
        sel = np.where(inband[b])[0]; R = {'n': int(len(sel)), 'n_M4E': int((cls[sel] == 'M4E').sum()), 'n_M400': int((cls[sel] == 'M400').sum())}
        for arm, F in FF.items():
            ref = S.metrics(P0[arm], g, K, sel, c); me = fm(F, sel, c)   # pack per-band logic (metrics on the band's frames)
            assert abs(me['AP'] - ref['AP']) < 1e-9 and abs(me['UDS'] - ref['UDS']) < 1e-9
            R[arm] = dict(AP=ref['AP'], ATE=ref['ATE'], ASE=ref['ASE'], AOE=ref['AOE'], mDR=ref['mDR'], UDS=ref['UDS'], n_tp=int(F['tp'][sel].sum()))
        res['C2'][('%g-%g' % b).replace('-999', '+')] = R
    rng = np.random.default_rng(0); bs = {('%g-%g' % b).replace('-999', '+'): {m: [] for m in ('AP', 'UDS', 'ATE', 'AOE90')} for b in BANDS}
    for it in range(B):
        idx = block_indices(rng, groups, 176)
        for b in BANDS:
            ii = idx[inband[b][idx]]; v = {arm: fm(FF[arm], ii, c) for arm in ('PST', 'RGB-only')}
            for m in ('AP', 'UDS', 'ATE', 'AOE90'):
                bs[('%g-%g' % b).replace('-999', '+')][m].append(v['PST'][m] - v['RGB-only'][m])
    for bn, v in bs.items():
        R = res['C2'][bn]
        R['PST - RGB-only'] = {m: dict(point=float(R['PST'][m if m != 'AOE90' else 'AOE'] - R['RGB-only'][m if m != 'AOE90' else 'AOE']), ci=ci(v[m])) for m in v}
        print('C2', bn, json.dumps(R, default=lambda x: round(float(x), 3)), flush=True)
json.dump(res, open(OUT + '/orient_dist.json', 'w'), indent=1, default=float)
print('wrote', OUT + '/orient_dist.json')
