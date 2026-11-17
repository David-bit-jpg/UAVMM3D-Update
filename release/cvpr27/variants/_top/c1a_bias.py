# -*- coding: utf-8 -*-
"""[review1113 C1a] signed distance / size bias of final PST vs source-only on AntiUAV3D v2 test (CPU only, read-only).
Matching = exactly score_x.score_meta -> pack score.metrics: top-1 box per frame, TP if projected-2D-box IoU >= 0.1."""
import importlib.util, json, pickle, sys
import numpy as np
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
META = 'D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz'
GTP = 'D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/test_gt.pkl'
PD = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs/preds/'
PREDS = {'PST (pcmd_fused_test)': PD + 'pcmd_fused_test.pkl', 'source-only (srconly_test)': PD + 'srconly_test.pkl'}
M = np.load(META); N = len(M['seq']); g = M['gt']; K = M['K']; c = S.CFG['outdoor_test']
key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
GT = pickle.load(open(GTP, 'rb')); assert all(r['seq'] == s and r['frame'] == f for r, s, f in zip(GT, M['seq'], M['frame']))
fx = K[:, 0, 0]; cls = M['cls']; seq = M['seq']; dist = np.linalg.norm(g[:, :3], axis=1)


def load(pred):   # = score_meta
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(pred, 'rb')):
        i = key.get((str(r['seq']), str(r['frame'])))
        b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def tp_mask(P):   # = pack metrics() TP rule
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf']); tp = np.zeros(N, bool)
    for i in np.where(has)[0]:
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        tp[i] = S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) >= 0.1
    return tp


def stats(sel, P):
    if not sel.sum():
        return dict(n=0)
    dp = np.linalg.norm(P['pt'][sel], axis=1); dg = dist[sel]; lr = np.log(dp / dg)
    zr = np.log(P['pt'][sel, 2] / g[sel, 2])
    return dict(n=int(sel.sum()), mean_log=float(lr.mean()), median_log=float(np.median(lr)), med_abs_rel=float(np.median(np.abs(dp / dg - 1))),
                mean_abs_rel=float(np.mean(np.abs(dp / dg - 1))), mean_logz=float(zr.mean()),
                frac_over=float((lr > 0).mean()))


def sizestats(sel, P):
    r = np.log(P['plwh'][sel] / g[sel, 3:6])
    return dict(n=int(sel.sum()), med_log_l=float(np.median(r[:, 0])), med_log_w=float(np.median(r[:, 1])), med_log_h=float(np.median(r[:, 2])),
                mean_log_l=float(r[:, 0].mean()), mean_log_w=float(r[:, 1].mean()), mean_log_h=float(r[:, 2].mean()),
                pred_lwh_median=np.median(P['plwh'][sel], 0).round(3).tolist(), gt_lwh=np.median(g[sel, 3:6], 0).round(3).tolist())


DB = [(0, 20), (20, 40), (40, 60), (60, 80)]
# fx takes only 3 distinct values on this test split (1908.7 / 3393.1 / 3694.7 px), so terciles are degenerate -> natural fx groups
FB = [('fx=%.0f px' % v, np.isclose(fx, v, atol=1.0)) for v in sorted(set(np.round(fx, 1)))]
out = {}
for name, path in PREDS.items():
    P = load(path); tp = tp_mask(P)
    rec = {'score': {k: {m: float(v) for m, v in S.metrics(P, g, K, np.arange(N), c).items()} for k in ['all']}, 'n_tp': int(tp.sum())}
    rec['all'] = stats(tp, P)
    rec['aircraft'] = {a: stats(tp & (cls == a), P) for a in ('M4E', 'M400')}
    rec['dist'] = {'%d-%d m' % (lo, hi): stats(tp & (dist >= lo) & (dist < hi), P) for lo, hi in DB}
    rec['fx'] = {n: stats(tp & m, P) for n, m in FB}
    rec['aircraft_x_dist'] = {'%s %d-%d' % (a, lo, hi): stats(tp & (cls == a) & (dist >= lo) & (dist < hi), P) for a in ('M4E', 'M400') for lo, hi in DB}
    rec['size'] = {a: sizestats(tp & (cls == a), P) for a in ('M4E', 'M400')}
    # frames per gt distance band / fx band (all frames, for n context)
    out[name] = rec
out['_frames'] = {'dist': {'%d-%d' % (lo, hi): int(((dist >= lo) & (dist < hi)).sum()) for lo, hi in DB},
                  'fx': {n: int(m.sum()) for n, m in FB}, 'aircraft': {a: int((cls == a).sum()) for a in ('M4E', 'M400')},
                  'seq': {s: int((seq == s).sum()) for s in sorted(set(seq))}, 'fx_unique': sorted(set(np.round(fx, 1).tolist()))}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'c1a_bias.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
