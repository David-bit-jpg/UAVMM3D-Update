# -*- coding: utf-8 -*-
"""[clean3 item 3a] Per-frame errors on the test sets + rule-based example selection for Fig. 3 (no image is looked at for selection).
Per-frame quantities follow the pack scorer exactly: top-scoring box per frame; TP = projected-2D IoU >= 0.1 with the meta K; distance error
d = |t_pred - t_gt| (m), AntiUAV3D also relative d / |t_gt|; AOE = quadrotor-symmetric geodesic angle (pack sym).
Selection (fixed before rendering; TP frames only, i.e. frames where a prediction overlaps the drone):
  e_pos = d (MAV6D) or d/|t_gt| (AntiUAV3D, the scorer's ATE unit);  rank-normalise e_pos and AOE to percentiles p_pos, p_aoe in [0,1].
  success pool : p_pos <= 0.15 and p_aoe <= 0.15            -> 3 frames
  typical      : the frame minimising max(|p_pos-0.5|, |p_aoe-0.5|)
  failures     : F1 = position failure: p_pos >= 0.95 ; F2 = orientation failure: p_aoe >= 0.95 (worst ~5 % each)
  Within each pool the frames are spread over GT distance: sort the pool by |t_gt| and take the elements at quantiles (1/6, 1/2, 5/6) for 3
  frames, the median for 1; MAV6D additionally prefers distinct sequences (scan outward from the quantile index until an unused sequence is found).
  AntiUAV3D test has only 2 sequences (M4E 2,576 frames, M400 48), so only a minimum gap of 200 frame indices is enforced there."""
import os, sys, json, pickle, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as Rot
os.environ['CUDA_VISIBLE_DEVICES'] = ''
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'; G = GP.rsplit('/', 1)[0]
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
DS = {'mav': dict(pred=GP + '/srcx/in2mav/preds/pcmd_fused_test.pkl', meta=G + '/cmpeval/test_meta.npz', rel=False,
                  cache='E:/mmcache/mav6d_cn_obj/test/index.pkl'),
      'anti': dict(pred=GP + '/srcx/out2antifs/preds/pcmd_fused_test.pkl', meta='D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz',
                   rel=True, cache='D:/mmcache/outdoor_real_fs/test/index.pkl')}


def per_frame(ds):
    c = DS[ds]; M = np.load(c['meta']); N = len(M['seq']); g = M['gt']
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    pb = np.full((N, 9), np.nan); cf = np.full(N, np.nan)
    for r in pickle.load(open(c['pred'], 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores']).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); pb[i] = b[j]; cf[i] = sc[j]
    d = np.linalg.norm(pb[:, :3] - g[:, :3], axis=1); rng_ = np.linalg.norm(g[:, :3], axis=1)
    tp = np.zeros(N, bool); aoe = np.full(N, np.nan)
    for i in np.where(np.isfinite(pb).all(1))[0]:
        if S.iou(S.box2d(pb[i, :3], pb[i, 3:6], pb[i, 6:9], M['K'][i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], M['K'][i])) < 0.1:
            continue
        tp[i] = True; aoe[i] = S.sym(Rot.from_euler('xyz', pb[i, 6:9]).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix())
    return M, pb, cf, d, rng_, tp, aoe


def pick(pool, rng_, seq, frm, qs, used, ds):
    o = pool[np.argsort(rng_[pool])]; out = []
    for q in qs:
        k0 = int(round(q * (len(o) - 1)))
        for dk in sorted(range(-len(o), len(o)), key=abs):
            k = k0 + dk
            if not 0 <= k < len(o):
                continue
            i = o[k]
            if i in used:
                continue
            if ds == 'mav' and seq[i] in {seq[u] for u in used} and len({seq[x] for x in o} - {seq[u] for u in used}) > 0:
                continue
            if ds == 'anti' and any(seq[u] == seq[i] and abs(int(frm[u][:6]) - int(frm[i][:6])) < 200 for u in used):
                continue
            out.append(int(i)); used.add(int(i)); break
    return out


if __name__ == '__main__':
    res = {}
    for ds in ('mav', 'anti'):
        M, pb, cf, d, rng_, tp, aoe = per_frame(ds); seq = np.array([str(s) for s in M['seq']]); frm = np.array([str(f) for f in M['frame']])
        e = d / rng_ if DS[ds]['rel'] else d
        T = np.where(tp)[0]
        pp = np.full(len(e), np.nan); pa = np.full(len(e), np.nan)
        pp[T] = (np.argsort(np.argsort(e[T])) + 0.5) / len(T); pa[T] = (np.argsort(np.argsort(aoe[T])) + 0.5) / len(T)
        used = set()
        succ = pick(T[(pp[T] <= 0.15) & (pa[T] <= 0.15)], rng_, seq, frm, (1 / 6, 1 / 2, 5 / 6), used, ds)
        typ = [int(T[np.argmin(np.maximum(np.abs(pp[T] - 0.5), np.abs(pa[T] - 0.5)))])]; used.add(typ[0])
        f1 = pick(T[pp[T] >= 0.95], rng_, seq, frm, (0.5,), used, ds)
        f2 = pick(T[pa[T] >= 0.95], rng_, seq, frm, (0.5,), used, ds)
        sel = [('success', i) for i in succ] + [('typical', i) for i in typ] + [('failure (position)', i) for i in f1] + [('failure (orientation)', i) for i in f2]
        rows = []
        for tag, i in sel:
            rows.append(dict(kind=tag, idx=int(i), seq=seq[i], frame=frm[i], cls=str(M['cls'][i]), range_m=float(rng_[i]), dist_err_m=float(d[i]),
                             dist_err_rel=float(d[i] / rng_[i]), aoe=float(aoe[i]), p_pos=float(pp[i]), p_aoe=float(pa[i]), conf=float(cf[i]),
                             pred=pb[i].tolist(), gt=M['gt'][i].tolist()))
        res[ds] = dict(n_frames=int(len(e)), n_tp=int(len(T)), median_pos_err=float(np.median(e[T])), median_aoe=float(np.median(aoe[T])),
                       mean_aoe=float(np.mean(aoe[T])), rows=rows)
        for r in rows:
            print(ds, r['kind'], r['seq'], r['frame'], r['cls'], 'z %.2f d %.3f (%.1f%%) aoe %.1f p %.2f/%.2f' % (r['range_m'], r['dist_err_m'], 100 * r['dist_err_rel'], r['aoe'], r['p_pos'], r['p_aoe']))
        print(ds, 'TP', len(T), 'median e', np.median(e[T]), 'median aoe', np.median(aoe[T]))
    json.dump(res, open(OUT + '/selection.json', 'w'), indent=1)
