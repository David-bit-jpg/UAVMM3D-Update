# -*- coding: utf-8 -*-
"""[review1113 heads H6] height ratio from the height network's BOX-CORNER keypoints instead of its LiDAR structure points.
usage: python kpsbox_x.py <kps ckpt> <names tr1,val1,te1> <out pattern with %s> [cap 80]
Single factor vs ori_x.cmd_kps (the reported height reading): the SAME network (sx.S['kps'] = KZ-KPS seed 0, i.e. KZ-IRLiD + structure-point
head), the SAME BN re-estimation (train crops conf >= 0.5, <= 80 per sequence, rng 0, kps head run in the BN pass exactly as cmd_kps), the SAME
read cell (heatmap peak within 5x5 of the detection cell). Only the target the ratio is computed from changes:
  reported : r = z-extent / mean(x, y extents) of the 8 predicted LiDAR STRUCTURE points (kps head, LiDAR-supervised)
  H6       : the 8 predicted 3-D BOX CORNERS (kp2d head, box-supervised; decoded as ori_x.ensemble: centre cell + centre offset + 64 * kp2d)
             -> PnP with s_bar (ori_x.pnp, the corner PnP of the chain) -> least-squares refit of (R, t, box height) with length / width
             fixed at s_bar (scale gauge) -> r = h / mean(l, w).  Bounded |log(h / h_bar)| <= 1.5; frames where PnP / refit fail are left out
             (chain_x deploy then uses the median of the train readings, its existing fallback).
Writes the same {(seq, frame): r} pickles as cmd_kps. No labels read."""
import os, sys, pickle
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ori_x as OO  # noqa  (this lib copy)
from scipy.optimize import least_squares  # noqa


def fit_ratio(c2, Kc, lwh):
    s = OO.pnp(c2, Kc, lwh)
    if s is None:
        return None
    R0, t0 = s
    rv0 = cv2.Rodrigues(np.asarray(R0, np.float64))[0].reshape(3)
    c2 = np.asarray(c2, np.float64)

    def res(p):
        R = cv2.Rodrigues(p[:3].reshape(3, 1))[0]
        X = (OO.PROTO8 * np.array([lwh[0], lwh[1], lwh[2] * np.exp(p[6])])) @ R.T + p[3:6]
        if (X[:, 2] <= 1e-3).any():
            return np.full(16, 1e3)
        q = X @ Kc.T
        return (q[:, :2] / q[:, 2:3] - c2).ravel()
    p0 = np.r_[rv0, t0, 0.0]
    lo = np.r_[np.full(6, -np.inf), -1.5]; hi = np.r_[np.full(6, np.inf), 1.5]
    try:
        r = least_squares(res, p0, bounds=(lo, hi), method='trf', max_nfev=200)
    except Exception:
        return None
    if not r.success or not np.isfinite(r.x).all() or r.x[5] <= 0:
        return None
    return float(lwh[2] * np.exp(r.x[6]) / np.mean(lwh[:2]))


def cmd_kpsbox(ck, name, out, cap=80):
    import torch
    for p in ('E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kzlid', 'E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kzbody', 'E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kpsize'):
        sys.path.insert(0, p)
    import uavdet3d.model.detectors as dets
    from center_det_mm import CenterDetMM
    from center_det_mmk import CenterDetMMK
    dets.__all__['CenterDetMM'] = CenterDetMM; dets.__all__['CenterDetMMK'] = CenterDetMMK
    OO.CFG = OO.sx.S['kps_cfg']
    m = OO.build(ck)
    names = name.split(',')
    W = OO.W
    X = np.load('%s/oc_%s.npy' % (W, names[0]), mmap_mode='r'); info = pickle.load(open('%s/oc_%s.pkl' % (W, names[0]), 'rb'))['info']
    if cap >= 0:   # = ori_x.cmd_kps BN pass, verbatim
        el = [i for i, r in enumerate(info) if r['conf'] >= 0.5]
        rng = np.random.default_rng(0); by = {}
        for i in el:
            by.setdefault(info[i]['seq'], []).append(i)
        el = sorted(sum([list(rng.choice(v, min(cap, len(v)), replace=False)) if cap else v for v in by.values()], []))
        bns = [mm for mm in m.modules() if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm)]
        for b in bns:
            b.reset_running_stats(); b.momentum = None; b.train()
        with torch.no_grad():
            for _, x in OO.batches(X, np.array(el), 16):
                tb = {'image': x, 'batch_size': x.shape[0]}
                for mod in m.module_list:
                    tb = mod(tb)
                    if mod is m.dense_head_2d:
                        break
                f = tb['features_2d']; m.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:]))
        m.eval()
        print('kpsbox adabn on %d crops' % len(el), flush=True)
    lim = int(os.environ.get('KPSBOX_LIMIT', '0'))   # plumbing tests only
    for nm_ in names:
        X = np.load('%s/oc_%s.npy' % (W, nm_), mmap_mode='r'); I = pickle.load(open('%s/oc_%s.pkl' % (W, nm_), 'rb'))
        info = I['info']; lwh = np.asarray(I['size'], np.float64)
        res = {}; nfail = 0
        ids = np.arange(len(info)) if not lim else np.arange(min(lim, len(info)))
        with torch.no_grad():
            for ii, x in OO.batches(X, ids, int(os.environ.get('ORI_BS', '16'))):
                tb = {'image': x, 'batch_size': x.shape[0]}
                for mod in m.module_list:
                    tb = mod(tb)
                    if mod is m.dense_head_2d:
                        break
                pc = tb.get('pred_center_dict', m.dense_head_2d.forward_loss_dict['pred_center_dict'])
                hm = torch.sigmoid(pc['hm'].float()).cpu().numpy().reshape(len(ii), -1, 36, 64)[:, 0]
                cr = pc['center_res'].float().cpu().numpy().reshape(len(ii), -1, 36, 64)
                kp = pc['kp2d'].float().cpu().numpy().reshape(len(ii), -1, 36, 64)
                for b, i in enumerate(ii):
                    r0, c0 = int(np.clip(info[i]['vc'] // 8, 0, 35)), int(np.clip(info[i]['uc'] // 8, 0, 63))
                    rs, cs = np.clip(np.arange(r0 - 2, r0 + 3), 0, 35), np.clip(np.arange(c0 - 2, c0 + 3), 0, 63)
                    h = hm[b][np.ix_(rs, cs)]; r, c = np.unravel_index(int(np.argmax(h)), h.shape)
                    gr, gc = rs[r], cs[c]
                    ctr = (np.array([gc, gr], np.float64) + cr[b][:2, gr, gc]) * 8.0
                    c2 = ctr[None] + kp[b][:, gr, gc].reshape(8, 2) * 64.0
                    v = fit_ratio(c2, np.asarray(info[i]['K_crop'], np.float64), lwh)
                    if v is None:
                        nfail += 1
                    else:
                        res[(info[i]['seq'], info[i]['frame'])] = v
        pickle.dump(res, open(out % nm_, 'wb')); v = np.array(list(res.values()))
        print('kpsbox %s on %s: %d / %d crops (fit failed %d), box-corner height ratio median %.3f, s_bar ratio %.3f' % (
            os.path.basename(ck), nm_, len(res), len(ids), nfail, np.median(v) if len(v) else np.nan, lwh[2] / lwh[:2].mean()), flush=True)


if __name__ == '__main__':
    cmd_kpsbox(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 80)
