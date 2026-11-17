# -*- coding: utf-8 -*-
"""[cleanfix A3] Clean-chain replacement of X_suppl.tex:162 (rotation collapse; '12-15 % narrow'; '69.51 -> 55.89'; '|rho| <= 0.09').
Replicates gproto/wf21/orient2pos/orient2pos.py (arms base / gt / gtn) on the clean MAV6D chain srcx/in2mav, CPU only, no retraining:
  detector depth = f * W(s_bar, R_det) / w2d (width-only decoding, det_infer_x.w2d_of). For every det1 test detection whose centre is < 1 m
  from the GT centre, re-solve z' so that the projected width with R_gt equals the detector's w2d (same ray, same s_bar, 12 perspective
  iterations = orient2pos.resolve); s = z'/z.  Arms (everything else = the deployed predictions):
    base      deployed predictions (pcmd_nofuse_test = refined detector chain before the corner fusion; pcmd_fused_test = PST)
    gt        position * s (nofuse);  fused: z_fused * s^(1-w)  (the fusion's log-linear weight on the refined depth; b_k kept = deployed,
              because re-estimating it would need GT rotations on the calibration images, whose labels are never read)
    gtn       s / median(s on TRAIN detections, GT within 1 m)   (orient2pos normalised by the other split's median; here the train split)
Diagnostics (test TP frames, centre < 1 m): width bias log(w2d / projected width of the GT box with K_in); share of frames with
W(R_gt) > W(R_det) (collapse toward facing the camera); Spearman of the final depth error log(z/z_gt) with log s and with AOE.
GT is read ONLY for these post-hoc diagnostics; nothing is selected with it. Test scored once per arm (score_x.score_meta)."""
import json, os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import spearmanr
os.environ.setdefault('SX_SRC', 'in'); os.environ.setdefault('SX_TGT', 'mav')
L = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib'; sys.path.insert(0, L)
import sx, score_x  # noqa
X = sx.O; HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(HERE + '/preds', exist_ok=True)
P8 = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]


def proj_w(ctr, lwh, R, K):
    c = (P8 * lwh) @ R.T + ctr
    if (c[:, 2] <= 1e-6).any():
        return np.nan
    return float(np.ptp(K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2]))


def resolve(ctr0, lwh, R0, R1, K):
    wt = proj_w(ctr0, lwh, R0, K)
    if not np.isfinite(wt) or wt <= 1e-6:
        return 1.0
    ray = ctr0 / ctr0[2]
    a0 = np.ptp(((P8 * lwh) @ R0.T)[:, 0]); a1 = np.ptp(((P8 * lwh) @ R1.T)[:, 0]); z = ctr0[2] * a1 / max(a0, 1e-9)
    for _ in range(12):
        w_ = proj_w(ray * z, lwh, R1, K)
        if not np.isfinite(w_) or w_ <= 1e-9:
            return 1.0
        r = w_ / wt; z *= r
        if abs(r - 1) < 1e-10:
            break
    return float(z / ctr0[2])


def sym(Rp, Rg):
    return min(np.degrees(np.arccos(np.clip((np.trace((Rp @ RZ[q]).T @ Rg) - 1) / 2, -1, 1))) for q in range(4))


D = pickle.load(open(X + '/recs/det1.pkl', 'rb')); sbar = np.asarray(D['size'], np.float64)
w = json.load(open(X + '/w_sim.json'))['w']


def gt_lookup(split):
    if split == 'test':
        M = np.load(sx.T['gt_meta'])
        return {(str(s), str(f)): g for s, f, g, n in zip(M['seq'], M['frame'], M['gt'], M['n_gt']) if n > 0}
    I = pickle.load(open(sx.REAL + '/train/index.pkl', 'rb')); assert I.get('euler_seq', 'xyz') == 'xyz'
    return {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64).reshape(-1, 9) for m in I['metas'] if m is not None}


def scales(split):
    G = gt_lookup(split); out = {}; diag = []
    for r in D['recs'][split]:
        if r['box'] is None:
            continue
        k = (r['seq'], r['frame']); g = G.get(k)
        if g is None:
            continue
        g = np.asarray(g, np.float64).reshape(-1, 9); b = np.asarray(r['box'], np.float64)
        j = int(np.argmin(np.linalg.norm(g[:, :3] - b[:3], axis=1))); g = g[j]
        if np.linalg.norm(g[:3] - b[:3]) >= 1.0:
            continue
        K = np.asarray(r['K_in'], np.float64); R0 = Rot.from_euler('xyz', b[6:9]).as_matrix(); Rg = Rot.from_euler('xyz', g[6:9]).as_matrix()
        s = resolve(b[:3], b[3:6], R0, Rg, K); out[k] = s
        wgt = proj_w(g[:3], g[3:6], Rg, K)
        diag.append(dict(s=s, wbias=np.log(r['w2d'] / wgt) if np.isfinite(wgt) else np.nan,
                         gt_span_wider=np.ptp(((P8 * b[3:6]) @ Rg.T)[:, 0]) > np.ptp(((P8 * b[3:6]) @ R0.T)[:, 0]),
                         det_aoe=sym(R0, Rg)))
    return out, diag


S_tr, diag_tr = scales('train'); S_te, diag_te = scales('test')
med_tr = float(np.median(list(S_tr.values())))
res = dict(n_train_tp=len(S_tr), n_test_tp=len(S_te), median_s_train=med_tr, median_s_test=float(np.median(list(S_te.values()))),
           w=w, s_bar=sbar.tolist())
wb = np.array([d['wbias'] for d in diag_te]); wbt = np.array([d['wbias'] for d in diag_tr])
res['width_bias_test_median_pct'] = float(100 * (np.exp(np.nanmedian(wb)) - 1)); res['width_bias_train_median_pct'] = float(100 * (np.exp(np.nanmedian(wbt)) - 1))
res['width_bias_test_mad_pct'] = float(100 * np.nanmedian(np.abs(wb - np.nanmedian(wb))))
res['share_gt_span_wider_than_det_span_test'] = float(np.mean([d['gt_span_wider'] for d in diag_te]))
res['share_gt_span_wider_than_det_span_train'] = float(np.mean([d['gt_span_wider'] for d in diag_tr]))
res['det_rot_head_sym_AOE_test_mean'] = float(np.mean([d['det_aoe'] for d in diag_te]))
res['det_rot_head_sym_AOE_test_median'] = float(np.median([d['det_aoe'] for d in diag_te]))

M = np.load(sx.T['gt_meta']); midx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
arms = {}
for base_name, expo in (('pcmd_nofuse_test', 1.0), ('pcmd_fused_test', 1.0 - w)):
    B = pickle.load(open(X + '/preds/%s.pkl' % base_name, 'rb'))
    for arm, norm in (('base', None), ('gt', 1.0), ('gtn', med_tr)):
        P = []
        for r in B:
            q = np.asarray(r['boxes9d'], np.float64).reshape(1, 9).copy(); k = (r['seq'], r['frame'])
            if norm is not None and k in S_te:
                q[0, :3] *= (S_te[k] / norm) ** expo
            P.append(dict(seq=r['seq'], frame=r['frame'], boxes9d=q, scores=r['scores']))
        fn = '%s/preds/%s_%s.pkl' % (HERE, base_name, arm); pickle.dump(P, open(fn, 'wb'))
        sc = score_x.score_meta(fn, sx.T['gt_meta']); json.dump(sc, open(fn + '.score_real_test.json', 'w'), indent=1, default=float)
        lz = []; ls = []; ao = []
        for r in P:
            k = (r['seq'], r['frame']); i = midx.get(k)
            if i is None or M['n_gt'][i] == 0:
                continue
            g = M['gt'][i]; q = r['boxes9d'][0]
            if np.linalg.norm(q[:3] - g[:3]) < 1.0:
                lz.append(np.log(q[2] / g[2])); ls.append(np.log(S_te.get(k, 1.0)))
                ao.append(sym(Rot.from_euler('xyz', q[6:9]).as_matrix(), Rot.from_euler('xyz', g[6:9]).as_matrix()))
        lz, ls, ao = map(np.array, (lz, ls, ao))
        a = dict(UDS=sc['all']['UDS'], AP=sc['all']['AP'], ATE=sc['all']['ATE'], AOE=sc['all']['AOE'], n_tp=len(lz),
                 depth_abs_med_pct=float(100 * np.median(np.abs(lz))), depth_bias_med_pct=float(100 * np.median(lz)))
        if arm == 'base':
            a['spearman_deptherr_vs_logs_gt'] = float(spearmanr(lz, ls)[0])
            a['spearman_absdeptherr_vs_AOE'] = float(spearmanr(np.abs(lz), ao)[0])
            a['spearman_deptherr_vs_AOE'] = float(spearmanr(lz, ao)[0])
        arms['%s|%s' % (base_name, arm)] = a
        print(base_name, arm, json.dumps(a), flush=True)
res['arms'] = arms
json.dump(res, open(HERE + '/a3_result.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != 'arms'}, indent=1, default=float))
