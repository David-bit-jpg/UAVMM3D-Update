# -*- coding: utf-8 -*-
"""[review1113 C1a-2] where does the distance come from? Decompose log(z_pred/z_gt) on the scorer's TP frames (same matching as c1a_bias).
Chain (verified in code): detector depth z_det = fx*span_x(s_bar, R)/w2d_hat (object_encoder_mav6d.center_point_decoder, SIZE2D_MODE=width,
SIZE_SOURCE=known=s_bar) -> PST: z *= exp(d_hat - c_k) (chain_x.base, |d_hat|<=log 1.4) -> log z = (1-w) log z + w (log z_PnP - b_k) (chain_x.cmd_deploy).
Terms per frame (all natural logs):
  size_px   = log( span_x of the box at the TRUE pose with lwh=s_bar  / span_x of the TRUE box )   (pixel-width inflation the size bias would cause)
  w2d       = log( w2d_gt / w2d_hat )   predicted 2-D width vs projected GT box width (both at the 512-px input)
  det       = log(z_det / z_gt)          (~ size_px + w2d)
  refiner   = d_hat - c_k                (PST only)
  fusion    = log z_final - log z_det - refiner   (PST only; PnP fusion, w = 0.096)"""
import importlib.util, json, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113')
import c1a_bias as B  # noqa  (re-runs the bias table; cheap)
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs/'
P8 = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
DEP = json.load(open(X + 'deploy_pcmd.json'))
cc = np.array(DEP['col_centres']); C = {int(k): v for k, v in DEP['c_k'].items()}; GUARD = float(np.log(1.4))


def span(box, K):
    c = (P8 * box[3:6]) @ Rot.from_euler('xyz', box[6:9]).as_matrix().T + box[:3]
    u = (K @ c.T).T; u = u[:, 0] / u[:, 2]
    return float(np.ptp(u))


def run(name, pred, det, use_ref):
    P = B.load(pred); tp = B.tp_mask(P)
    D = pickle.load(open(X + 'recs/%s.pkl' % det, 'rb')); sbar = np.asarray(D['size'], float)
    R = {(r['seq'], r['frame']): r for r in D['recs']['test'] if r['box'] is not None}
    U = pickle.load(open(X + 'ref/inf_fin_det1_test.pkl', 'rb')) if use_ref else {}
    rows = []
    for i in np.where(tp)[0]:
        k = (str(B.seq[i]), str(B.M['frame'][i])); r = R[k]; Kin = np.asarray(r['K_in'], float); g = B.g[i]; zb = r['box']
        gs = g.copy(); gs[3:6] = sbar; gs[6:9] = zb[6:9]
        zg = np.linalg.norm(g[:3]); zd = np.linalg.norm(zb[:3]); zf = np.linalg.norm(P['pt'][i])
        ref = 0.0
        if use_ref and k in U and abs(U[k]['d']) <= GUARD:
            u = U[k]; c = int(np.argmin(np.abs(cc - u['lum']))) if u['lum'] is not None else -1
            ref = u['d'] - C[c]
        rows.append(dict(cls=str(B.cls[i]), dist=zg, size_px=np.log(span(gs, Kin) / span(g, Kin)), w2d=np.log(span(g, Kin) / r['w2d']),
                         det=np.log(zd / zg), refiner=ref, fusion=np.log(zf / zd) - ref, final=np.log(zf / zg)))
    out = {}
    for grp, sel in [('all', lambda x: True), ('M4E', lambda x: x['cls'] == 'M4E'), ('M400', lambda x: x['cls'] == 'M400')] + \
            [('M4E %d-%d m' % (lo, hi), (lambda lo, hi: lambda x: x['cls'] == 'M4E' and lo <= x['dist'] < hi)(lo, hi)) for lo, hi in B.DB]:
        S = [x for x in rows if sel(x)]
        if S:
            out[grp] = dict(n=len(S), **{t: float(np.median([x[t] for x in S])) for t in ('size_px', 'w2d', 'det', 'refiner', 'fusion', 'final')},
                            **{'mean_' + t: float(np.mean([x[t] for x in S])) for t in ('size_px', 'w2d', 'det', 'refiner', 'fusion', 'final')})
    return out


res = {'s_bar_det0': pickle.load(open(X + 'recs/det0.pkl', 'rb'))['size'].tolist() if hasattr(pickle.load(open(X + 'recs/det0.pkl', 'rb'))['size'], 'tolist') else None,
       'source-only (det0 boxes)': run('srconly', B.PREDS['source-only (srconly_test)'], 'det0', False),
       'PST': run('pst', B.PREDS['PST (pcmd_fused_test)'], 'det1', True), 'deploy_w': DEP['w'], 'b_k_val_images': DEP['b_k_val_images'], 'c_k': DEP['c_k']}
json.dump(res, open('c1a_decomp.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
