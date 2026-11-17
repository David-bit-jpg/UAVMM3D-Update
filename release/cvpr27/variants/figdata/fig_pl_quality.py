# -*- coding: utf-8 -*-
"""Figure data (a): per-frame pseudo-label quality on the unlabeled TRAINING frames (post-hoc; GT only to evaluate; CPU only; no inference).
MAV6D chain srcx/in2mav (+ RGB-only control srcx/in2mav_none); AntiUAV3D chain srcx/out2antifs (+ srcx/out2antifs_none).
Detector raw pseudo label = recs/det0.pkl (sim-only 3-seed ensemble + flip; the box make_pseudo_x.py starts from).
Teacher position pseudo label = index_pseudo_srcx_<chain>.pkl (what the students trained on; conf >= 0.5 frames).
Teacher orientation pseudo label = srcx_work/<chain>/pseudo_tr1.npz R (gate = IR vs LiDAR teacher disagreement <= 10 deg; ori students train on gate frames).
Usage: python fig_pl_quality.py mav|anti"""
import os, sys, pickle
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
OUT = 'D:/CVPR27_UAV/figures/data'
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
PROTO = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]


def sym(Rp, Rg):   # = pack score.sym (AOE, 4-fold about body z)
    return min(np.degrees(np.arccos(np.clip((np.trace((Rp @ RZ[q]).T @ Rg) - 1) / 2, -1, 1))) for q in range(4))


def E(e):
    return Rot.from_euler('xyz', np.asarray(e, np.float64)).as_matrix()


P = lambda f: pickle.load(open(f, 'rb'))
which = sys.argv[1]
if which == 'mav':
    X, XN = GP + '/srcx/in2mav', GP + '/srcx/in2mav_none'; W, WN = 'D:/S2MUAV3D/srcx_work/in2mav', 'D:/S2MUAV3D/srcx_work/in2mav_none'
    IXF, IXFN = 'E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav.pkl', 'E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav_none.pkl'
    I = P('E:/mmcache/mav6d_cn_obj/train/index.pkl')
    GT = {(m['seq'], m['frame']): (np.asarray(m['boxes9d'], np.float64)[0], m['cls'], np.asarray(m['K_in'], np.float64)) for m in I['metas'] if m is not None and len(m['boxes9d'])}
    fn = OUT + '/pl_quality_mav6d.csv'
else:
    X, XN = GP + '/srcx/out2antifs', GP + '/srcx/out2antifs_none'; W, WN = 'D:/S2MUAV3D/srcx_work/out2antifs', 'D:/S2MUAV3D/srcx_work/out2antifs_none'
    IXF, IXFN = 'D:/mmcache/outdoor_real_fs/train/index_pseudo_srcx_out2antifs.pkl', 'D:/mmcache/outdoor_real_fs/train/index_pseudo_srcx_out2antifs_none.pkl'
    GT = {(g['seq'], g['frame']): (np.asarray(g['box9d_ours'], np.float64), g['aircraft'], np.asarray(g['K'], np.float64))
          for g in P('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/train_gt.pkl')}
    fn = OUT + '/pl_quality_antiuav3d.csv'


def ixtab(f):
    IX = P(f)
    return {(IX['metas'][int(i)]['seq'], IX['metas'][int(i)]['frame']): np.asarray(IX['metas'][int(i)]['boxes9d'], np.float64)[0] for i in IX['valid_idx']}


def oritab(w):
    z = np.load(w + '/pseudo_tr1.npz')
    return {(str(s), str(f)): (z['R'][i], bool(z['gate'][i]), float(z['d_il'][i])) for i, (s, f) in enumerate(zip(z['seq'], z['frame']))}


D0 = P(X + '/recs/det0.pkl')['recs']['train']; D1 = P(X + '/recs/det1.pkl')['recs']['train']; D1N = P(XN + '/recs/det1.pkl')['recs']['train']
d1 = {(r['seq'], r['frame']): r['box'] for r in D1}; d1n = {(r['seq'], r['frame']): r['box'] for r in D1N}
PL, PLN, OR, ORN = ixtab(IXF), ixtab(IXFN), oritab(W), oritab(WN)
rows = []
nan = float('nan')
for r in D0:
    k = (r['seq'], r['frame'])
    if k not in GT or r['box'] is None:
        continue
    g, ac, Kg = GT[k]
    if not np.isfinite(g[2]):
        continue
    b = np.asarray(r['box'], np.float64); Rg = E(g[6:9])
    Kd = np.asarray(r['K_in'], np.float64) if which == 'mav' else Kg   # det0 boxes live in the K_in frame (MAV6D) / raw K (AntiUAV3D, = index K_raw)
    cor = (PROTO * g[3:6]) @ Rg.T + g[:3]; q = (Kd @ cor.T).T; q = q[:, :2] / q[:, 2:3]
    u = Kd @ b[:3]; u = u[:2] / u[2]
    in2d = bool(q[:, 0].min() <= u[0] <= q[:, 0].max() and q[:, 1].min() <= u[1] <= q[:, 1].max())
    rel = lambda z: 100 * (z / g[2] - 1)
    row = dict(frame_id='%s/%s' % k, seq=k[0], frame=k[1], aircraft=ac, gt_dist_m=float(np.linalg.norm(g[:3])), gt_depth_z_m=float(g[2]),
               det_conf=float(r['conf']), det_center_err_m=float(np.linalg.norm(b[:3] - g[:3])), det_center_in_gt2d=in2d,
               det_depth_relerr_pct=rel(b[2]), det_yaw_err_deg=sym(E(b[6:9]), Rg))
    row['has_pseudo'] = k in PL
    row['teacher_depth_relerr_pct'] = rel(PL[k][2]) if k in PL else nan
    row['rgbonly_teacher_depth_relerr_pct'] = rel(PLN[k][2]) if k in PLN else nan
    o = OR.get(k)
    row['has_ori_pseudo'] = o is not None
    row['teacher_yaw_err_deg'] = sym(o[0], Rg) if o else nan
    row['ir_lidar_disagree_deg'] = o[2] if o else nan
    row['gate_pass'] = o[1] if o else False
    on = ORN.get(k)
    row['rgbonly_teacher_yaw_err_deg'] = sym(on[0], Rg) if on else nan
    row['rgbonly_gate_pass'] = on[1] if on else False
    s = d1.get(k); row['student_depth_relerr_pct'] = rel(s[2]) if s is not None else nan
    row['student_yaw_err_deg'] = sym(E(np.asarray(s)[6:9]), Rg) if s is not None else nan
    s = d1n.get(k); row['rgbonly_student_depth_relerr_pct'] = rel(s[2]) if s is not None else nan
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(fn, index=False, float_format='%.5g')
# ---- checks
log = []
def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; return np.median(np.abs(x)), np.median(100 * np.log1p(x / 100)), len(x)
log.append('%s: %d frames with GT and a det0 box (det0 train recs %d); has_pseudo %d (index %d; all in GT: %s); ori pseudo %d, gate pass %d'
           % (fn, len(df), len(D0), df.has_pseudo.sum(), len(PL), set(PL) <= set(df[df.has_pseudo].apply(lambda r: (r.seq, r.frame), axis=1)), df.has_ori_pseudo.sum(), df.gate_pass.sum()))
k = df[df.has_pseudo]
for c in ('det_depth_relerr_pct', 'teacher_depth_relerr_pct', 'rgbonly_teacher_depth_relerr_pct', 'student_depth_relerr_pct', 'rgbonly_student_depth_relerr_pct'):
    m, b, n = med(k[c]); log.append('  kept frames  %-36s n %5d  median|rel| %.1f%%  bias(median log, %%) %+.1f' % (c, n, m, b))
log.append('  det centre inside projected GT box: all %d / %d, kept %d / %d' % (df.det_center_in_gt2d.sum(), len(df), k.det_center_in_gt2d.sum(), len(k)))
kh = k[k.det_center_in_gt2d]
for c in ('det_depth_relerr_pct', 'teacher_depth_relerr_pct', 'rgbonly_teacher_depth_relerr_pct', 'student_depth_relerr_pct'):
    m, b, n = med(kh[c]); log.append('  kept & centre-in-GT-box  %-28s n %5d  median|rel| %.1f%%  bias %+.1f' % (c, n, m, b))
for sub, nm in ((df, 'all'), (df[df.gate_pass], 'gate-pass'), (df[df.has_ori_pseudo & ~df.gate_pass], 'ori pseudo, gate fail')):
    log.append('  %-22s n %5d  det yaw mean %.2f median %.2f | teacher yaw mean %.2f median %.2f | RGB-only teacher mean %.2f' % (
        nm, len(sub), sub.det_yaw_err_deg.mean(), sub.det_yaw_err_deg.median(), sub.teacher_yaw_err_deg.mean(), sub.teacher_yaw_err_deg.median(),
        sub.rgbonly_teacher_yaw_err_deg.mean()))
open(fn.replace('.csv', '_check.log'), 'w', encoding='utf-8').write('\n'.join(log) + '\n'); print('\n'.join(log))
