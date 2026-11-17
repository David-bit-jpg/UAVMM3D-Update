# -*- coding: utf-8 -*-
"""[plq_1116 task 2] Why some AntiUAV3D rows of tab_main have AP ~0. CPU only; reads test GT ONLY via the pack scorer's own meta file
(the file that produced the table). Diagnostic numbers (scale-corrected AP etc.) use GT and are NOT for the paper table.
Usage: python task2_outdoor_zero_ap.py"""
import os, sys, pickle, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
sys.path.insert(0, 'D:/UAV3D_baselines_pack/scripts')
import score as S   # exact scorer functions (box2d, iou, sym, nus_ap, metrics, CFG)
from scipy.spatial.transform import Rotation as Rot

OUT = os.path.dirname(os.path.abspath(__file__))
R = 'D:/advisor_results_1115/runs/'
ROWS = [('GUPNet (sim only)', 'gupnet_out_sim_s0'), ('GUPNet + ST', 'gupnet_out_st_anti_s0'), ('GUPNet + PST', 'gupnet_out_kd_anti_s0'),
        ('MonoDETR (sim only)', 'monodetr_out_sim_s0'), ('MonoDETR + ST', 'monodetr_out_st_anti_s0'), ('MonoDETR + PST', 'monodetr_out_kd_anti_s0'),
        ('MonoCD (sim only)', 'monocd_out_sim_s0'), ('MonoCD + ST', 'monocd_out_st_anti_s0'), ('MonoCD + PST', 'monocd_out_kd_anti_s0'),
        ('MonoCoP (sim only)', 'monocop_out_sim_s0'), ('MonoDGP + ST (good row)', 'monodgp_out_st_anti_s0')]
M = np.load('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz')   # = server pack data/gt/outdoor_test_meta.npz (make_pack_update.py)
key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
N = len(key); g = M['gt']; K = M['K']; c = S.CFG['outdoor_test']
gd = np.linalg.norm(g[:, :3], axis=1)
m4e = np.all(np.abs(g[:, 3:6] - [0.307, 0.3875, 0.1495]) < 1e-3, axis=1)


def load(f):
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(f, 'rb')):
        i = key.get((str(r['seq']), str(r['frame'])))
        b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def ap_of(P):
    return S.metrics(P, g, K, np.arange(N), c)['AP']


res = []; lines = []
for name, d in ROWS:
    f = R + d + '/pred_outdoor_test.pkl'
    if not os.path.exists(f):
        lines.append('missing %s' % f); continue
    P = load(f); m = S.metrics(P, g, K, np.arange(N), c)
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf'])
    pd_ = np.linalg.norm(P['pt'], axis=1)
    rel_c = np.linalg.norm(P['pt'] - g[:, :3], axis=1) / gd          # AP matching quantity (centre distance / true distance)
    ratio = pd_ / gd                                                  # range ratio
    zr = P['pt'][:, 2] / g[:, 2]
    # TP set (projected 2D IoU >= 0.1), same as scorer
    tp = np.zeros(N, bool)
    for i in np.where(has)[0]:
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        tp[i] = S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) >= 0.1
    h = has
    # direction error (angle between predicted and GT rays), the part a depth-scale fix cannot repair
    cosang = np.sum(P['pt'] * g[:, :3], 1) / (pd_ * gd)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    lat = np.sin(np.radians(ang))                                     # lateral error / distance at correct range
    # diagnostic: global range-scale correction by median ratio (on frames with a box), and per-frame oracle range
    s_med = np.nanmedian(ratio[h]); s_med_tp = np.nanmedian(ratio[tp]) if tp.any() else np.nan
    Pc = dict(P); Pc['pt'] = P['pt'] / s_med; ap_corr = ap_of(Pc)
    Pc2 = dict(P); Pc2['pt'] = P['pt'] / s_med_tp if np.isfinite(s_med_tp) else P['pt']; ap_corr_tp = ap_of(Pc2)
    Po = dict(P); Po['pt'] = P['pt'] / ratio[:, None]; ap_or = ap_of(Po)           # every frame at the exact GT range (direction kept)
    # size: implied distance if depth came from 2-D width with predicted size -> size ratio
    sz_ratio = np.nanmedian(np.mean(P['plwh'][h], 1) / np.mean(g[h, 3:6], 1))
    r = dict(row=name, run=d, n_frames=N, n_with_box=int(h.sum()), AP=m['AP'], ATE_rel=m['ATE'], ASE_cm=m['ASE'], AOE=m['AOE'], mDR=m['mDR'], UDS=m['UDS'],
             relcentre_med=float(np.nanmedian(rel_c[h])), relcentre_mean=float(np.nanmean(rel_c[h])),
             **{'frac_relcentre_le_%g' % (100 * t): float(100 * np.mean(rel_c[h] <= t) * h.mean()) for t in c['thr']},
             signed_range_rel_med_pct=float(100 * (np.nanmedian(ratio[h]) - 1)), signed_range_rel_med_TP_pct=float(100 * (s_med_tp - 1)),
             signed_z_rel_med_pct=float(100 * (np.nanmedian(zr[h]) - 1)),
             abs_range_rel_med_pct=float(100 * np.nanmedian(np.abs(ratio[h] - 1))),
             ray_angle_med_deg=float(np.nanmedian(ang[h])), lateral_rel_med_pct=float(100 * np.nanmedian(lat[h])),
             frac_lateral_le_5pct=float(100 * np.mean(lat[h] <= 0.05) * h.mean()),
             frac_TP_2d=float(100 * tp.mean()),
             size_err_med_cm=float(np.nanmedian(100 * np.mean(np.abs(P['plwh'][h] - g[h, 3:6]), 1))),
             pred_lwh_med_M4E=np.round(np.nanmedian(P['plwh'][h & m4e], 0), 3).tolist(), gt_lwh_M4E=[0.307, 0.3875, 0.1495],
             size_ratio_med=float(sz_ratio),
             DIAG_AP_global_scale_fix=ap_corr, DIAG_AP_global_scale_fix_TPmedian=ap_corr_tp, DIAG_AP_oracle_range=ap_or,
             DIAG_scale_used=float(s_med))
    # score ties: AP depends on how tied scores are ordered (np.argsort default is not stable); range over 50 random tie-breaks
    cf = P['conf'][h]; r['n_unique_scores'] = int(len(np.unique(cf))); r['frac_score_eq0'] = float(100 * np.mean(cf == 0))
    rng = np.random.default_rng(0); aps = []
    for _ in range(50):
        Pt = dict(P); Pt['conf'] = P['conf'] + 1e-12 * rng.random(N); aps.append(ap_of(Pt))
    r['AP_tiebreak_min'], r['AP_tiebreak_med'], r['AP_tiebreak_max'] = float(np.min(aps)), float(np.median(aps)), float(np.max(aps))
    # near / mid / far signed bias
    for lo, hi in c['bins']:
        s = h & (gd >= lo) & (gd < hi)
        r['signed_range_rel_med_%g-%g' % (lo, hi)] = float(100 * (np.nanmedian(ratio[s]) - 1)) if s.any() else np.nan
    res.append(r)

import pandas as pd
df = pd.DataFrame(res); df.to_csv(OUT + '/task2_outdoor_rows.csv', index=False, float_format='%.4g')
json.dump(res, open(OUT + '/task2_outdoor_rows.json', 'w'), indent=1, ensure_ascii=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(df.T.to_string())
