# R4 diagnostic (CPU): why our Sim-Indoor detector reaches mDR 4.9% on AntiUAV3D v2 while full-image baselines reach 76-92%.
# Uses the UNSEALED val labels for our detector; for the baselines (test predictions only) the test GT meta is read for diagnosis only.
import pickle, json, sys, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('S', 'D:/UAV3D_pack_update_1112/scripts/score.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
SX = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/'
def load_pred(p):
    out = {}
    for r in pickle.load(open(p, 'rb')):
        b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if len(b):
            j = int(np.argmax(sc)); out[(str(r['seq']), str(r['frame']))] = (b[j], sc[j], len(b))
    return out
def diag(pred, gts, name):
    rows = []
    for (key, g, K) in gts:
        g = np.asarray(g, np.float64); K = np.asarray(K, np.float64)
        bg = S.box2d(g[:3], g[3:6], g[6:9], K)
        if key not in pred:
            rows.append(dict(has=0)); continue
        q, c, nb = pred[key]
        bp = S.box2d(q[:3], q[3:6], q[6:9], K)
        iou = S.iou(bp, bg)
        gw = bg[2] - bg[0]; gh = bg[3] - bg[1]
        if bp is None:
            rows.append(dict(has=1, iou=0, conf=c, nb=nb, gw=gw)); continue
        cp = (bp[:2] + bp[2:]) / 2; cg = (bg[:2] + bg[2:]) / 2
        rows.append(dict(has=1, iou=iou, conf=c, nb=nb, gw=gw, cdist=np.linalg.norm(cp - cg) / max(gw, 1e-6), cdist_px=np.linalg.norm(cp - cg),
                         wr=(bp[2] - bp[0]) / gw, hr=(bp[3] - bp[1]) / gh, zrel=q[2] / g[2], gz=g[2]))
    h = np.array([r['has'] for r in rows]); hasr = [r for r in rows if r['has']]
    A = lambda k: np.array([r.get(k, np.nan) for r in hasr], float)
    iou, cd, wr, hr, zr, conf, gw, cdp = A('iou'), A('cdist'), A('wr'), A('hr'), A('zrel'), A('conf'), A('gw'), A('cdist_px')
    loc = cd < 0.5   # predicted 2D box centre within half a GT box width of the GT centre
    res = dict(name=name, n_gt=len(rows), frac_with_pred=float(h.mean()), mean_boxes_per_frame=float(np.nanmean(A('nb'))),
               mDR_iou01=float((iou >= 0.1).sum() / len(rows)), conf_median=float(np.nanmedian(conf)),
               centre_within_half_width=float(loc.mean()), centre_within_2_widths=float((cd < 2).mean()), centre_err_px_1920_median=float(np.nanmedian(cdp)),
               gt_w2d_px_1920_median=float(np.nanmedian(gw)),
               among_localized=dict(n=int(loc.sum()), iou01=float((iou[loc] >= 0.1).mean()) if loc.any() else None,
                                    w_ratio_median=float(np.nanmedian(wr[loc])) if loc.any() else None, h_ratio_median=float(np.nanmedian(hr[loc])) if loc.any() else None,
                                    z_ratio_median=float(np.nanmedian(zr[loc])) if loc.any() else None),
               all_w_ratio_median=float(np.nanmedian(wr)), all_h_ratio_median=float(np.nanmedian(hr)), all_z_ratio_median=float(np.nanmedian(zr)))
    # mDR by GT 2D width (px at 1920) for localization
    bins = [(0, 20), (20, 40), (40, 80), (80, 1e9)]
    res['by_gt_width'] = {('%d-%d' % (lo, min(hi, 9999))): dict(n=int(((gw >= lo) & (gw < hi)).sum()), centre_within_half_width=float(loc[(gw >= lo) & (gw < hi)].mean()) if ((gw >= lo) & (gw < hi)).any() else None,
                                                                iou01=float((iou[(gw >= lo) & (gw < hi)] >= 0.1).mean()) if ((gw >= lo) & (gw < hi)).any() else None) for lo, hi in bins}
    return res
OUT = {}
V = pickle.load(open('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/val.pkl', 'rb'))
gv = [((str(r['seq']), str(r['frame'])), r['box9d_ours'], r['K']) for r in V]
OUT['ours_simonly_in2antifs_val'] = diag(load_pred(SX + 'in2antifs/preds/srconly_val.pkl'), gv, 'ours sim-only (indoor) val')
OUT['ours_simonly_out2antifs_val'] = diag(load_pred(SX + 'out2antifs/preds/srconly_val.pkl'), gv, 'ours sim-only (outdoor) val')
M = np.load('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz')
gt = [((str(s), str(f)), g, K) for s, f, g, K in zip(M['seq'], M['frame'], M['gt'], M['K'])]
OUT['ours_simonly_in2antifs_test'] = diag(load_pred(SX + 'in2antifs/preds/srconly_test.pkl'), gt, 'ours sim-only (indoor) test')
OUT['ours_simonly_out2antifs_test'] = diag(load_pred(SX + 'out2antifs/preds/srconly_test.pkl'), gt, 'ours sim-only (outdoor) test')
for m in ('gupnet', 'monocd', 'monodgp'):
    OUT[m + '_test'] = diag(load_pred('D:/cmpfull_runs/%s_s0/pred_outdoor_v2_test.pkl' % m), gt, m + ' (Sim-Indoor) test')
json.dump(OUT, open('r4_diag.json', 'w'), indent=1)
for k, v in OUT.items():
    a = v['among_localized']
    print('%-30s mDR %.3f | pred/frame %.2f boxes %.2f | conf med %.3f | centre<0.5w %.3f <2w %.3f err px %.0f (gt w %.0f) | localized: n %d iou>=.1 %s w %s h %s z %s | all w %.2f h %.2f z %.2f' % (
        k, v['mDR_iou01'], v['frac_with_pred'], v['mean_boxes_per_frame'], v['conf_median'], v['centre_within_half_width'], v['centre_within_2_widths'], v['centre_err_px_1920_median'], v['gt_w2d_px_1920_median'],
        a['n'], *[('%.2f' % a[x]) if a[x] is not None else '-' for x in ('iou01', 'w_ratio_median', 'h_ratio_median', 'z_ratio_median')], v['all_w_ratio_median'], v['all_h_ratio_median'], v['all_z_ratio_median']))
    print('    by gt width', v['by_gt_width'])
