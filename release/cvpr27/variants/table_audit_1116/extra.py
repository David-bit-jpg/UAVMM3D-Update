# Extra checks: best-case tie order for MonoCD+PST (RealDrone3D), exact distance-bin UDS, Mavic-2 ATE.
import numpy as np, json
import derived_lib as L
S = L.S
W = 'C:/Users/jiang/OneDrive/文档/xwechat_files/wxid_pu1uzzbuzwsx22_b4ed/msg/file/2026-11/results_to_send_20261116_1459/runs/'
G = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/'
out = {}
for nm in ('monocd_out_sim_s0', 'monocd_out_st_anti_s0', 'monocd_out_kd_anti_s0'):
    P, M = L.load(W + nm + '/pred_outdoor_test.pkl', 'outdoor_test')
    old = S.nus_ap
    def best_ap(sc, tp, n_gt):  # ties broken in favour of true positives (upper bound over tie orders)
        if not len(sc): return 0.0
        o = np.lexsort((-tp.astype(float), -sc)); t = tp[o].astype(float)
        ctp, cfp = np.cumsum(t), np.cumsum(1 - t); rec, prec = ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1)
        rs = np.linspace(0, 1, 101); p = np.array([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in rs])[rs > 0.1]
        return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)
    def worst_ap(sc, tp, n_gt): return best_ap(sc, ~tp.astype(bool), n_gt) if False else _worst(sc, tp, n_gt)
    def _worst(sc, tp, n_gt):
        if not len(sc): return 0.0
        o = np.lexsort((tp.astype(float), -sc)); t = tp[o].astype(float)
        ctp, cfp = np.cumsum(t), np.cumsum(1 - t); rec, prec = ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1)
        rs = np.linspace(0, 1, 101); p = np.array([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in rs])[rs > 0.1]
        return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)
    S.nus_ap = best_ap; b = L.met(P, M, 'outdoor_test'); S.nus_ap = _worst; w = L.met(P, M, 'outdoor_test'); S.nus_ap = old
    out['tie best/worst ' + nm] = dict(best_UDS=round(b['UDS'], 3), best_AP=round(b['AP'], 3), worst_UDS=round(w['UDS'], 3), worst_AP=round(w['AP'], 3))
    print(nm, out['tie best/worst ' + nm], flush=True)
for nm, p in [('PST', G + 'srcx/out2antifs/preds/pcmd_fused_test.pkl'), ('src', G + 'srcx/out2antifs/preds/srconly_test.pkl')]:
    P, M = L.load(p, 'outdoor_test'); dist = np.linalg.norm(M['gt'][:, :3], axis=1)
    out['RD bins exact ' + nm] = [L.met(P, M, 'outdoor_test', np.where((dist >= lo) & (dist < hi))[0])['UDS'] for lo, hi in ((0, 25), (25, 50), (50, 999))]
    print(nm, out['RD bins exact ' + nm])
for nm, p in [('PST', G + 'srcx/in2mav/preds/pcmd_fused_test.pkl'), ('RGBonly', G + 'srcx/in2mav_none/preds/pcmd_fused_test.pkl')]:
    P, M = L.load(p, 'real_test'); sel = np.where(np.array([s.startswith('mavic2') for s in M['seq']]))[0]
    out['MAV mavic2 exact ' + nm] = L.met(P, M, 'real_test', sel); print(nm, out['MAV mavic2 exact ' + nm])
json.dump(out, open('extra.json', 'w'), indent=1)
