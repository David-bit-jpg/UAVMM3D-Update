# -*- coding: utf-8 -*-
"""[clean2 item 3] The two distances of the clean chain (srcx/in2mav). CPU only, existing files only; labels read only for evaluation.
  z_IR  = refined IR distance = z of preds/pcmd_nofuse (t_det1 * exp(d_hat - c_k), phase-3 refiners s5+s6)
  z_cor = corner-PnP distance of the orientation students (S_ab0, S_ab1 + mirror; s_bar box), chain_x.zL_table on the det1 crops (before b_k)
Errors e = log(z / z_gt) on the frames that are TP of the deployed PST (pcmd_fused, scorer rule: top box, 2D IoU >= 0.1) and have a corner reading.
Global bias removed per quantity and split (b = median e over all those frames); robust spread = 1.4826 * median |e - b| (in %);
correlation = Pearson of (e_IR - b_IR, e_cor - b_cor) (also Spearman). Same per brightness class (deploy rule: phase-3 silhouette brightness,
nearest deployment colour centre: 0 = dark, 1 = bright), keeping the GLOBAL bias (as in the development numbers, 'without global offsets');
per-class own-median spreads are listed too. Dev version (paper text, 4_experiments.tex:31): IR 8.3 / 5.1 %, corner 15.5 / 8.6 %, corr 0.36 / 0.00 (dark / bright)."""
import os, sys, json, pickle
import numpy as np
from scipy.stats import spearmanr
os.environ.update(SX_SRC='in', SX_TGT='mav', SX_TEACH='full')
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'; X = G + '/gproto/srcx/in2mav'
sys.path.insert(0, G + '/gproto/review1113/clean2/buildup')
import buildup as BU  # noqa
CX = BU.CX
DEP = json.load(open(X + '/deploy_pcmd.json')); CC = np.asarray(DEP['col_centres'])


def rs(e, b):
    return float(100 * 1.4826 * np.median(np.abs(e - b)))


if __name__ == '__main__':
    res = {}
    for sp in ('val', 'test'):
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); gt = M['gt']; N = len(gt)
        key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
        Pf = BU.load_list(pickle.load(open('%s/preds/pcmd_fused_%s.pkl' % (X, sp), 'rb')), M)
        Pn = BU.load_list(pickle.load(open('%s/preds/pcmd_nofuse_%s.pkl' % (X, sp), 'rb')), M)
        tp = BU.frame_q(Pf, M)['tp']
        nm = {'val': 'val1', 'test': 'te1'}[sp]
        ZL = CX.zL_table(nm, '%s/ou/read_%s_stu.pkl' % (X, nm))
        zc = np.full(N, np.nan)
        for k, z in ZL.items():
            i = key.get((str(k[0]), str(k[1])))
            if i is not None:
                zc[i] = z['z']
        U = pickle.load(open('%s/ref/inf_fin_det1_%s.pkl' % (X, sp), 'rb')); cls = np.full(N, -1)
        for k, u in U.items():
            i = key.get((str(k[0]), str(k[1])))
            if i is not None:
                cls[i] = CX.cls_of(CC, u['lum'])
        eI = np.log(Pn['pt'][:, 2] / gt[:, 2]); eC = np.log(zc / gt[:, 2])
        ok = tp & np.isfinite(eI) & np.isfinite(eC)
        bI, bC = float(np.median(eI[ok])), float(np.median(eC[ok]))
        out = dict(n_tp=int(tp.sum()), n_used=int(ok.sum()), bias_IR=bI, bias_cor=bC, groups={})
        for g, m in (('overall', ok), ('dark', ok & (cls == 0)), ('bright', ok & (cls == 1))):
            if not m.any():
                continue
            a, b = eI[m] - bI, eC[m] - bC
            out['groups'][g] = dict(n=int(m.sum()), IR_spread=rs(eI[m], bI), cor_spread=rs(eC[m], bC), corr=float(np.corrcoef(a, b)[0, 1]),
                                    spearman=float(spearmanr(a, b)[0]), IR_spread_own=rs(eI[m], np.median(eI[m])), cor_spread_own=rs(eC[m], np.median(eC[m])),
                                    IR_med_resid=float(100 * np.median(a)), cor_med_resid=float(100 * np.median(b)),
                                    IR_bias_own=float(100 * np.median(eI[m])), cor_bias_own=float(100 * np.median(eC[m])))
        res[sp] = out
        print(sp, json.dumps(out, default=lambda x: round(float(x), 4)), flush=True)
    json.dump(res, open(OUT + '/dist_clean.json', 'w', encoding='utf-8'), indent=1, default=float)
