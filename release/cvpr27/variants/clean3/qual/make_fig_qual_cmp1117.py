# -*- coding: utf-8 -*-
"""[11-17] Main-text qualitative figure, one row of 4 crops, single column, with a comparison method.
Boxes: ground truth (cyan), PST (magenta), MonoCoP +ST (orange). Same per-frame rules as select_qual.py
(top-1 box per frame, TP = projected-2D IoU >= 0.1, AOE = quadrotor-symmetric angle).
Runs (as in Tab. 1):
  PST  MAV6D       srcx/in2mav/preds/pcmd_fused_test.pkl            (full pipeline, UDS 60.81)
  PST  RealDrone3D srcx/out2antifs_k249/preds/pcmd_fused_test.pkl   (limited-frame run, UDS 38.69)
  MonoCoP +ST      D:/advisor_results_1117b/runs/monocop_in_st_mav_s0, monocop_out_st_anti_s0
Selection (fixed rule, no image looked at): frames where both methods are TP; percentiles of position error and AOE
are computed per method over that common set; the representative frame minimises max |p - 0.5| over the four percentiles,
i.e. both methods are close to their own median error. MAV6D: one Mavic 2 and one Phantom 4.
RealDrone3D: one near (GT range below the median) and one far (above the median)."""
import os, sys, json, pickle
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as Rot
sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import select_qual as SQ
import make_fig_qual as MF

GP = SQ.GP
RUNS = {'mav': dict(pst=GP + '/srcx/in2mav/preds/pcmd_fused_test.pkl',
                    cmp='D:/advisor_results_1117b/runs/monocop_in_st_mav_s0/pred_real_test.pkl'),
        'anti': dict(pst=GP + '/srcx/out2antifs_k249/preds/pcmd_fused_test.pkl',
                     cmp='D:/advisor_results_1117b/runs/monocop_out_st_anti_s0/pred_outdoor_test.pkl')}
OUTPDF = 'D:/CVPR27_UAV/figures/fig_qual_main.pdf'
OUTPNG = HERE + '/fig_qual_main_cmp1117.png'
C_GT, C_PST, C_CMP = MF.C_GT, MF.C_PRED, '#ff9f1c'


def pf(ds, pred):
    SQ.DS[ds]['pred'] = pred
    M, pb, cf, d, rng_, tp, aoe = SQ.per_frame(ds)
    e = d / rng_ if SQ.DS[ds]['rel'] else d
    return M, pb, d, e, rng_, tp, aoe


def pct(x, idx):
    p = np.full(len(x), np.nan); p[idx] = (np.argsort(np.argsort(x[idx])) + 0.5) / len(idx); return p


def corner_err(b, g, K):
    """mean pixel distance of the 8 projected corners, divided by the GT box's 2D diagonal (visible error)"""
    u, v = MF.proj(b, K), MF.proj(g, K)
    return float(np.linalg.norm(u - v, axis=1).mean() / np.linalg.norm(np.ptp(v, 0)))


def select(ds):
    M, pa, da, ea, rng_, ta, aa = pf(ds, RUNS[ds]['pst'])
    _, pc, dc, ec, _, tc, ac = pf(ds, RUNS[ds]['cmp'])
    T = np.where(ta & tc)[0]
    P = np.stack([pct(ea, T), pct(aa, T), pct(ec, T), pct(ac, T)], 1)
    cls = np.array([str(c) for c in M['cls']]); seq = np.array([str(x) for x in M['seq']])
    # PST at or below its own median on both errors; the comparison method between its own median and 90th percentile on both
    ok = T[(P[T, 0] <= 0.5) & (P[T, 1] <= 0.5) & (P[T, 2] >= 0.5) & (P[T, 3] >= 0.5) & (P[T, 2] <= 0.9) & (P[T, 3] <= 0.9)]   # no outliers of the comparison method
    K = M['K']
    vis = {int(i): corner_err(pc[i], M['gt'][i], K[i]) - corner_err(pa[i], M['gt'][i], K[i]) for i in ok}
    if ds == 'mav':
        groups = [ok[cls[ok] == 'mavic2'], ok[cls[ok] == 'phantom4']]
    else:
        med = np.median(rng_[T]); groups = [ok[rng_[ok] < med], ok[rng_[ok] >= med]]
    rows = []
    for g in groups:
        i = int(max(g, key=lambda j: vis[int(j)]))   # largest visible gap between the two methods in the image
        rows.append(dict(ds=ds, idx=i, seq=str(M['seq'][i]), frame=str(M['frame'][i]), cls=cls[i], range_m=float(rng_[i]),
                         gt=M['gt'][i].tolist(), pst=pa[i].tolist(), cmp=pc[i].tolist(),
                         pst_d=float(da[i]), pst_rel=float(da[i] / rng_[i]), pst_aoe=float(aa[i]),
                         cmp_d=float(dc[i]), cmp_rel=float(dc[i] / rng_[i]), cmp_aoe=float(ac[i]),
                         pct=[float(x) for x in P[i]], vis=vis[i]))
    summ = dict(n_common_tp=int(len(T)), n_pool=int(len(ok)), pst_median_e=float(np.median(ea[T])), pst_median_aoe=float(np.median(aa[T])),
                cmp_median_e=float(np.median(ec[T])), cmp_median_aoe=float(np.median(ac[T])))
    return rows, summ


FIXED = {'mav': [('mavic2/02/0211', '1649918421087831974.jpg'), ('phantom4/01/0109', '172.jpg')],
         'anti': [('train_seq_0010', '000903.jpg'), ('train_seq_0010', '006144.jpg')]}


def fixed(ds):
    M, pa, da, ea, rng_, ta, aa = pf(ds, RUNS[ds]['pst'])
    _, pc, dc, ec, _, tc, ac = pf(ds, RUNS[ds]['cmp'])
    key = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(M['seq'], M['frame']))}
    rows = []
    for sq, fr in FIXED[ds]:
        i = key[(sq, fr)]
        rows.append(dict(ds=ds, idx=int(i), seq=sq, frame=fr, cls=str(M['cls'][i]), range_m=float(rng_[i]),
                         gt=M['gt'][i].tolist(), pst=pa[i].tolist(), cmp=pc[i].tolist(),
                         pst_d=float(da[i]), pst_rel=float(da[i] / rng_[i]), pst_aoe=float(aa[i]),
                         cmp_d=float(dc[i]), cmp_rel=float(dc[i] / rng_[i]), cmp_aoe=float(ac[i])))
    return rows


ASP = 0.62   # crop height / width


def panel(ax, r, meta, which, label=None):
    """one method per panel: ground truth + that method's box; the crop window is the same for both methods of a frame"""
    img = MF.imread(meta)[:, :, ::-1]; H, W = img.shape[:2]; K = np.asarray(meta['K_raw'], np.float64)
    U = {k: MF.proj(r[k], K) for k in ('gt', 'pst', 'cmp')}
    ug = U['gt']; allp = np.concatenate(list(U.values()))
    cx, cy = (ug.min(0) + ug.max(0)) / 2
    ext = max(np.ptp(ug[:, 0]), np.ptp(ug[:, 1]) / ASP)
    hw = max(1.25 * ext, 1.08 * np.abs(allp[:, 0] - cx).max(), 1.08 / ASP * np.abs(allp[:, 1] - cy).max(), 48 if r['ds'] == 'anti' else 110)
    hh = hw * ASP
    x0, y0, x1, y1 = (int(round(v)) for v in (cx - hw, cy - hh, cx + hw, cy + hh))
    pad = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    TW = 800; s = TW / pad.shape[1]
    crop = cv2.resize(pad, (TW, int(round(pad.shape[0] * s))), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    ax.imshow(crop, interpolation='none'); ax.set_xlim(0, crop.shape[1]); ax.set_ylim(crop.shape[0], 0)
    col = C_PST if which == 'pst' else C_CMP
    for k, c in (('gt', C_GT), (which, col)):
        v = (U[k] - [x0, y0]) * s
        for a, b in MF.EDGES:
            ax.plot([v[a, 0], v[b, 0]], [v[a, 1], v[b, 1]], color=c, lw=0.8, solid_capstyle='round')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    bx = dict(boxstyle='square,pad=0.1', fc='black', ec='none', alpha=0.6)
    if label:
        ax.text(0.03, 0.04, label, transform=ax.transAxes, ha='left', va='bottom', color='white', fontsize=6.0, bbox=bx)
    d = (('%.0f cm' % (100 * r[which + '_d'])) if r['ds'] == 'mav' else ('%.0f%%' % (100 * r[which + '_rel'])))
    ax.text(0.97, 0.95, '$\\Delta d$ ' + d, transform=ax.transAxes, ha='right', va='top', color=col, fontsize=6.0, fontweight='bold', bbox=bx)
    return [x0, y0, x1, y1]


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    rows, summ = [], {}
    for ds in ('mav', 'anti'):
        r_, s_ = select(ds); summ[ds] = s_
        # 11-17 user: show frames where PST fits well. Candidates ranked by a small PST 2D corner error and a larger
        # MonoCoP +ST error (cand2.png / cand_anti.png); the four below were picked by eye from the top of those rankings.
        rows += fixed(ds)
    metas = {}
    for ds, p in MF.CACHE.items():
        I = pickle.load(open(p, 'rb')); metas[ds] = {(m['seq'], m['frame']): m for m in I['metas']}
    FW = 3.33; LW = 0.20; gap = 0.02; vg = 0.02; pw = (FW - LW - 3 * gap) / 4; ph = pw * ASP; hdr = 0.12
    FH = hdr + 2 * ph + vg
    fig = plt.figure(figsize=(FW, FH))
    for row, (which, name, col) in enumerate((('cmp', 'MonoCoP\n+ST', C_CMP), ('pst', 'PST\n(ours)', C_PST))):
        yb = (1 - row) * (ph + vg)
        for k, r in enumerate(rows):
            ax = fig.add_axes([(LW + k * (pw + gap)) / FW, yb / FH, pw / FW, ph / FH])
            lab = (('%.1f m' % r['range_m']) if r['ds'] == 'mav' else ('%.0f m' % r['range_m'])) if row == 0 else None
            r['crop_px'] = panel(ax, r, metas[r['ds']][(r['seq'], r['frame'])], which, lab)
        fig.text(LW * 0.45 / FW, (yb + ph / 2) / FH, name, rotation=90, ha='center', va='center', fontsize=6.0, fontweight='bold', color=col, linespacing=1.0, multialignment='center')
    for j, t in ((0, 'MAV6D'), (1, 'RealDrone3D')):
        xc = (LW + j * 2 * (pw + gap) + pw + gap / 2) / FW
        fig.text(xc, (FH - hdr / 2) / FH, t, ha='center', va='center', fontsize=6.0, fontweight='bold')
    fig.savefig(OUTPDF, dpi=600); fig.savefig(OUTPNG, dpi=600)
    json.dump(dict(rows=rows, summary=summ, runs=RUNS), open(HERE + '/selection_cmp1117.json', 'w'), indent=1)
    for r in rows:
        print(r['ds'], r['seq'], r['frame'], r['cls'], 'range %.1f | PST d %.3f (%.1f%%) aoe %.1f | MonoCoP+ST d %.3f (%.1f%%) aoe %.1f' % (
            r['range_m'], r['pst_d'], 100 * r['pst_rel'], r['pst_aoe'], r['cmp_d'], 100 * r['cmp_rel'], r['cmp_aoe']))
    print('size in', fig.get_size_inches())
