# -*- coding: utf-8 -*-
"""[11-17] Main-text qualitative figure, 4 rows x 4 columns: three RGB self-training baselines of Tab. 1 and PST.
Same four frames as make_fig_qual_cmp1117.py (FIXED, chosen where PST fits well). Each panel: ground truth (cyan) +
one method's top-1 box, Δd of that method top right. One crop window per column, shared by all rows.
Baseline runs (seed 0 of each Tab. 1 row): advisor_results_1117b/runs/<det>_in_st_mav_s0, <det>_out_st_anti_s0."""
import os, sys, json, pickle
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_fig_qual_cmp1117 as C
MF = C.MF
R = 'D:/advisor_results_1117b/runs/'
METHODS = [('monocop', 'MonoCoP\n+ST', '#ff9f1c', {'mav': R + 'monocop_in_st_mav_s0/pred_real_test.pkl', 'anti': R + 'monocop_out_st_anti_s0/pred_outdoor_test.pkl'}),
           ('monodgp', 'MonoDGP\n+ST', '#f0e442', {'mav': R + 'monodgp_in_st_mav_s0/pred_real_test.pkl', 'anti': R + 'monodgp_out_st_anti_s0/pred_outdoor_test.pkl'}),
           ('centernet', 'CenterNet\n+ST', '#3ddc84', {'mav': 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_st_test.pkl', 'anti': 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_st_test.pkl'}),
           ('pst', 'PST', C.C_PST, {'mav': C.RUNS['mav']['pst'], 'anti': C.RUNS['anti']['pst']})]
OUTPDF = 'D:/CVPR27_UAV/figures/fig_qual_main.pdf'; OUTPNG = HERE + '/fig_qual_main_cmp4_1117.png'
ASP = 0.58


def collect():
    cols = []
    for ds in ('mav', 'anti'):
        res = {}
        for key, _, _, runs in METHODS:
            M, pb, d, e, rng_, tp, aoe = C.pf(ds, runs[ds]); res[key] = (pb, d, rng_, tp, aoe)
        idx = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(M['seq'], M['frame']))}
        for sq, fr in C.FIXED[ds]:
            i = idx[(sq, fr)]
            c = dict(ds=ds, seq=sq, frame=fr, cls=str(M['cls'][i]), range_m=float(res['pst'][2][i]), gt=M['gt'][i].tolist(), m={})
            for key, *_ in METHODS:
                pb, d, rng_, tp, aoe = res[key]
                ok = bool(np.isfinite(pb[i]).all())
                c['m'][key] = dict(box=pb[i].tolist() if ok else None, d=float(d[i]) if ok else None, rel=float(d[i] / rng_[i]) if ok else None,
                                   aoe=float(aoe[i]) if np.isfinite(aoe[i]) else None, tp=bool(tp[i]))
            cols.append(c)
    return cols


def window(c, meta):
    K = np.asarray(meta['K_raw'], np.float64)
    ug = MF.proj(c['gt'], K); pts = [ug, MF.proj(c['m']['pst']['box'], K)]   # zoom on the UAV: window from GT and PST; baseline boxes may run past the edge
    allp = np.concatenate(pts); cx, cy = (ug.min(0) + ug.max(0)) / 2
    ext = max(np.ptp(ug[:, 0]), np.ptp(ug[:, 1]) / ASP)
    far = np.abs(allp - [cx, cy])
    hw = max(1.0 * ext, 1.15 * far[:, 0].max(), 1.15 / ASP * far[:, 1].max(), 40 if c['ds'] == 'anti' else 100)
    return K, [int(round(v)) for v in (cx - hw, cy - hw * ASP, cx + hw, cy + hw * ASP)]


def panel(ax, img, K, win, c, key, col, label):
    H, W = img.shape[:2]; x0, y0, x1, y1 = win
    pad = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    TW = 800; s = TW / pad.shape[1]
    crop = cv2.resize(pad, (TW, int(round(pad.shape[0] * s))), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    ax.imshow(crop, interpolation='none'); ax.set_xlim(0, crop.shape[1]); ax.set_ylim(crop.shape[0], 0)
    v = c['m'][key]
    for b, cc in ((c['gt'], C.C_GT), (v['box'] if v['tp'] else None, col)):
        if b is None:
            continue
        u = (MF.proj(b, K) - [x0, y0]) * s
        for a, bb in MF.EDGES:
            ax.plot([u[a, 0], u[bb, 0]], [u[a, 1], u[bb, 1]], color=cc, lw=0.75, solid_capstyle='round')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    bx = dict(boxstyle='square,pad=0.08', fc='black', ec='none', alpha=0.6)
    if label:
        ax.text(0.03, 0.05, label, transform=ax.transAxes, ha='left', va='bottom', color='white', fontsize=6.0, bbox=bx)
    if v['box'] is None or not v['tp']:
        t = 'missed'   # top-1 box does not overlap the UAV (2D IoU < 0.1), as in the scorer
    else:
        t = '$\Delta d$ ' + (('%.2f m' % v['d']) if c['ds'] == 'mav' else ('%.1f m' % v['d']))   # one unit (m) everywhere
    ax.text(0.97, 0.94, t, transform=ax.transAxes, ha='right', va='top', color=col, fontsize=6.0, fontweight='bold', bbox=bx)


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    cols = collect()
    metas = {}
    for ds, p in MF.CACHE.items():
        I = pickle.load(open(p, 'rb')); metas[ds] = {(m['seq'], m['frame']): m for m in I['metas']}
    FW = 3.33; LW = 0.42; gap = 0.02; vg = 0.012; pw = (FW - LW - 3 * gap) / 4; ph = pw * ASP; hdr = 0.21
    NR = len(METHODS); FH = hdr + NR * ph + (NR - 1) * vg
    fig = plt.figure(figsize=(FW, FH))
    for k, c in enumerate(cols):
        meta = metas[c['ds']][(c['seq'], c['frame'])]; img = MF.imread(meta)[:, :, ::-1]
        K, win = window(c, meta); c['crop_px'] = win
        for row, (key, name, col, _) in enumerate(METHODS):
            yb = (NR - 1 - row) * (ph + vg)
            ax = fig.add_axes([(LW + k * (pw + gap)) / FW, yb / FH, pw / FW, ph / FH])
            lab = None
            panel(ax, img, K, win, c, key, col, lab)
    for row, (key, name, col, _) in enumerate(METHODS):
        yb = (NR - 1 - row) * (ph + vg)
        fig.text((LW - 0.03) / FW, (yb + ph / 2) / FH, name, ha='right', va='center', fontsize=6.0, fontweight='bold',
                 color=col if key != 'monodgp' else '#b8a800', linespacing=1.0, multialignment='right')
    for k, c in enumerate(cols):   # column header: airframe and true distance
        t = MF.CLS[c['cls']] + (', %.1f m' if c['ds'] == 'mav' else ', %.0f m') % c['range_m']   # per column: airframe, true distance
        fig.text((LW + k * (pw + gap) + pw / 2) / FW, (FH - 0.75 * hdr) / FH, t, ha='center', va='center', fontsize=6.0, fontweight='bold')
    for j, t in ((0, 'MAV6D'), (1, 'RealDrone3D')):   # dataset name centred over its two columns
        fig.text((LW + j * 2 * (pw + gap) + pw + gap / 2) / FW, (FH - 0.27 * hdr) / FH, t, ha='center', va='center', fontsize=6.0, fontweight='bold')
    fig.savefig(OUTPDF, dpi=600); fig.savefig(OUTPNG, dpi=600)
    json.dump(dict(cols=cols, methods=[(m[0], m[3]) for m in METHODS]), open(HERE + '/selection_cmp4_1117.json', 'w'), indent=1)
    for c in cols:
        print(c['ds'], c['seq'], c['frame'], 'range %.1f' % c['range_m'])
        for key, *_ in METHODS:
            v = c['m'][key]
            print('   %-8s' % key, 'none' if v['box'] is None else 'd %.3f m (%.1f%%) aoe %s tp %s' % (v['d'], 100 * v['rel'], 'nan' if v['aoe'] is None else '%.1f' % v['aoe'], v['tp']))
    print('size in', fig.get_size_inches())
