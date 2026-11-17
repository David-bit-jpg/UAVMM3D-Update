# -*- coding: utf-8 -*-
"""[clean3 item 3b] Fig. 3: qualitative results, single column (3.33 in). Reads selection.json (select_qual.py); raw frames:
MAV6D = distorted JPEG under E:/MAV6D undistorted to K_raw exactly as srcx/lib/sx.imread_raw; AntiUAV3D = raw_path of the cache metas (pinhole).
Boxes: 8 corners (pack PROTO order) projected with K_raw, 12 edges drawn as vector lines over the embedded raster crop."""
import os, sys, json, pickle
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, 'E:/Open3DUAVDet/tools'); sys.path.insert(0, 'E:/Open3DUAVDet')
from build_mav6d_cache import K_CALIB, D_CALIB  # noqa
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
PDF = 'D:/CVPR27_UAV/figures/fig_qual.pdf'; PNG = OUT.rsplit('/', 1)[0] + '/fig_qual.png'
SEL = json.load(open(OUT + '/selection.json'))
CACHE = {'mav': 'E:/mmcache/mav6d_cn_obj/test/index.pkl', 'anti': 'D:/mmcache/outdoor_real_fs/test/index.pkl'}
P8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
C_PRED, C_GT = '#ff2d95', '#00e0ff'     # magenta = prediction, cyan = ground truth
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 5.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
_MAPS = {}
CLS = {'mavic2': 'Mavic 2', 'phantom4': 'Phantom 4', 'M4E': 'Matrice 4E', 'M400': 'Matrice 400'}


def imread(m):
    if 'raw_path' in m:
        return cv2.imread(m['raw_path'], cv2.IMREAD_COLOR)
    cls, scene, sq = m['seq'].split('/')
    img = cv2.imread(os.path.join('E:/MAV6D', cls, 'JPEGImages', scene, sq, m['frame']), cv2.IMREAD_COLOR)
    assert img is not None
    K_raw = np.asarray(m['K_raw'], np.float64); key = tuple(np.round(K_raw.reshape(-1), 6))
    if key not in _MAPS:
        _MAPS[key] = cv2.initUndistortRectifyMap(K_CALIB, D_CALIB, None, K_raw, tuple(int(v) for v in m['raw_wh']), cv2.CV_32FC1)
    return cv2.remap(img, *_MAPS[key], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def proj(b, K):
    b = np.asarray(b, np.float64); c = (P8 * b[3:6]) @ Rot.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
    q = c @ K.T
    return q[:, :2] / q[:, 2:3]


def panel(ax, ds, r, meta, letter):
    img = imread(meta)[:, :, ::-1]; H, W = img.shape[:2]; K = np.asarray(meta['K_raw'], np.float64)
    ug, up = proj(r['gt'], K), proj(r['pred'], K)
    gx0, gy0 = ug.min(0); gx1, gy1 = ug.max(0); allp = np.r_[ug, up]
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    ext = max(gx1 - gx0, (gy1 - gy0) * 4 / 3)
    hw = max(1.25 * ext, 1.15 * np.abs(allp[:, 0] - cx).max(), 1.15 * 4 / 3 * np.abs(allp[:, 1] - cy).max(), 48 if ds == 'anti' else 110)
    hh = hw * 3 / 4
    x0, y0 = int(round(cx - hw)), int(round(cy - hh)); x1, y1 = int(round(cx + hw)), int(round(cy + hh))
    pad = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)       # black padding if the window leaves the image
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    s = 480.0 / pad.shape[1]
    crop = cv2.resize(pad, (480, int(round(pad.shape[0] * s))), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    ax.imshow(crop, interpolation='none'); ax.set_xlim(0, crop.shape[1]); ax.set_ylim(crop.shape[0], 0)
    for u, col in ((ug, C_GT), (up, C_PRED)):
        v = (u - [x0, y0]) * s
        for a, b in EDGES:
            ax.plot([v[a, 0], v[b, 0]], [v[a, 1], v[b, 1]], color=col, lw=0.7, solid_capstyle='round')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    kind = {'success': 'success', 'typical': 'typical', 'failure (position)': 'position failure',
            'failure (orientation)': 'orient. failure'}[r['kind']]
    kcol = {'success': '#1a7f37', 'typical': '#555555'}.get(r['kind'], '#b42318')
    ax.text(0.02, 0.97, '(%s) %s' % (letter, kind), transform=ax.transAxes, ha='left', va='top', color='white', fontsize=5.0,
            bbox=dict(boxstyle='square,pad=0.15', fc=kcol, ec='none', alpha=0.9))
    if ds == 'mav':
        txt = '%s, %.1f m\n$\\Delta d$ %.2f m, AOE %.0f\u00b0' % (CLS[r['cls']], r['range_m'], r['dist_err_m'], r['aoe'])
    else:
        txt = '%s, %.0f m\n$\\Delta d$ %.1f m (%.0f%%), AOE %.0f\u00b0' % (CLS[r['cls']], r['range_m'], r['dist_err_m'], 100 * r['dist_err_rel'], r['aoe'])
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, ha='left', va='bottom', color='white', fontsize=5.0, linespacing=1.1,
            bbox=dict(boxstyle='square,pad=0.15', fc='black', ec='none', alpha=0.6))
    if ds == 'anti':      # full-frame thumbnail with the crop window, so the drone's true size is visible
        ia = ax.inset_axes([0.69, 0.69, 0.30, 0.30]); th = cv2.resize(img, (192, 108), interpolation=cv2.INTER_AREA)
        ia.imshow(th, interpolation='none'); ia.set_xticks([]); ia.set_yticks([])
        for sp in ia.spines.values():
            sp.set_color('white'); sp.set_linewidth(0.4)
        f = 192.0 / W
        ia.add_patch(Rectangle((x0 * f, y0 * f), max((x1 - x0) * f, 2.5), max((y1 - y0) * f, 2.5), fill=False, ec='yellow', lw=0.5))
    return dict(crop_px=[x0, y0, x1, y1], scale=s)


if __name__ == '__main__':
    metas = {}
    for ds, p in CACHE.items():
        I = pickle.load(open(p, 'rb')); metas[ds] = {(m['seq'], m['frame']): m for m in I['metas']}
    FW = 3.33; pw = (FW - 0.03) / 3; phh = pw * 0.75
    legend_h, hdr_h, gap_h = 0.16, 0.14, 0.04
    FH = legend_h + 2 * hdr_h + gap_h + 4 * phh + 0.02
    fig = plt.figure(figsize=(FW, FH))
    info = {}; L = iter('abcdefghijkl'); y = FH - legend_h
    for ds, title in (('mav', 'MAV6D test (indoor, 1.6\u20135.6 m)'), ('anti', 'AntiUAV3D test (outdoor, 13\u201380 m)')):
        fig.text(0.5, (y - hdr_h / 2) / FH, title, ha='center', va='center', fontsize=6.0, fontweight='bold'); y -= hdr_h
        for k, r in enumerate(SEL[ds]['rows']):
            rr, cc = divmod(k, 3)
            ax = fig.add_axes([(0.005 + cc * (pw + 0.01)) / FW, (y - (rr + 1) * phh + 0.01) / FH, (pw - 0.005) / FW, (phh - 0.02) / FH])
            info['%s_%d' % (ds, k)] = panel(ax, ds, r, metas[ds][(r['seq'], r['frame'])], next(L))
        y -= 2 * phh + gap_h
    fig.legend(handles=[Line2D([], [], color=C_PRED, lw=1.0, label='prediction (PST)'), Line2D([], [], color=C_GT, lw=1.0, label='ground truth')],
               loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=5.5, handlelength=1.8, borderaxespad=0.1)
    os.makedirs(os.path.dirname(PDF), exist_ok=True)
    fig.savefig(PDF, dpi=400); fig.savefig(PNG, dpi=400)
    json.dump(info, open(OUT + '/fig_qual_crops.json', 'w'), indent=1)
    print('wrote', PDF, PNG, fig.get_size_inches())
