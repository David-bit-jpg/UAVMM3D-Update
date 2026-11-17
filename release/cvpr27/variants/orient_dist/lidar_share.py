# -*- coding: utf-8 -*-
"""[review1113 orient_dist C2b] share of simulated outdoor TRAINING objects (D:/S2MUAV3D/sim_outdoor_carla/cache/train, all labelled boxes
of the 5,367 cached frames) with a non-empty LiDAR body map, per GT range band. CPU only, reads raw CARLA LiDAR.
Body-map rule = lidar/kzlid/center_det_mm.py targets(): non-empty iff >= 5 drone-tagged LiDAR hits with |z_hit - z_box| < 0.75 * max(lwh) + 0.3.
Here counted as RAW returns (not the 5x5-dilated sidecar pixels the network sees), using the sidecar's projection exactly
(tools/build_mm_sidecar_obj.py: lidar_1 frame index + lidar_offset 6, LiDAR extrinsic -> world -> RGB camera, K_raw, in-image points);
a return is assigned to a box if it projects inside the box's projected 2-D extent expanded by 30 % + 4 px (the sidecar's per-object window)."""
import os, sys, pickle, json, collections
import numpy as np
from scipy.spatial.transform import Rotation as Rot
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, 'E:/Open3DUAVDet/tools'); sys.path.insert(0, 'E:/Open3DUAVDet')
import build_mm_cache as BM  # noqa
from build_near_subset import frame_sort_key  # noqa
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = 'E:/data_collect'
D = pickle.load(open('D:/S2MUAV3D/sim_outdoor_carla/cache/train/index.pkl', 'rb')); OFF = int(D.get('lidar_offset', 6) or 6)
P8 = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
SMALL = {'DJI-avata2', 'DJI-mavic-mini', 'DJI-phantom4', 'drone-unk3'}
FR, INFO = {}, {}
rows = []
for n, m in enumerate(D['metas']):
    if not m:
        continue
    base = os.path.join(ROOT, m['seq'])
    if base not in FR:
        FR[base] = sorted([f for f in os.listdir(base + '/images_rgb') if f.endswith('.png')], key=frame_sort_key)
        INFO[base] = (pickle.load(open(base + '/im_info.pkl', 'rb')), pickle.load(open(base + '/lidar_radar_info.pkl', 'rb')))
    info, lr = INFO[base]; E = np.array(info['rgb']['extrinsic'], np.float64); Le = np.array(lr['lidars'][0]['extrinsic'], np.float64)
    K = np.asarray(m['K_raw'], np.float64); W, H = m['raw_wh']; fr = FR[base]
    li = min(fr.index(m['frame']) + OFF, len(fr) - 1); f = base + '/lidar_1/' + os.path.splitext(fr[li])[0] + '.npy'
    b = np.asarray(m['boxes9d'], np.float64).reshape(-1, 9); q = np.asarray(m['qualified'], bool)
    if os.path.exists(f):
        P = np.load(f); P = P[P[:, 4] == 1]
        w = (Le @ np.c_[P[:, :3], np.ones(len(P))].T).T
        cv = (BM.CARLA_TO_OPENCV @ (np.linalg.inv(E) @ w.T)).T[:, :3]; cv = cv[cv[:, 2] > 0.1]
        uv = (K @ cv.T).T; uv = uv[:, :2] / uv[:, 2:3]
        ins = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H); uv, z = uv[ins], cv[ins, 2]
    else:
        uv, z = np.zeros((0, 2)), np.zeros(0)
    for k, bk in enumerate(b):
        cor = (P8 * bk[3:6]) @ Rot.from_euler('xyz', bk[6:9]).as_matrix().T + bk[:3]
        L = max(float(np.max(bk[3:6])), 0.05); cnt = 0; wpx = np.nan
        if (cor[:, 2] > 0.1).all():
            u = (K @ cor.T).T; u = u[:, :2] / u[:, 2:3]; x0, y0 = u.min(0); x1, y1 = u.max(0); wpx = x1 - x0
            mw, mh = 0.3 * (x1 - x0) + 4, 0.3 * (y1 - y0) + 4
            sel = (uv[:, 0] >= x0 - mw) & (uv[:, 0] <= x1 + mw) & (uv[:, 1] >= y0 - mh) & (uv[:, 1] <= y1 + mh) & (np.abs(z - bk[2]) < 0.75 * L + 0.3)
            cnt = int(sel.sum())
        rows.append(dict(name=m['names'][k], rng=float(np.linalg.norm(bk[:3])), n=cnt, q=bool(q[k]), small=m['names'][k] in SMALL, wpx=float(wpx), lidar=os.path.exists(f)))
    if len(rows) % 2000 < len(b):
        print(n, len(rows), flush=True)
pickle.dump(rows, open(OUT + '/lidar_share_rows.pkl', 'wb'))
BANDS = ((0, 20), (20, 40), (40, 60), (60, 999)); res = {}
for grp, f in (('all objects', lambda r: True), ('small airframes', lambda r: r['small']), ('large airframes', lambda r: not r['small']),
               ('qualified objects', lambda r: r['q'])):
    res[grp] = {}
    for lo, hi in BANDS:
        R = [r for r in rows if f(r) and lo <= r['rng'] < hi]; c = np.array([r['n'] for r in R])
        res[grp][('%g-%g' % (lo, hi)).replace('-999', '+')] = dict(n_obj=len(R), share_nonempty=float(np.mean(c >= 5)) if len(R) else np.nan,
                                                                   median_returns=float(np.median(c)) if len(R) else np.nan, share_zero=float(np.mean(c == 0)) if len(R) else np.nan)
res['n_frames'] = sum(1 for m in D['metas'] if m); res['n_missing_lidar'] = sum(1 for r in rows if not r['lidar']); res['lidar_offset'] = OFF
json.dump(res, open(OUT + '/lidar_share.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
