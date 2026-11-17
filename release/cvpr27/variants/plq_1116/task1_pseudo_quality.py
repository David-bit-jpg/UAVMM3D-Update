# -*- coding: utf-8 -*-
"""[plq_1116 task 1] Pseudo-box quality on MAV6D real TRAIN frames: conventional self-training tables vs PST. CPU only.
GT (E:/mmcache/mav6d_cn_obj/train/index.pkl, same as figures/data/pl_quality_mav6d.csv) is used ONLY to report, never to select.
All boxes: box9d = centre (camera frame of K_in, undistorted 512x288), lwh, euler 'xyz' (Rot.from_euler('xyz')), as the pack scorer.
Usage: python task1_pseudo_quality.py"""
import os, sys, pickle, json, glob
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot
OUT = os.path.dirname(os.path.abspath(__file__))
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
PROTO = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count('1') == 1]
P = lambda f: pickle.load(open(f, 'rb'))


def sym(Rp, Rg):   # identical to D:/UAV3D_baselines_pack/scripts/score.py sym
    return min(np.degrees(np.arccos(np.clip((np.trace((Rp @ RZ[q]).T @ Rg) - 1) / 2, -1, 1))) for q in range(4))


def E(e):
    return Rot.from_euler('xyz', np.asarray(e, np.float64)).as_matrix()


def corners(b):
    return (PROTO * b[3:6]) @ E(b[6:9]).T + b[:3]


def inside(pts, b, eps=1e-9):
    loc = (pts - b[:3]) @ E(b[6:9])          # world -> box frame
    return np.all(np.abs(loc) <= b[3:6] / 2 + eps, axis=1)


def clip_edges(ca, b):
    """intersections of the 12 edges of polytope with corners ca against the 6 face planes of box b (kept if on b's surface)"""
    R = E(b[6:9]); h = b[3:6] / 2; out = []
    la = (ca - b[:3]) @ R
    for i, j in EDGES:
        p, q = la[i], la[j]; d = q - p
        for ax in range(3):
            if abs(d[ax]) < 1e-12:
                continue
            for s in (-1, 1):
                t = (s * h[ax] - p[ax]) / d[ax]
                if 0 <= t <= 1:
                    x = p + t * d
                    if np.all(np.abs(x) <= h + 1e-9):
                        out.append(x @ R.T + b[:3])
    return out


def iou3d(a, b):
    """exact oriented 3-D IoU of two boxes (general rotation): hull of {A corners in B, B corners in A, edge-face crossings both ways}"""
    if np.linalg.norm(a[:3] - b[:3]) > 0.5 * (np.linalg.norm(a[3:6]) + np.linalg.norm(b[3:6])):
        return 0.0
    ca, cb = corners(a), corners(b)
    pts = list(ca[inside(ca, b)]) + list(cb[inside(cb, a)]) + clip_edges(ca, b) + clip_edges(cb, a)
    if len(pts) < 4:
        return 0.0
    try:
        vi = ConvexHull(np.array(pts)).volume
    except Exception:
        return 0.0
    va, vb = np.prod(a[3:6]), np.prod(b[3:6])
    return float(vi / (va + vb - vi))


# --- self-test of the IoU
_b = np.array([0.1, 0.2, 3.0, 0.3, 0.4, 0.15, 0.3, -0.2, 1.1])
assert abs(iou3d(_b, _b) - 1) < 1e-6
_u = np.array([0, 0, 5, 1, 1, 1, 0, 0, 0.]); _v = _u.copy(); _v[0] = 0.5
assert abs(iou3d(_u, _v) - 1 / 3) < 1e-6
_w = _u.copy(); _w[8] = np.pi / 4                         # cube vs same cube rotated 45 deg about z: inter = octagon area 2(sqrt2-1) = 0.8284
assert abs(iou3d(_u, _w) - (2 * (np.sqrt(2) - 1)) / (2 - 2 * (np.sqrt(2) - 1))) < 1e-6
_r = _b.copy(); _r[6:9] = Rot.from_matrix(E(_b[6:9]) @ RZ[1]).as_euler('xyz'); _r[3], _r[4] = _b[4], _b[3]   # same physical box: body-z +90 deg and swapped l/w
assert abs(iou3d(_b, _r) - 1) < 1e-6
print('IoU self-test ok (identical=1, half-shift cube=1/3, 45deg cube analytic, yaw+90 & swapped lw = 1)')

# --- GT
I = P('E:/mmcache/mav6d_cn_obj/train/index.pkl')
GT = {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64)[0] for m in I['metas'] if m is not None and len(m['boxes9d'])}
print('GT frames', len(GT))

# --- tables: name -> {key: box9d}
T = {}; NOTE = {}


def from_list_boxes(L):     # make_pseudo.py labels format
    return {(d['seq'], d['frame']): np.r_[d['boxes'][0]['center'], d['boxes'][0]['lwh'], d['boxes'][0]['euler_xyz']].astype(np.float64) for d in L if d['boxes']}


def top1(L):                # predict.py output: top-1 box per frame
    o = {}
    for r in L:
        sc = np.asarray(r['scores'], np.float64).reshape(-1); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9)
        if len(sc):
            o[(r['seq'], r['frame'])] = b[int(np.argmax(sc))]
    return o


def ixtab(f):
    IX = P(f)
    return {(IX['metas'][int(i)]['seq'], IX['metas'][int(i)]['frame']): np.asarray(IX['metas'][int(i)]['boxes9d'], np.float64)[0] for i in IX['valid_idx']}


# sim-only detectors on all train frames (reference)
D0 = P(GP + '/srcx/in2mav/recs/det0.pkl')['recs']['train']
T['CenterNet sim-only (all frames)'] = {(r['seq'], r['frame']): np.asarray(r['box'], np.float64) for r in D0 if r['box'] is not None}
NOTE['CenterNet sim-only (all frames)'] = 'srcx/in2mav/recs/det0.pkl (3-seed sim-only ensemble + flip, own box incl. own size); no filtering'
T['MonoCD sim-only (all frames)'] = top1(P('D:/cmpfull_runs/monocd_st_real_s0/pred_real_train_by_sim.pkl'))
NOTE['MonoCD sim-only (all frames)'] = 'D:/cmpfull_runs/monocd_st_real_s0/pred_real_train_by_sim.pkl (local sim-only MonoCD seed 0), top-1 per frame, no filtering'
# conventional self-training tables
T['MonoCD +ST'] = from_list_boxes(P('D:/cmpfull_runs/monocd_st_real_s0/pseudo_real_train.pkl'))
NOTE['MonoCD +ST'] = 'D:/cmpfull_runs/monocd_st_real_s0/pseudo_real_train.pkl (make_pseudo.py: own top-1, top-80% score)'
# regenerated make_pseudo.py tables (if other baselines' train predictions were found) live in OUT/regen_*.pkl
for f in sorted(glob.glob(OUT + '/regen_*_pseudo.pkl')):
    nm = os.path.basename(f)[6:-11] + ' +ST (regenerated)'; T[nm] = from_list_boxes(P(f)); NOTE[nm] = f
T['CenterNet +ST (naive)'] = ixtab('E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav_naive.pkl')
NOTE['CenterNet +ST (naive)'] = 'index_pseudo_srcx_in2mav_naive.pkl (review1113/B4: own centre, s_bar, own rotation; frames conf>=0.5 = same frames as PST)'
# PST
C = P('D:/UAV3D_pack_fix_1114/data/pcmd_teacher_pseudo_real_train_clean.pkl')
T['PST pseudo labels (+PST, step 1)'] = {(d['seq'], d['frame']): np.r_[d['center'], d['lwh'], d['euler_xyz']].astype(np.float64) for d in C}
NOTE['PST pseudo labels (+PST, step 1)'] = 'D:/UAV3D_pack_fix_1114/data/pcmd_teacher_pseudo_real_train_clean.pkl (centre = step-1 IR-corrected, size s_bar, orientation = teacher)'
# PST step-3 fused depth target (chain_x.cmd_targets): z_target = z_det0 * exp(dp); centre = step-1 pseudo centre rescaled to z_target (same ray)
meta = P(GP + '/srcx/in2mav/ref/real_d0_meta.pkl'); dp = np.load(GP + '/srcx/in2mav/ref/dp_fused.npy')
st1 = T['PST pseudo labels (+PST, step 1)']; t3 = {}
for m, d in zip(meta, dp):
    k = (m['seq'], m['frame'])
    if np.isfinite(d) and k in st1:
        b = st1[k].copy(); zt = float(m['z_det']) * np.exp(float(d)); b[:3] = b[:3] * zt / b[2]; t3[k] = b
T['PST step-3 fused depth target'] = t3
NOTE['PST step-3 fused depth target'] = 'srcx/in2mav/ref/dp_fused.npy + real_d0_meta.pkl (log(z_target/z_det0)); centre = step-1 centre moved along its ray to z_target; size/orientation as step 1 (only depth is a step-3 target)'
# check MonoCD R vs euler convention on a few frames
L = P('D:/cmpfull_runs/monocd_st_real_s0/pseudo_real_train.pkl')
print('MonoCD R==from_euler(xyz) max diff', max(np.abs(E(d['boxes'][0]['euler_xyz']) - d['boxes'][0]['R']).max() for d in L[:200]))


def evaluate(tab, keys):
    rows = []
    for k in keys:
        b = tab[k]; g = GT[k]
        rows.append((abs(np.linalg.norm(b[:3]) / np.linalg.norm(g[:3]) - 1), abs(b[2] / g[2] - 1), np.linalg.norm(b[:3] - g[:3]),
                     iou3d(b, g), sym(E(b[6:9]), E(g[6:9])), np.linalg.norm(b[:3]) / np.linalg.norm(g[:3]) - 1, 100 * np.mean(np.abs(b[3:6] - g[3:6]))))
    a = np.array(rows)
    return dict(n=len(a), relR_med=100 * np.median(a[:, 0]), relR_mean=100 * np.mean(a[:, 0]), relZ_med=100 * np.median(a[:, 1]), relZ_mean=100 * np.mean(a[:, 1]),
                signed_relR_med=100 * np.median(a[:, 5]), ctr_err_med_m=np.median(a[:, 2]), ctr_err_mean_m=np.mean(a[:, 2]),
                iou_gt0=100 * np.mean(a[:, 3] > 0), iou_ge025=100 * np.mean(a[:, 3] >= 0.25), iou_ge05=100 * np.mean(a[:, 3] >= 0.5), iou_mean=np.mean(a[:, 3]),
                ori_med=np.median(a[:, 4]), ori_mean=np.mean(a[:, 4]), size_err_med_cm=np.median(a[:, 6]))


res = []
for nm, tab in T.items():
    keys = [k for k in tab if k in GT]
    r = evaluate(tab, keys); r.update(table=nm, frames='own', n_table=len(tab)); res.append(r); print(nm, 'own', {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
common = set(GT)
for nm in T:
    if 'all frames' not in nm:
        common &= set(T[nm])
common = sorted(common)
for nm, tab in T.items():
    r = evaluate(tab, common); r.update(table=nm, frames='common', n_table=len(tab)); res.append(r); print(nm, 'common', {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
df = pd.DataFrame(res)
cols = ['table', 'frames', 'n_table', 'n', 'relR_med', 'relR_mean', 'relZ_med', 'relZ_mean', 'signed_relR_med', 'ctr_err_med_m', 'ctr_err_mean_m', 'iou_gt0', 'iou_ge025', 'iou_ge05', 'iou_mean', 'ori_med', 'ori_mean', 'size_err_med_cm']
df[cols].to_csv(OUT + '/task1_pseudo_quality_mav6d_train.csv', index=False, float_format='%.4g')
json.dump(dict(notes=NOTE, n_common=len(common), common_def='frames present in every non-"all frames" table'), open(OUT + '/task1_sources.json', 'w'), indent=1, ensure_ascii=False)
print('common frames', len(common))
