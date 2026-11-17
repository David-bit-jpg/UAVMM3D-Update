# -*- coding: utf-8 -*-
"""A3-B step 1: sim-only height-to-width aspects (no real labels, no real LiDAR).
(a) mean h / ((l+w)/2) over all sim airframe 3D boxes of the indoor sim train cache (E:/mmcache/full_v2/train), per object and per model.
(b) per-brightness-class aspect: each sim object's crop brightness lum = mean grey (cv2 BGR2GRAY) over the airframe pixels
    (the same statistic as color_feats.color_stats), assigned to the nearest of the chain's real colour centres col_c
    (k-means-2 on unlabeled real train lum; = consts.col_c of the deployed run). The airframe pixel mask in sim comes from the
    sim render itself (sim depth channel inside the projected 3D box; variant: projected 3D-box hull, RGB + sim box only).
CPU only. Writes b_sim_aspect.json."""
import os, json, pickle, importlib.util
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
OUT = os.path.dirname(os.path.abspath(__file__))
R = 'E:/mmcache/full_v2/train/'
COL_C = np.array([62.621883392333984, 144.9248809814453])   # deployed consts.col_c (review1113/height/mav_rk/results.json), from unlabeled real train
PROTO = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
N_PER = 400
rng = np.random.default_rng(0)


def km2(v, it=50):   # == chain_eval_m_nl.km2
    c = np.quantile(v, [0.25, 0.75])
    for _ in range(it):
        a = np.abs(v[:, None] - c[None]).argmin(1); c = np.array([np.median(v[a == j]) for j in range(2)])
    return np.sort(c)


def dec(f, i, flag):
    ji = IDX[f]; o, ln = ji[i]
    return cv2.imdecode(np.asarray(BLOB[f][int(o):int(o) + int(ln)]), flag)


if __name__ == '__main__':
    d = pickle.load(open(R + 'index.pkl', 'rb')); metas = d['metas']; valid = np.asarray(d['valid_idx'], int)
    IDX = {f: np.load(R + f + '_index.npy', mmap_mode='r') for f in ('rgb_jpg', 'depth_png')}
    BLOB = {f: np.memmap(R + f + '.bin', dtype=np.uint8, mode='r') for f in ('rgb_jpg', 'depth_png')}
    # (a) aspects over all objects of valid frames
    rows = []
    for i in valid:
        m = metas[int(i)]
        for b, n in zip(np.asarray(m['boxes9d']).reshape(-1, 9), m['names']):
            if np.abs(b).sum() == 0:
                continue
            rows.append((n, int(i), b))
    names = sorted({r[0] for r in rows})
    asp = np.array([r[2][5] / ((r[2][3] + r[2][4]) / 2) for r in rows]); nm = np.array([r[0] for r in rows])
    per_model = {n: dict(n_obj=int((nm == n).sum()), aspect_mean=float(asp[nm == n].mean()), aspect_sd=float(asp[nm == n].std()),
                         lwh_median=np.median(np.array([r[2][3:6] for r in rows if r[0] == n]), 0).tolist()) for n in names}
    res = dict(n_frames_valid=int(len(valid)), n_obj=int(len(rows)), per_model=per_model,
               a_per_object_mean=float(asp.mean()), a_per_model_mean=float(np.mean([per_model[n]['aspect_mean'] for n in names])))
    # (b) brightness of sim crops
    lum = {'depth': [], 'hull': []}; lab = []; asp_b = []
    for n in names:
        cand = [k for k, r in enumerate(rows) if r[0] == n]
        pick = rng.choice(cand, min(N_PER, len(cand)), replace=False)
        for k in pick:
            _, i, b = rows[k]; m = metas[i]; K = np.asarray(m['K_in'], np.float64).reshape(3, 3)
            P = (PROTO * b[3:6]) @ Rot.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
            if (P[:, 2] <= 0.05).any():
                continue
            u = P @ K.T; u = u[:, :2] / u[:, 2:]
            u0, v0 = np.floor(u.min(0)).astype(int); u1, v1 = np.ceil(u.max(0)).astype(int)
            if not (6 <= u1 - u0) or u0 < 0 or v0 < 0 or u1 >= d['W'] or v1 >= d['H']:   # width >= 6 px (chain lower gate; upper 170 px gate dropped so large sim airframes get crops), fully inside image
                continue
            img = dec('rgb_jpg', i, cv2.IMREAD_COLOR); dep = dec('depth_png', i, cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hull = np.zeros(gray.shape, np.uint8); cv2.fillConvexPoly(hull, cv2.convexHull(u.astype(np.float32)).astype(np.int32), 1)
            rad = 0.5 * np.linalg.norm(b[3:6]) + 0.05
            box = np.zeros(gray.shape, bool); box[v0:v1 + 1, u0:u1 + 1] = True
            md = box & (hull > 0) & (dep > 0) & (np.abs(dep - b[2]) <= rad)
            if md.sum() < 10:
                continue
            lum['depth'].append(float(gray[md].mean())); lum['hull'].append(float(gray[hull > 0].mean()))
            lab.append(n); asp_b.append(float(b[5] / ((b[3] + b[4]) / 2)))
    lab = np.array(lab); asp_b = np.array(asp_b)
    res['b'] = {}
    for mk in ('depth', 'hull'):
        L = np.array(lum[mk])
        for cen_name, cen in (('real_col_c', COL_C), ('sim_km2', km2(L))):
            cls = np.abs(L[:, None] - cen[None]).argmin(1)
            per_obj = [float(asp_b[cls == j].mean()) if (cls == j).any() else None for j in range(2)]
            # per-model majority class (models assigned whole), class aspect = mean of member-model aspects
            mcls = {n: int(np.round(np.mean(cls[lab == n]))) for n in names}
            per_mod = [float(np.mean([per_model[n]['aspect_mean'] for n in names if mcls[n] == j])) if any(mcls[n] == j for n in names) else None for j in range(2)]
            res['b']['%s|%s' % (mk, cen_name)] = dict(centres=cen.tolist(), r_dark_bright_per_object=per_obj, r_dark_bright_per_model=per_mod,
                                                      n_dark_bright=[int((cls == 0).sum()), int((cls == 1).sum())], model_class=mcls,
                                                      model_frac_bright={n: float(np.mean(cls[lab == n])) for n in names},
                                                      model_lum_median={n: float(np.median(L[lab == n])) for n in names})
    res['n_crops'] = int(len(lab)); res['n_crops_per_model'] = {n: int((lab == n).sum()) for n in names}
    json.dump(res, open(OUT + '/b_sim_aspect.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
