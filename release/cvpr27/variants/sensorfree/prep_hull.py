# -*- coding: utf-8 -*-
"""[review1113 sensorfree] box-hull matte for the ORIENTATION STUDENTS' cross-domain composites (CPU only, no labels of any real split).
usage: python prep_hull.py <src in|out> [N_check]        (idempotent; writes only D:/S2MUAV3D/srcx_work/sensorfree/hull_<src>.npy + .json)

The students (srcx/lib/ou_train_x.py, KP.CacheDS with xbg MATTE 'v') composite a simulated source crop over a real background crop with the
alpha a_v.npy of the source crop cache sx.S['sim_crops'] (= kzpaste_train.matte(..., 'v'): IR silhouette cross-checked by LiDAR).
Here a_v is replaced by the FILLED CONVEX HULL OF THE PROJECTED 3-D GT BOX(ES) of the same crop (crop intrinsics K and boxes9d of the cache
meta; corner convention = refiner_x.proj: centre + R_xyz(euler) * (P8 * lwh)), 255 inside, 0 outside, uint8, same (M, 288, 512) layout.
Every box of the crop is filled (a_v also covers every drone of the crop). The composite gate (meta has_matte, a.max() > 0) is NOT changed,
so the composite rate stays that of PST. Check printed: fraction of a_v mass inside the hull (should be ~1 if the convention is right)."""
import os, sys, json, pickle
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
CACHE = {'in': 'C:/Users/jiang/AppData/Local/Temp/claude/0fa2f3b7e932577a/26739ba4-3497-41f0-8f86-fdad1ac856ee/scratchpad/kz/crops_6000',
         'out': 'D:/S2MUAV3D/pcmd_out_work/crops_5367'}
P8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
OUTD = 'D:/S2MUAV3D/srcx_work/sensorfree'


def hull_of(K, boxes, H, W):
    h = np.zeros((H, W), np.uint8)
    for b in np.asarray(boxes, np.float64).reshape(-1, 9):
        c = (P8 * b[3:6]) @ Rot.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
        q = (np.asarray(K, np.float64) @ c.T).T
        if (q[:, 2] <= 1e-3).any():
            continue
        uv = q[:, :2] / q[:, 2:3]
        cv2.fillConvexPoly(h, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
    return h


if __name__ == '__main__':
    src = sys.argv[1]; nchk = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    C = CACHE[src]; meta = pickle.load(open(C + '/meta.pkl', 'rb')); av = np.load(C + '/a_v.npy', mmap_mode='r')
    M, H, W = av.shape; assert M == len(meta), (M, len(meta))
    os.makedirs(OUTD, exist_ok=True); out = '%s/hull_%s.npy' % (OUTD, src)
    idx = np.arange(M) if not nchk else np.random.default_rng(0).choice(M, nchk, replace=False)
    if not nchk and os.path.exists(out) and os.path.exists(out[:-4] + '.json'):
        print('exists', out, open(out[:-4] + '.json').read()); sys.exit(0)
    A = np.lib.format.open_memmap(out + '.tmp.npy', mode='w+', dtype=np.uint8, shape=(M, H, W)) if not nchk else None
    inside, iou, area_h, area_v, nm = [], [], [], [], 0
    for n in idx:
        m = meta[n]; h = hull_of(m['K'], m['boxes'], H, W)
        if A is not None:
            A[n] = h
        if m['has_matte']:
            a = np.asarray(av[n], np.float32) / 255.0
            if a.sum() > 0:
                nm += 1; hb = h > 128; inside.append(float(a[hb].sum() / a.sum()))
                ab = a > 0.5; u = (ab | hb).sum(); iou.append(float((ab & hb).sum() / max(u, 1))); area_h.append(int(hb.sum())); area_v.append(int(ab.sum()))
    info = dict(src=src, cache=C, n=int(len(idx)), n_with_matte=nm, av_mass_inside_hull_median=float(np.median(inside)), av_mass_inside_hull_p05=float(np.percentile(inside, 5)),
                iou_median=float(np.median(iou)), hull_over_av_area_median=float(np.median(np.array(area_h) / np.maximum(area_v, 1))),
                hull_empty_frac=None)
    if A is not None:
        A.flush(); info['hull_empty_frac'] = float(np.mean(np.asarray(A).reshape(M, -1).max(1) == 0)); del A
        os.replace(out + '.tmp.npy', out); json.dump(info, open(out[:-4] + '.json', 'w'), indent=1)
    print('prep_hull', json.dumps(info))
