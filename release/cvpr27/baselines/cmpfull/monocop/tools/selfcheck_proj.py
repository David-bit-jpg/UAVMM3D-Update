# -*- coding: utf-8 -*-
"""[UAV port] format/convention self-check of predict.py output on sim_val (first N frames).

python tools/selfcheck_proj.py --pred <pred_sim_val.pkl> [--n 50]
  1. order / seq / frame / dtype / shape vs D:/cmpfull/sim_val/labels.pkl
  2. projects the 8 corners of the top-1 prediction and of the GT box into the stored image (K of labels.pkl)
     and reports the mean corner distance (px) and the center distance (m)  -- meaningless for a smoke model, only
     shows that the numbers live in the same image / camera frame (finite, on-image, right magnitude).
The convention check proper is `predict.py --roundtrip` (GT -> training encoder -> decoder -> boxes9d, error ~0).
"""
import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot

PROTO = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])


def corners9(b):
    return (PROTO * b[3:6]) @ Rot.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]


def proj(c, K):
    uv = c @ np.asarray(K).T
    return uv[:, :2] / uv[:, 2:3]


ap = argparse.ArgumentParser()
ap.add_argument('--pred', required=True)
ap.add_argument('--n', type=int, default=50)
a = ap.parse_args()
pred = pickle.load(open(a.pred, 'rb'))
gt = pickle.load(open('D:/cmpfull/sim_val/labels.pkl', 'rb'))[:len(pred)]
assert len(pred) == len(gt)
bad = sum((p['seq'], p['frame']) != (g['seq'], g['frame']) for p, g in zip(pred, gt))
print('frames %d, order mismatches %d, boxes9d %s %s, scores %s, max boxes %d' % (
    len(pred), bad, pred[0]['boxes9d'].shape, pred[0]['boxes9d'].dtype, pred[0]['scores'].shape,
    max(len(p['scores']) for p in pred)))
dpx, dm, onimg, n = [], [], 0, 0
for p, g in zip(pred[:a.n], gt[:a.n]):
    if not len(p['scores']) or not g['boxes']:
        continue
    b = g['boxes'][0]
    cg = (PROTO * np.asarray(b['lwh'])) @ np.asarray(b['R']).T + np.asarray(b['center'])
    cp = corners9(p['boxes9d'][0].astype(np.float64))
    ug, up = proj(cg, g['K']), proj(cp, g['K'])
    dpx.append(np.linalg.norm(ug - up, axis=1).mean())
    dm.append(np.linalg.norm(p['boxes9d'][0, :3] - np.asarray(b['center'])))
    c = up.mean(0)
    onimg += int(0 <= c[0] < g['W'] and 0 <= c[1] < g['H'])
    n += 1
print('top-1 vs GT on %d frames: mean corner reprojection distance %.1f px (median %.1f), center distance mean %.2f m, '
      'pred center projected inside the image %d/%d, all finite %s' % (
          n, np.mean(dpx), np.median(dpx), np.mean(dm), onimg, n,
          all(np.isfinite(p['boxes9d']).all() and np.isfinite(p['scores']).all() for p in pred)))
