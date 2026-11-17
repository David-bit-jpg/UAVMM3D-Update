# -*- coding: utf-8 -*-
"""[11-17] CPU check of the upper-bound view augmentation: (1) contact sheet of 20 augmented training samples with the GT 3-D box projected
with the AUGMENTED K onto the augmented input image; (2) distribution (over N samples) of apparent size in the 512-px input and
t = log(Zv/diag), Zv = Z*512/f, vs the plain 512x288 npy cache (= previous upper-bound training input).
usage: python check_ub_zoom.py <cfg yaml> <out png> [N]"""
import os, sys, json
sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict
from uavdet3d.config import cfg, cfg_from_yaml_file
import ub_zoom_dataset as UZ

os.chdir('E:/Open3DUAVDet/tools')
cfg_from_yaml_file(sys.argv[1], cfg)
out = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
ds = UZ.build(cfg.DATA_CONFIG, training=True)
ds.data_pre_processor = lambda d: d                 # raw sample: image (1,3,H,W) normalised, gt_box9d, intrinsic (augmented)
P8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(b, K):
    c = (P8 * b[3:6]) @ R.from_euler(ds.euler_seq, b[6:9]).as_matrix().T + b[:3]
    p = (K @ c.T).T
    return p[:, :2] / p[:, 2:3]


def stats(b, K):
    uv = corners(b, K); f = float(np.sqrt(K[0, 0] * K[1, 1]))
    return float(uv[:, 0].max() - uv[:, 0].min()), float(np.log(b[2] * 512.0 / f / np.linalg.norm(b[3:6])))


rng = np.random.RandomState(0)
np.random.seed(0)
tiles = []
for n in range(20):
    d = ds[int(rng.randint(len(ds)))]
    im = d['image'][0].transpose(1, 2, 0) * ds.norm_std + ds.norm_mean
    im = np.clip(im * 255, 0, 255).astype(np.uint8)[..., ::-1].copy()
    K = d['intrinsic'][0]
    for b in d['gt_box9d'].astype(np.float64):
        uv = corners(b, K)
        for a, c in E:
            cv2.line(im, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[c]).astype(int)), (0, 255, 0), 1, cv2.LINE_AA)
        w, t = stats(b, K)
        cu = K @ b[:3]; cu = cu[:2] / cu[2]
        x0, y0 = int(max(0, cu[0] - 40)), int(max(0, cu[1] - 40))
        crop = cv2.resize(im[y0:y0 + 80, x0:x0 + 80], (160, 160), interpolation=cv2.INTER_NEAREST)
        cv2.putText(im, 'w=%.1fpx t=%.2f Z=%.1fm' % (w, t, b[2]), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        im[288 - 160:288, 512 - 160:512] = cv2.copyMakeBorder(crop, 0, 160 - crop.shape[0], 0, 160 - crop.shape[1], cv2.BORDER_CONSTANT)
    tiles.append(im)
rows = [np.concatenate(tiles[r * 4:(r + 1) * 4], 1) for r in range(5)]
cv2.imwrite(out, np.concatenate(rows, 0))
# distributions
W, T = [], []
for n in range(N):
    d = ds[int(rng.randint(len(ds)))]
    for b in d['gt_box9d'].astype(np.float64):
        w, t = stats(b, d['intrinsic'][0]); W.append(w); T.append(t)
W0, T0 = [], []
for i in ds.valid_idx:
    m = ds.metas[int(i)]; K = np.array(m['K_raw']) ; K = UZ.cg.scale_K(K, 512 / 1920., 288 / 1080.)
    for b in np.asarray(m['boxes9d'], np.float64).reshape(-1, 9):
        w, t = stats(b, K); W0.append(w); T0.append(t)
q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
res = {'n_aug_boxes': len(W), 'aug_width_px_pcts': dict(zip(q, np.percentile(W, q).round(2).tolist())),
       'aug_t_pcts': dict(zip(q, np.percentile(T, q).round(3).tolist())),
       'aug_frac_width_lt_9px': float(np.mean(np.array(W) < 9)), 'aug_frac_width_lt_6px': float(np.mean(np.array(W) < 6)),
       'plain_width_px_pcts': dict(zip(q, np.percentile(W0, q).round(2).tolist())), 'plain_t_pcts': dict(zip(q, np.percentile(T0, q).round(3).tolist()))}
print(json.dumps(res, indent=1))
json.dump(res, open(out.replace('.png', '.json'), 'w'), indent=1)
