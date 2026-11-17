# -*- coding: utf-8 -*-
"""[UAV port] MonoDGP inference on a cmpfull split -> pkl (same order as <split>/labels.pkl).

python predict.py --ckpt <out>/checkpoint.pth --split real_test --out pred_real_test.pkl [--limit 50]
python predict.py --roundtrip --split sim_val --limit 50      # GT -> encode -> decode self-check (no model)

Output: list of dict(seq, frame, boxes9d (n,9) = center(3, geometric center, camera frame), lwh(3), euler_xyz(3)),
scores (n,)), n <= 10.
Decoding follows lib/helpers/decode_helper.py exactly (top-50 queries by class prob, score = prob * exp(-log_sigma),
depth = geometry-error-corrected projected depth, center = back-projection of the predicted projected 3D center,
yaw via alpha2ry with the 2D-box center); MonoDGP/DETR has no NMS, so we keep the 10 highest final scores
and apply no score threshold (the official tester's 0.2 threshold is left to the evaluator).
"""
import os
import sys
import argparse
import pickle
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import torch
import yaml
from scipy.spatial.transform import Rotation as Rot

from lib.datasets.uav.uav_dataset import UAV_Dataset, decode_angles, PROTO, ANG

DATA = '@@ROOT@@/data/cmpfull'
TOPK, KEEP = 50, 10


def decode_frame(prob, x2d_n, xy3d_n, depth, heading72, size3, sigma, K, img_size, mean_size, keep=KEEP):
    """All inputs for the top-k queries of one image (numpy). x2d_n = 2D-box center x, xy3d_n = projected 3D
    center, both normalised by the 960x544 canvas. Returns boxes9d (n, 9), scores (n,)."""
    W, H = float(img_size[0]), float(img_size[1])
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out, scores = [], []
    for j in range(len(prob)):
        x3d, y3d = xy3d_n[j]
        x2d = x2d_n[j] * W  # 2D-box center x, as decode_detections (alpha2ry(alpha, x))
        d = float(depth[j])
        dims = size3[j] + mean_size  # [h, w, l]
        u, v = x3d * W, y3d * H
        center = np.array([(u - cu) * d / fu, (v - cv) * d / fv, d])  # == calib.img_to_rect (no bottom-center shift)
        eul = ANG.euler_xyz_from_ypr(decode_angles(heading72[j], x2d, cu, fu))  # ypr -> our euler_xyz
        out.append(np.concatenate([center, [dims[2], dims[1], dims[0]], eul]))
        scores.append(float(prob[j]) * float(sigma[j]))
    out, scores = np.array(out).reshape(-1, 9), np.array(scores)
    o = np.argsort(-scores)[:keep]
    return out[o], scores[o]


def corners(b9):
    R = Rot.from_euler('xyz', b9[6:9]).as_matrix()
    return (PROTO * b9[3:6]) @ R.T + b9[:3]


def proj(c, K):
    uv = c @ K.T
    return uv[:, :2] / uv[:, 2:3]


def roundtrip(args):
    """encode GT with the training encoder (no augmentation), decode with decode_frame, compare corners."""
    ds = UAV_Dataset('train', {'label_files': ['%s/%s/labels.pkl' % (DATA, args.split)], 'random_mixup3d': 0,
                               'random_flip': 0, 'aug_pd': False, 'aug_crop': False})
    ds.data_augmentation = False
    e3, e2, ea, n, nskip = [], [], [], 0, 0
    for i in range(min(args.limit or 50, len(ds))):
        img, P2, T, info = ds[i]
        K = P2[:, :3].astype(np.float64)
        gts = [b for b in ds.recs[i]['boxes']]
        m = T['mask_2d']
        nskip += len(gts) - int(m.sum())
        idx = np.where(m)[0]
        heading = np.zeros((len(idx), 72), np.float32)
        for q, j in enumerate(idx):
            for k in range(3):
                heading[q, 24 * k + T['heading_bin'][j, k]] = 10.
                heading[q, 24 * k + 12 + T['heading_bin'][j, k]] = T['heading_res'][j, k]
        bx = T['boxes_3d'][idx]
        b9, _ = decode_frame(np.ones(len(idx)) - 1e-3 * np.arange(len(idx)), bx[:, 0] + (bx[:, 3] - bx[:, 2]) / 2, bx[:, :2], T['depth'][idx, 0],
                             heading, T['size_3d'][idx], np.ones(len(idx)), K, info['img_size'], ds.cls_mean_size[0], keep=50)
        for q, j in enumerate(idx):
            g = gts[j]
            g9 = np.concatenate([g['center'], g['lwh'], g['euler_xyz']])
            cg, cp = corners(g9), corners(b9[q])
            e3.append(np.abs(cg - cp).max())
            e2.append(np.abs(proj(cg, K) - proj(cp, K)).max())
            Rg, Rp = Rot.from_euler('xyz', g9[6:9]), Rot.from_euler('xyz', b9[q, 6:9])
            ea.append(np.degrees((Rg.inv() * Rp).magnitude()))
            n += 1
    print('roundtrip on %d GT boxes (%d filtered by the KITTI-equivalent filters): max corner err %.2e m, '
          'max corner reprojection err %.2e px, max rotation err %.2e deg' % (n, nskip, max(e3), max(e2), max(ea)))


@torch.no_grad()
def predict(args):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_FRAC', '0.2')))
    from lib.helpers.model_helper import build_model
    from lib.helpers.decode_helper import extract_dets_from_outputs
    ck_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg_path = args.config or (os.path.join(ck_dir, 'config_used.yaml') if os.path.exists(os.path.join(ck_dir, 'config_used.yaml'))
                               else os.path.join(ROOT_DIR, 'configs', 'monodgp_uav.yaml'))
    cfg = yaml.load(open(cfg_path), Loader=yaml.Loader)
    model, _ = build_model(cfg['model'])
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()

    ds = UAV_Dataset('test', cfg['dataset'], label_files=['%s/%s/labels.pkl' % (DATA, args.split)])
    if args.limit:
        ds.idx_list = ds.idx_list[:args.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    res = []
    for inputs, calibs, _, info in dl:
        inputs, calibs = inputs.to(dev), calibs.to(dev)
        img_sizes = info['img_size'].to(dev)
        outputs = model(inputs, calibs, None, img_sizes, dn_args=0)
        dets = extract_dets_from_outputs(outputs=outputs, K=ds.max_objs, topk=TOPK).cpu().numpy()
        # dets: [label, score, xs2d, ys2d, w2d, h2d, depth, heading(72), size_3d(3), xs3d, ys3d, sigma]
        for bi in range(dets.shape[0]):
            d = dets[bi]
            idx = int(info['img_id'][bi])
            K = calibs[bi, :, :3].double().cpu().numpy()
            b9, sc = decode_frame(d[:, 1], d[:, 2], d[:, 82:84], d[:, 6], d[:, 7:79], d[:, 79:82], d[:, 84],
                                  K, info['img_size'][bi].numpy(), ds.cls_mean_size[0])
            r = ds.recs[idx]
            res.append(dict(seq=r['seq'], frame=r['frame'], boxes9d=b9.astype(np.float32), scores=sc.astype(np.float32)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pickle.dump(res, open(args.out, 'wb'))
    print('wrote', args.out, len(res), 'frames')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt')
    ap.add_argument('--config', default=None)
    ap.add_argument('--split', required=True, choices=['real_test', 'sim_val', 'real_train', 'sim_train', 'outdoor_test', 'outdoor_train', 'outdoor_val', 'sim_outdoor_train', 'sim_outdoor_val'])
    ap.add_argument('--out')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--roundtrip', action='store_true')
    a = ap.parse_args()
    roundtrip(a) if a.roundtrip else predict(a)
