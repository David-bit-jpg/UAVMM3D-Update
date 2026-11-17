# -*- coding: utf-8 -*-
"""[UAV port] encode -> decode round trip on ground truth (no network), N frames of a split.

For each GT box: dataset targets (heatmap center, 3D offset, depth, dims, multi-bin orientation of 3 angles)
are turned into "perfect predictions" in the network's output parametrisation, then decoded with the
official decoders (decode_location_flatten, decode_dimension, decode_euler_orientation) and converted to
boxes9d exactly as predict.py does. Reports center / lwh / rotation errors and the reprojection error of the
8 corners in the ORIGINAL image against labels.pkl. Also tests the horizontal-flip augmentation (decode a
flipped sample and mirror back) and reports how well the keypoint- and compensated-depth formulas reproduce
the GT depth when fed GT keypoints / GT Y (property of the geometry, not of the port).
"""
import os, sys, pickle, argparse
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))  # cmpfull/angles.py
from angles import euler_xyz_from_ypr
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

PROTO = np.array([[i, j, k] for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)])


def corners9(b9):
    return (PROTO * b9[3:6]) @ Rot.from_euler('xyz', b9[6:9]).as_matrix().T + b9[:3]


def project(K, X):
    uv = X @ np.asarray(K).T
    return uv[:, :2] / uv[:, 2:]


def perfect_vector_ori(orient, nb=4, na=3):
    """GT multi-bin encoding (per angle [cls4, off4]) -> network output layout [cls (na*2nb), off (na*2nb)]"""
    N = orient.shape[0]
    cls = torch.zeros(N, na * nb * 2); off = torch.zeros(N, na * nb * 2)
    for k in range(na):
        g = orient[:, k * 2 * nb:(k + 1) * 2 * nb]
        for i in range(nb):
            on = g[:, i] > 0.5
            cls[:, k * 2 * nb + 2 * i] = torch.where(on, -10., 10.)
            cls[:, k * 2 * nb + 2 * i + 1] = torch.where(on, 10., -10.)
            o = g[:, nb + i]
            off[:, k * 2 * nb + 2 * i] = torch.sin(o)
            off[:, k * 2 * nb + 2 * i + 1] = torch.cos(o)
    return torch.cat([cls, off], 1)


def decode_sample(enc, cfg, target):
    m = target.get_field('reg_mask').bool()
    pts = target.get_field('target_centers')[m].float()
    off = target.get_field('offset_3D')[m].float()
    depth = target.get_field('locations')[m][:, 2].float()
    calib = target.get_field('calib')
    pad = target.get_field('pad_size').view(1, 2).float()
    bidx = torch.zeros(len(pts), dtype=torch.long)
    loc = enc.decode_location_flatten(pts, off, depth, [calib], pad, bidx)
    dims_gt = target.get_field('dimensions')[m].float()
    dims = enc.decode_dimension(torch.zeros(len(pts)), torch.log(dims_gt / enc.dim_mean[0]))
    eul, _ = enc.decode_euler_orientation(perfect_vector_ori(target.get_field('orientations')[m].float()), loc)
    b9 = []
    for j in range(len(pts)):
        e = euler_xyz_from_ypr(eul[j].numpy().astype(np.float64))  # decoded ypr -> our euler_xyz (as predict.py)
        d = dims[j].numpy()
        b9.append(np.concatenate([loc[j].numpy(), d[[0, 2, 1]], e]))
    return np.array(b9).reshape(-1, 9), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='@@ROOT@@/data/cmpfull/sim_val/labels.pkl')
    ap.add_argument('--n', type=int, default=50)
    a = ap.parse_args()
    from config import cfg
    cfg.merge_from_file(os.path.join(HERE, 'runs', 'monocd_uav.yaml'))
    cfg.MODEL.DEVICE = 'cpu'
    from data.datasets.uav import UAVDataset
    from model.anno_encoder import Anno_Encoder
    enc = Anno_Encoder(cfg)
    recs = pickle.load(open(a.labels, 'rb'))
    idxs = np.linspace(0, len(recs) - 1, a.n).astype(int)
    ds = UAVDataset(cfg, a.labels, is_train=False, transforms=None, augment=False)
    dsf = UAVDataset(cfg, a.labels, is_train=True, transforms=None, augment=True)
    dsf.flip_prob = 1.0
    errs = {'center_m': [], 'lwh_m': [], 'rot_deg': [], 'reproj_px': [], 'flip_center_m': [], 'flip_rot_deg': [],
            'kp_depth_rel': [], 'comp_depth_rel': []}
    n_gt = n_kept = 0
    for i in idxs:
        r = recs[i]
        _, t, _ = ds[i]
        b9, m = decode_sample(enc, cfg, t)
        gts = [np.concatenate([b['center'], b['lwh'], b['euler_xyz']]) for b in r['boxes']]
        n_gt += len(gts); n_kept += len(b9)
        # match decoded boxes to GT by center (order is preserved, but be safe)
        for p in b9:
            g = min(gts, key=lambda g: np.linalg.norm(g[:3] - p[:3]))
            errs['center_m'].append(np.linalg.norm(g[:3] - p[:3]))
            errs['lwh_m'].append(np.abs(g[3:6] - p[3:6]).max())
            Rg, Rp = Rot.from_euler('xyz', g[6:]), Rot.from_euler('xyz', p[6:])
            errs['rot_deg'].append(np.degrees((Rg.inv() * Rp).magnitude()))
            # reprojection in the ORIGINAL image with the ORIGINAL K (labels.pkl); corner sets compared order-free
            cg, cp = project(r['K'], corners9(g)), project(r['K'], corners9(p))
            errs['reproj_px'].append(np.abs(np.sort(cg, 0) - np.sort(cp, 0)).max())
        # flip: decode the mirrored sample, mirror back
        _, tf, _ = dsf[i]
        b9f, _ = decode_sample(enc, cfg, tf)
        for p in b9f:
            Rm = np.diag([-1., 1., 1.])
            c = Rm @ p[:3]
            # undo the dataset flip (official ry -> pi - ry generalised): R = M R' S, S = width reflection in object frame
            R0 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]); S = R0.T @ np.diag([1., 1., -1.]) @ R0
            R = Rm @ Rot.from_euler('xyz', p[6:]).as_matrix() @ S
            g = min(gts, key=lambda g: np.linalg.norm(g[:3] - c))
            errs['flip_center_m'].append(np.linalg.norm(g[:3] - c))
            errs['flip_rot_deg'].append(np.degrees((Rot.from_euler('xyz', g[6:]).inv() * Rot.from_matrix(R)).magnitude()))
        # depth formulas with GT inputs (geometry check)
        kps = t.get_field('keypoints')[m][..., :2].float()
        pts = t.get_field('target_centers')[m].float()
        dims = t.get_field('dimensions')[m].float()
        z = t.get_field('locations')[m][:, 2].float()
        calib = t.get_field('calib')
        kd = enc.decode_depth_from_keypoints_batch(kps, dims, [calib])
        cd = enc.decode_depth_from_roof_and_bottom(t.get_field('EL')[m].float(), pts, kps, dims, [calib],
                                                   t.get_field('pad_size').view(1, 2).float())
        errs['kp_depth_rel'] += list(((kd - z[:, None]).abs() / z[:, None]).view(-1).numpy())
        errs['comp_depth_rel'] += list(((cd - z[:, None]).abs() / z[:, None]).view(-1).numpy())
    print('frames %d, GT boxes %d, encoded (kept by the dataset filters) %d' % (len(idxs), n_gt, n_kept))
    for k, v in errs.items():
        v = np.array(v)
        print('%-15s mean %.3g  median %.3g  max %.3g' % (k, v.mean(), np.median(v), v.max()))


if __name__ == '__main__':
    main()
