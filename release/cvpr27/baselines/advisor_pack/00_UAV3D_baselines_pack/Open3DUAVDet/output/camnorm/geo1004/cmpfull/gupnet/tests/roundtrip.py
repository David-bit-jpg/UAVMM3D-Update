# -*- coding: utf-8 -*-
"""编码 -> 解码往返自检：把数据集生成的训练目标当成"完美网络输出"，走 extract/decode 的同一套解码，
与 labels.pkl 真值比 8 角点误差（米、像素）和旋转测地误差。可选 --flip 检查翻转增广的标签变换。"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from lib.datasets.uav import UAV, load_records
from lib.datasets.uav_utils import corners_of
from lib.helpers.decode_helper import decode_detections_uav

ap = argparse.ArgumentParser()
ap.add_argument('--labels', default='@@ROOT@@/data/cmpfull/sim_val/labels.pkl')
ap.add_argument('--n', type=int, default=50)
ap.add_argument('--flip', action='store_true')
ap.add_argument('--stride', type=int, default=1)
a = ap.parse_args()
cfg = dict(use_3d_center=True, random_flip=1.0 if a.flip else 0.0, random_crop=0.0, scale=0.4, shift=0.1)
recs = load_records([a.labels])
ds = UAV(None, 'train', cfg, records=recs)
if not a.flip: ds.data_augmentation = False
S = np.diag([-1., 1, 1])
cerr, perr, rerr, nobj, nmiss = [], [], [], 0, 0
for idx in range(0, a.n * a.stride, a.stride):
    _, P2, _, t, info = ds[idx]
    W = info['img_size'][0]; fw = W // 4
    calib = ds.get_calib(idx)
    if a.flip: calib.flip(info['img_size'])
    m = t['mask_2d'].astype(bool)
    dets = []
    for i in np.where(m)[0]:
        ind = t['indices'][i]; xs, ys = ind % fw, ind // fw
        hd = np.zeros(72)
        for k in range(3):
            hd[24 * k + t['heading_bin'][i, k]] = 10.0
            hd[24 * k + 12 + t['heading_bin'][i, k]] = t['heading_res'][i, k]
        dets.append(np.concatenate([[0, 1.0, xs + t['offset_2d'][i, 0], ys + t['offset_2d'][i, 1]], t['size_2d'][i], t['depth'][i],
                                    hd, t['size_3d'][i], [xs + t['offset_3d'][i, 0], ys + t['offset_3d'][i, 1]]]))
    dets = np.array(dets).reshape(1, -1, 84)
    res = decode_detections_uav(dets, {'img_id': [idx], 'bbox_downsample_ratio': [info['bbox_downsample_ratio']]},
                                [calib], ds.cls_mean_size, 0.0)[idx]
    gts = recs[idx]['boxes']
    nobj += len(gts); nmiss += len(gts) - len(res)
    for p in res:
        Rp = Rot.from_euler('xyz', p[6:9]).as_matrix()
        cp = corners_of(p[:3], p[3:6], Rp)
        best = None
        for g in gts:
            c, R = np.asarray(g['center']), np.asarray(g['R'])
            cg = corners_of(c, g['lwh'], R)
            if a.flip:
                cg = cg @ S.T; R = None  # 翻转后只比角点集合（独立镜像真值角点）
            e = np.abs(cp.mean(0) - cg.mean(0)).max()
            if best is None or e < best[0]: best = (e, cg, R)
        _, cg, R = best
        # 角点集合比较（翻转时角点顺序会变，用集合最近邻）
        d = np.linalg.norm(cp[:, None] - cg[None], axis=-1).min(1)
        cerr.append(d.max())
        K = calib.P2[:, :3].astype(np.float64)
        up = cp @ K.T; up = up[:, :2] / up[:, 2:]; ug = cg @ K.T; ug = ug[:, :2] / ug[:, 2:]
        perr.append(np.linalg.norm(up[:, None] - ug[None], axis=-1).min(1).max())
        if R is not None: rerr.append(np.degrees(Rot.from_matrix(Rp.T @ R).magnitude()))
print('frames %d objs %d decoded %d missing(过滤掉) %d' % (a.n, nobj, len(cerr), nmiss))
print('corner 3D err max %.2e m  mean %.2e m' % (np.max(cerr), np.mean(cerr)))
print('corner px err max %.2e px (960x544)' % np.max(perr))
print('rot geodesic err max %.2e deg' % (np.max(rerr) if rerr else float('nan')))
