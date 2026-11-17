# -*- coding: utf-8 -*-
"""[review1113/cleanfix B6] ORACLE ceiling (diagnostic only, not a method): the deployed run's orientation pseudo labels (srcx/in2mav
pseudo_tr1.npz) with the rotation R replaced by the MAV6D TRAIN ground-truth rotation of the nearest GT box (E:/mmcache/mav6d_cn_obj/train/
index.pkl, boxes9d euler xyz, camera frame = the crop camera frame; crops only change intrinsics). Translation t_pnp, size lwh, K_crop, gate,
weights, frame set unchanged (= kzst9 arm O recipe, geo1004/kzst9/kzst9_build.py: R -> GT, t/lwh/valid kept; nearest GT box regardless of
distance). Train labels are read ONLY here, to build this oracle arm; nothing is selected with them.
usage: python build_gt_pseudo.py <in npz> <out npz> <report json>"""
import json, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
src, out, rep = sys.argv[1:4]
P = dict(np.load(src, allow_pickle=False))
I = pickle.load(open('E:/mmcache/mav6d_cn_obj/train/index.pkl', 'rb'))
seqk = I.get('euler_seq', 'xyz')
gt = {(m['seq'], m['frame']): m for m in I['metas'] if m is not None}
RZ = [Rot.from_euler('z', 90 * k, degrees=True).as_matrix() for k in range(4)]
geo = lambda A, B: float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))
N = len(P['idx']); Rg = np.zeros((N, 3, 3)); dpos = np.zeros(N); err = np.zeros(N)
for i in range(N):
    m = gt[(str(P['seq'][i]), str(P['frame'][i]))]
    b = np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)
    j = int(np.argmin(np.linalg.norm(b[:, :3] - P['t_pos'][i], axis=1)))
    Rg[i] = Rot.from_euler(seqk, b[j, 6:9]).as_matrix()
    dpos[i] = float(np.linalg.norm(b[j, :3] - P['t_pos'][i]))
    err[i] = min(geo(P['R'][i] @ RZ[q], Rg[i]) for q in range(4))
Q = dict(P); Q['R'] = Rg; Q['eul'] = np.array([Rot.from_matrix(r).as_euler('xyz') for r in Rg])
np.savez(out, **Q)
g = P['gate'].astype(bool)
R = dict(n=int(N), n_gate=int(g.sum()), gate_det_within_1m_of_gt=int((g & (dpos < 1)).sum()),
         pseudo_label_sym_geodesic_err_deg_gate_mean=float(err[g].mean()), median=float(np.median(err[g])),
         note='replaced R for all rows; students train on gate rows (bag 0.8) exactly as the deployed students; t_pnp/lwh are the pseudo ones')
json.dump(R, open(rep, 'w'), indent=1); print('gt pseudo', R)
