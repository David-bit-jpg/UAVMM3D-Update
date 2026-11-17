# -*- coding: utf-8 -*-
"""11-17c: MAV6D PST table for the baselines under EXACTLY the RealDrone3D rule (make_pseudo_pcmd_out.py):
frames = pseudo box AND agreement check (2,126), and the teacher rotation replaced by the one of its 4 equivalent orientations
(multiples of 90 deg about the body vertical axis) closest to the detector rotation of that frame (only the representative changes; l/w swapped
when the representative turns by 90 / 270 deg). The shipped 5,227-frame MAV6D table never had this step: its detector-vs-teacher angle median is 110 deg.
No real labels read."""
import pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, 'D:/UAV3D_pack_fix_1115b/data')
import make_tables_1115b as M
pos = M.load_pos('E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav.pkl')
g, _ = M.mav_table('srcx_in2mav', 'D:/S2MUAV3D/srcx_work/in2mav/pseudo_tr1.npz', True)
out, ang = [], []
for r in g:
    b = pos[(r['seq'], r['frame'])]; Rd = Rot.from_euler('xyz', b[6:9]).as_matrix(); R0 = Rot.from_euler('xyz', r['euler_xyz']).as_matrix()
    q = int(np.argmin([Rot.from_matrix(Rd.T @ R0 @ M.RZ[j]).magnitude() for j in range(4)]))
    Rm = R0 @ M.RZ[q]; lwh = np.asarray(r['lwh'], float).copy()
    if q % 2: lwh[[0, 1]] = lwh[[1, 0]]
    ang.append(np.degrees(Rot.from_matrix(Rd.T @ Rm).magnitude()))
    out.append(dict(seq=r['seq'], frame=r['frame'], center=np.asarray(r['center'], float).copy(), lwh=lwh, euler_xyz=Rot.from_matrix(Rm).as_euler('xyz')))
print('MAV6D gated + representative swap: %d frames; detector-vs-teacher angle median %.1f deg (RealDrone3D table: 24.5)' % (len(out), np.median(ang)))
pickle.dump(out, open('pcmd_teacher_pseudo_real_train_clean_gated.pkl', 'wb'))
