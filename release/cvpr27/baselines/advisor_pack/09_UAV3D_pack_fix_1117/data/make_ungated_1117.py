# -*- coding: utf-8 -*-
"""11-17: AntiUAV3D (RealDrone3D) PST tables WITHOUT the agreement-check frame filter, i.e. the same rule as the MAV6D table
(every frame with a pseudo box gets the teacher pseudo orientation). Logic = make_tables_1115b.out_table minus the gate test.
No real labels are read (labels.pkl only gives seq/frame keys)."""
import pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, 'D:/UAV3D_pack_fix_1115b/data')
import make_tables_1115b as M
OUT_LAB = M.OUT_LAB; RZ = M.RZ

def out_table(tag, orient, gate):
    pos = M.load_pos('D:/mmcache/outdoor_real_fs/train/index_pseudo_%s.pkl' % tag)
    T = np.load(orient, allow_pickle=True)
    lab = {(l['seq'], l['frame']) for l in pickle.load(open(OUT_LAB, 'rb'))}
    out = []
    for i in range(len(T['seq'])):
        k = (str(T['seq'][i]), str(T['frame'][i]))
        if k not in pos or k not in lab or (gate and not T['gate'][i]): continue
        b = pos[k]; Rd = Rot.from_euler('xyz', b[6:9]).as_matrix(); R0 = np.asarray(T['R'][i], np.float64)
        q = int(np.argmin([Rot.from_matrix(Rd.T @ R0 @ RZ[j]).magnitude() for j in range(4)]))
        Rm = R0 @ RZ[q]; lwh = b[3:6].copy()
        if q % 2: lwh[[0, 1]] = lwh[[1, 0]]
        out.append(dict(seq=k[0], frame=k[1], center=b[:3].copy(), lwh=lwh, euler_xyz=Rot.from_matrix(Rm).as_euler('xyz')))
    return out, len(pos)

old = pickle.load(open('D:/UAV3D_pack_update_1112/data/cmpfull/pcmd_teacher_pseudo_outdoor_train.pkl', 'rb'))
for name, tag, orient in [('outdoor_train_ungated', 'srcx_out2antifs', 'D:/S2MUAV3D/srcx_work/out2antifs/pseudo_tr1.npz'),
                          ('outdoor_train_rgbonly_ungated', 'srcx_out2antifs_none', 'D:/S2MUAV3D/srcx_work/out2antifs_none/pseudo_tr1.npz'),
                          ('outdoor_train_nosensor_ungated', 'srcx_out2antifs_none_sfE1s0', 'D:/S2MUAV3D/srcx_work/sf_E1_out2antifs_none_s0/pseudo_tr1.npz')]:
    g, _ = out_table(tag, orient, True); u, npos = out_table(tag, orient, False)
    print('%-32s pseudo-box frames %d | gated %d | ungated %d' % (name, npos, len(g), len(u)))
    if name == 'outdoor_train_ungated': print('   gated replica vs shipped 249 table:', M.same(g, old))
    if len(sys.argv) > 1: pickle.dump(u, open('pcmd_teacher_pseudo_%s.pkl' % name, 'wb'))
