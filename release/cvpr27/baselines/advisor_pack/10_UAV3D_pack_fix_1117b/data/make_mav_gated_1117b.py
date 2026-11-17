# -*- coding: utf-8 -*-
"""11-17b: MAV6D PST table restricted to frames that pass the agreement check (= make_pseudo_pcmd_clean.py --gate-only on the clean chain).
Only the frame rule changes vs the shipped 5,227-frame clean table (same boxes / orientations, no 4-fold swap, as that table). No real labels read."""
import pickle, sys
sys.path.insert(0, 'D:/UAV3D_pack_fix_1115b/data')
import make_tables_1115b as M
pst = pickle.load(open('D:/UAV3D_pack_fix_1114/data/pcmd_teacher_pseudo_real_train_clean.pkl', 'rb'))
u, _ = M.mav_table('srcx_in2mav', 'D:/S2MUAV3D/srcx_work/in2mav/pseudo_tr1.npz', False)
g, _ = M.mav_table('srcx_in2mav', 'D:/S2MUAV3D/srcx_work/in2mav/pseudo_tr1.npz', True)
print('ungated replica vs shipped clean table:', M.same(u, pst))
P = {(r['seq'], r['frame']): r for r in pst}
import numpy as np
sub = all((r['seq'], r['frame']) in P and np.allclose(r['center'], P[(r['seq'], r['frame'])]['center']) and np.allclose(r['euler_xyz'], P[(r['seq'], r['frame'])]['euler_xyz']) for r in g)
print('gated table: %d frames; subset of the 5,227 table with identical boxes: %s' % (len(g), sub))
pickle.dump(g, open('pcmd_teacher_pseudo_real_train_clean_gated.pkl', 'wb'))
