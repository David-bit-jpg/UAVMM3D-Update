# -*- coding: utf-8 -*-
"""[review1113 rgbst_equal] GUPNet labels.pkl for the 'RGB self-training under equal conditions' pseudo labels on MAV6D.
Same frames and boxes as index_naive.pkl (PST step 1 without teacher correction: ensemble + flip, conf >= 0.5, size s_bar, own depth and
rotation). Conversion = cmpfull/make_pseudo_pcmd_clean.py (label dict of D:/cmpfull/real_train/labels.pkl for the frame -> image path, K;
boxes replaced), except that the rotation is the detector's own (no teacher orientation). No real label box is read (only file/K fields)."""
import os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull')
from angles import ypr_from_R  # noqa: E402
idx = pickle.load(open('F:/ourdet_baseproto_cache/mav6d_cn_obj/train/index_naive.pkl', 'rb'))
L = {(l['seq'], l['frame']): l for l in pickle.load(open('D:/cmpfull/real_train/labels.pkl', 'rb'))}
out = []; root = 'D:/cmpfull/real_train'
for m in idx['metas']:
    if m is None:
        continue
    b = np.asarray(m['boxes9d'], np.float64).reshape(9); Rm = Rot.from_euler('xyz', b[6:9]).as_matrix()
    d = dict(L[(m['seq'], m['frame'])]); d['file'] = os.path.join(root, d['file']).replace(chr(92), '/'); d['root'] = root
    d['boxes'] = [dict(center=b[:3].copy(), lwh=b[3:6].copy(), euler_xyz=b[6:9].copy(), R=Rm, cls='drone', ypr=ypr_from_R(Rm))]
    out.append(d)
o = sys.argv[1]; os.makedirs(os.path.dirname(o), exist_ok=True); pickle.dump(out, open(o, 'wb'))
print('GUPNet naive pseudo labels: %d frames -> %s' % (len(out), o))
