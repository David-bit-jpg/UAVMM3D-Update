# -*- coding: utf-8 -*-
"""[11-17] DIAGNOSTIC ONLY: score a model's predictions on the RealDrone3D TRAIN split against the sealed train labels (user-approved for the
upper-bound row), with exactly srcx/lib/score_x.py's score_val code (same metric code / outdoor_test rules, label list with box9d_ours + K).
usage (env SX_SRC=out SX_TGT=antifs): python score_traindiag.py <recs.pkl with recs['train']> <out json>"""
import sys, pickle, json, tempfile, os
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')
import score_x as SX
SX.sx.T['val_labels'] = 'D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/train_gt.pkl'
R = pickle.load(open(sys.argv[1], 'rb'))['recs']['train']
P = [dict(seq=r['seq'], frame=r['frame'], boxes9d=np.asarray(r['box'], np.float64).reshape(1, 9), scores=np.array([r['conf']]))
     for r in R if r['box'] is not None]
tmp = sys.argv[2] + '.preds.pkl'; pickle.dump(P, open(tmp, 'wb'))
res = SX.score_val(tmp)
json.dump(res, open(sys.argv[2], 'w'), indent=1, default=float)
for k, v in res.items():
    print('%-8s ' % k + '  '.join('%s %.3f' % (m, x) for m, x in v.items()))
