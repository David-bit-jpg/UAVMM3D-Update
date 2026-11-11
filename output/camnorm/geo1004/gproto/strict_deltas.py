# -*- coding: utf-8 -*-
"""全严格单帧伪标签第 2 步：在无热图时序背景的 ST2 五集成训练帧检测（st/real_train_preds_st2nb.pkl）上，
算第二段 V2Pp_s0..4（红外剪影蒸馏）逐检测修正量 d / 自估不确定度 lb / 成员分歧 sd（GPU）。输出 st2m/delta_IR_nb.pkl（train）。"""
import os, sys, pickle
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, '@@ROOT@@'); sys.path.insert(0, G); sys.path.insert(0, G + '/finalads')
import unc_dump as UD
from v2_feats import _TrainImages
d = pickle.load(open('@@ROOT@@/output/camnorm/st/real_train_preds_st2nb.pkl', 'rb'))
recs = [dict(seq_id=f['seq'], frame_id=f['frame'], pred=np.asarray(f['pred'], np.float64).reshape(-1, 9), conf=np.asarray(f['conf'], np.float64).reshape(-1), gt=np.zeros((0, 9))) for f in d['frames']]
tr = UD.deltas_full(recs, _TrainImages(), [('V2Pp', j) for j in range(5)])
pickle.dump(dict(train=tr), open(G + '/st2m/delta_IR_nb.pkl', 'wb')); print('完成', len(tr), flush=True)
