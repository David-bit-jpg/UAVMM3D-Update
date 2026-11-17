# -*- coding: utf-8 -*-
"""整图方法 predict.py 的输出（与 D:/cmpfull/real_test/labels.pkl 同序）→ cmpeval/pf/<NAME>_s<seed>_test_adabn.npz（按 seq/frame 对齐对比表的 4800 帧；
每帧取分数最高的框；文件名沿用 _adabn 只是为了让 paper_table 读到，整图方法不做 AdaBN）。python to_pf.py <pred.pkl> <NAME> <seed>"""
import pickle, sys
import numpy as np
C = 'E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval'
pred, name, seed = sys.argv[1], sys.argv[2], sys.argv[3]
P = pickle.load(open(pred, 'rb'))
M = np.load(C + '/test_meta.npz')
idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
N = len(M['seq'])
PF = {'pt': np.full((N, 3), np.nan), 'plwh': np.full((N, 3), np.nan), 'peul': np.full((N, 3), np.nan), 'conf': np.full(N, np.nan)}
hit = 0
for r in P:
    i = idx.get((str(r['seq']), str(r['frame'])))
    b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
    if i is None or not len(b):
        continue
    j = int(np.argmax(sc))
    PF['pt'][i], PF['plwh'][i], PF['peul'][i], PF['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    hit += 1
np.savez('%s/pf/%s_s%s_test_adabn.npz' % (C, name, seed), **PF)
print('%s s%s：%d/%d 帧有预测' % (name, seed, hit, N))
