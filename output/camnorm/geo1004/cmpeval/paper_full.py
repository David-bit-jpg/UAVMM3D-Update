# -*- coding: utf-8 -*-
"""把完整方法（ablation.py 的 6 行同一管线逐项开关）的逐帧结果对齐到对比表的 4800 帧，存成 pf/FULL<code>_s0_test_adabn.npz，
供 paper_table.py 出表。只用 CPU。每帧取置信度最高的那个预测；该帧没检出 → 全 NaN（算漏检）。python paper_full.py [test|val]"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
sys.path.insert(0, '@@ROOT@@/output/camnorm/geo1004/ablation')
import ablation as AB
C = '@@ROOT@@/output/camnorm/geo1004/cmpeval'
sp = sys.argv[1] if len(sys.argv) > 1 else 'test'
CODES = {'①': 'FULL1', '②′': 'FULL2', '③': 'FULL3', '④': 'FULL4'}
M = np.load('%s/%s_meta.npz' % (C, sp))
idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
os.makedirs(C + '/pf', exist_ok=True)
for code, name, orient, fac, height in AB.ROWS:
    if code not in CODES:
        continue
    recs, T, F = AB.build(sp, orient, fac, height)
    N = len(M['seq'])
    PF = {'pt': np.full((N, 3), np.nan), 'plwh': np.full((N, 3), np.nan), 'peul': np.full((N, 3), np.nan), 'conf': np.full(N, np.nan)}
    hit = 0
    for r in recs:
        i = idx.get((str(r['seq_id']), str(r['frame_id'])))
        if i is None or not len(r['pred']):
            continue
        j = int(np.argmax(r['conf'])) if len(r['conf']) else 0
        q = np.asarray(r['pred'], np.float64).reshape(-1, 9)[j]
        PF['pt'][i], PF['plwh'][i], PF['peul'][i] = q[:3], q[3:6], q[6:9]
        PF['conf'][i] = float(np.asarray(r['conf']).reshape(-1)[j]) if len(r['conf']) else 1.0
        hit += 1
    # 真值核对：同一帧的真值中心应一致（只核对，不进方法）
    gi = [idx[(str(r['seq_id']), str(r['frame_id']))] for r in recs if (str(r['seq_id']), str(r['frame_id'])) in idx and len(r['gt'])]
    gr = [np.asarray(r['gt'], np.float64).reshape(-1, 9)[0, :3] for r in recs if (str(r['seq_id']), str(r['frame_id'])) in idx and len(r['gt'])]
    dmax = float(np.max(np.linalg.norm(M['gt'][gi, :3] - np.array(gr), axis=1))) if gi else -1
    np.savez('%s/pf/%s_s0_%s_adabn.npz' % (C, CODES[code], sp), **PF)
    print('%s %s：对齐 %d/%d 帧有预测（管线共 %d 帧）真值中心最大差 %.4f m' % (code, name, hit, N, len(recs), dmax), flush=True)
