# -*- coding: utf-8 -*-
"""wf23 方案二：两种独立测距合并（单帧）。
  距离 1 = 整链最终距离 z_fin（检测框宽 × 红外剪影监督的测距网络修正）；
  距离 2 = 朝向网络（学过 LiDAR 机身几何）的 SA-TM 融合角点 + PnP 的平移深度 z_kp。
  两者误差几乎独立（wf23 诊断：相关 −0.13~0.28）。合并：log z = (1−w)·log z_fin + w·(log z_kp − b_k)，位置沿视线按 z/z_fin 缩放。
  b_k = 颜色原型 k（≈ 机型）上 median(log z_kp/z_fin)，只用无标签验证集图像统计（与方案一同一原则：常数取自没参与训练的无标签真实图像）。
  w：在有标签验证集上从网格里选一个数（与其余少数超参数同口径，写明），测试集只报告。
配置同 chain_eval_m.py（环境变量 FE_*，并设 FE_ZKP=1）。"""
import os, sys, json
os.environ['FE_ZKP'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import chain_eval_m as CE

G = CE.G
res = {}
R = {}
for sp in ('val', 'test'):
    M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
    recs, K = CE.build(sp)
    R[sp] = (M, recs, K)


def classes(K, lum):
    return int(np.argmin(np.abs(np.asarray(K['col_c']) - lum))) if lum is not None else -1


# 偏差 b_k：无标签验证集图像（不读真值）
Mv, recv, Kv = R['val']
lr = {}
for r in recv:
    if r.get('z_kp') and len(r['pred']):
        j = int(np.argmax(r['conf'])); zf = float(np.asarray(r['pred']).reshape(-1, 9)[j][2])
        if zf > 0 and r['z_kp'] > 0:
            lr.setdefault(classes(Kv, r['lum']), []).append(np.log(r['z_kp'] / zf))
B = {k: float(np.median(v)) for k, v in lr.items()}
print('偏差 b_k（无标签验证集图像，按颜色原型）：', {k: round(v, 4) for k, v in B.items()}, '帧数', {k: len(v) for k, v in lr.items()})


def evaluate(sp, w):
    M, recs, K = R[sp]
    idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    valid = np.where(M['n_gt'] > 0)[0]
    N = len(M['seq'])
    P = {'pt': np.full((N, 3), np.nan), 'plwh': np.full((N, 3), np.nan), 'peul': np.full((N, 3), np.nan), 'conf': np.full(N, np.nan)}
    nf = 0
    for r in recs:
        i = idx.get((str(r['seq_id']), str(r['frame_id'])))
        if i is None or not len(r['pred']):
            continue
        j = int(np.argmax(r['conf']))
        q = np.asarray(r['pred'], np.float64).reshape(-1, 9)[j].copy()
        k = classes(K, r['lum'])
        if w > 0 and r.get('z_kp') and r['z_kp'] > 0 and q[2] > 0 and k in B:
            lz = (1 - w) * np.log(q[2]) + w * (np.log(r['z_kp']) - B[k])
            q[:3] = q[:3] * (np.exp(lz) / q[2])
            nf += 1
        P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = q[:3], q[3:6], q[6:9], float(np.asarray(r['conf']).reshape(-1)[j])
    m = CE.PT.metrics(P, M, valid)
    pe = float(np.nanmedian(np.linalg.norm(P['pt'] - M['gt'][:, :3], axis=1)[valid]))
    return m, pe, nf, P, valid


W = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
for sp in ('val', 'test'):
    for w in W:
        m, pe, nf, P_, valid_ = evaluate(sp, w)
        import pickle as _pk
        _pk.dump(dict(P=P_, valid=valid_), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fe_%s_w%02d_%s_P.pkl' % (os.environ.get('FE_OUT', 'x'), int(round(w * 100)), sp)), 'wb'))
        res.setdefault(sp, {})[w] = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], pos_med=pe, n_fused=nf)
        print('%s w=%.2f 综合分 %.2f | 平均精度 %.2f | 位置误差 %.3f m（中位 %.3f）| 合并帧 %d' % (sp, w, m['UDS'], m['AP'], m['ATE'], pe, nf), flush=True)
wbest = max(W, key=lambda w: res['val'][w]['UDS'])
print('验证集选出的 w = %.2f → 测试 综合分 %.2f（w=0 为 %.2f），位置误差 %.3f（w=0 为 %.3f）' % (
    wbest, res['test'][wbest]['UDS'], res['test'][0.0]['UDS'], res['test'][wbest]['ATE'], res['test'][0.0]['ATE']))
json.dump({'bias': B, 'res': {sp: {str(w): v for w, v in d.items()} for sp, d in res.items()}, 'w_val': wbest, 'cfg': CE.CFG},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fuse_%s.json' % os.environ.get('FE_OUT', 'x')), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
