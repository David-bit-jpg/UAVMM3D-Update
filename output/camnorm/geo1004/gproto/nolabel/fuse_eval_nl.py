# -*- coding: utf-8 -*-
"""wf23/mirror/fuse_eval.py 的无标签链副本（11-10，PREREG.md）：两种独立测距合并 + 冻结后一次性评测。
  log z = (1−w)·log z_fin + w·(log z_kp − b_k)，位置沿视线按 z/z_fin 缩放；
  w 固定取 nolabel/<arm>/w_sim.json（仿真验证集逆方差，w_sim.py；与训练目标 build_targets_nl.py 同一个数）——不在验证集上选 w；
  b_k = 颜色原型 k 上 median(log z_kp/z_fin)，只用无标签验证集图像（不读真值，与原做法相同）；
  不做置信度重排（chain_eval_m_nl：τ = None）；α = 0.5。
真值只在本脚本最后的打分处读（cmpeval/<split>_meta.npz，封存的 val/test 真值），打完不回头改任何设置。
另报 w = 0（不做部署合并）一行作参照（同一冻结链，不用于任何选择）。
输出 nolabel/results_<arm>.json + nolabel/<arm>/fe_<w>_<split>_P.pkl。用法：NL_ARM=<arm> FE_UNC=z2 FE_ORI=ouS6I2M python fuse_eval_nl.py"""
import os, sys, json, pickle
os.environ['FE_ZKP'] = '1'
os.environ.setdefault('FE_CONST', 'train'); os.environ.setdefault('FE_COLOR', 'new'); os.environ.setdefault('FE_KPS', 'strict'); os.environ.setdefault('FE_HT', 'color')
os.environ.setdefault('FE_NMEM', '4')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import chain_eval_m_nl as CE

G = CE.G; ARM = CE.POSTAG; NLA = CE.NLA
W = float(json.load(open(NLA + '/w_sim.json', encoding='utf-8'))['w'])
print('臂 %s：w = %.4f（仿真验证集逆方差）' % (ARM, W), flush=True)
# ---------- 1. 冻结整链（不读任何真值）
R = {sp: CE.build(sp) for sp in ('val', 'test')}


def classes(K, lum):
    return int(np.argmin(np.abs(np.asarray(K['col_c']) - lum))) if lum is not None else -1


recv, Kv = R['val']
lr = {}
for r in recv:
    if r.get('z_kp') and len(r['pred']):
        j = int(np.argmax(r['conf'])); zf = float(np.asarray(r['pred']).reshape(-1, 9)[j][2])
        if zf > 0 and r['z_kp'] > 0:
            lr.setdefault(classes(Kv, r['lum']), []).append(np.log(r['z_kp'] / zf))
B = {k: float(np.median(v)) for k, v in lr.items()}
print('偏差 b_k（无标签验证集图像，按颜色原型）：', {k: round(v, 4) for k, v in B.items()}, '帧数', {k: len(v) for k, v in lr.items()}, flush=True)


def predict(sp, w):
    recs, K = R[sp]
    out, nf = {}, 0
    for r in recs:
        if not len(r['pred']):
            continue
        j = int(np.argmax(r['conf']))
        q = np.asarray(r['pred'], np.float64).reshape(-1, 9)[j].copy()
        k = classes(K, r['lum'])
        if w > 0 and r.get('z_kp') and r['z_kp'] > 0 and q[2] > 0 and k in B:
            lz = (1 - w) * np.log(q[2]) + w * (np.log(r['z_kp']) - B[k])
            q[:3] = q[:3] * (np.exp(lz) / q[2])
            nf += 1
        out[(str(r['seq_id']), str(r['frame_id']))] = (q, float(np.asarray(r['conf']).reshape(-1)[j]))
    return out, nf


PRED = {(sp, w): predict(sp, w) for sp in ('val', 'test') for w in (W, 0.0)}
consts = {sp: {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in R[sp][1].items()} for sp in R}
# ---------- 2. 一次性打分（只此处读封存的 val/test 真值）
res = {'arm': ARM, 'w': W, 'bias_val_images': B, 'alpha': CE.T.ALPHA, 'rerank': None, 'consts': consts, 'cfg': CE.CFG, 'nmem': CE.MEMS()}
for (sp, w), (pd, nf) in PRED.items():
    M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
    idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    valid = np.where(M['n_gt'] > 0)[0]
    N = len(M['seq'])
    P = {'pt': np.full((N, 3), np.nan), 'plwh': np.full((N, 3), np.nan), 'peul': np.full((N, 3), np.nan), 'conf': np.full(N, np.nan)}
    for k, (q, c) in pd.items():
        i = idx.get(k)
        if i is not None:
            P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = q[:3], q[3:6], q[6:9], c
    m = CE.PT.metrics(P, M, valid); am = CE.ang_metrics(P, M, valid)
    pe = float(np.nanmedian(np.linalg.norm(P['pt'] - M['gt'][:, :3], axis=1)[valid]))
    row = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], ASE=m['ASE'], AOE=m['AOE'], yaw_med=float(np.nanmedian(am['hs'])), tilt_med=float(np.nanmedian(am['tilt'])),
               pos_med=pe, mDR=m['mDR'], n_fused=nf)
    d = np.linalg.norm(M['gt'][:, :3], axis=1)
    for nm, lo, hi in CE.PT.BINS:
        sel = valid[(d[valid] >= lo) & (d[valid] < hi)]
        mb = CE.PT.metrics(P, M, sel)
        row['AP_' + nm] = mb['AP']; row['UDS_' + nm] = mb['UDS']; row['ATE_' + nm] = mb['ATE']
    key = 'deploy_fused' if w > 0 else 'no_fusion_w0'
    res.setdefault(key, {})[sp] = row
    pickle.dump(dict(P=P, valid=valid), open('%s/fe_%s_%s_P.pkl' % (NLA, key, sp), 'wb'))
    print('%s %s w=%.3f UDS %.2f | AP %.2f | ATE %.3f m（中位 %.3f）| ASE %.2f cm | AOE %.2f°（航向中位 %.2f）| 检出 %.1f%% | %s' % (
        ARM, sp, w, row['UDS'], row['AP'], row['ATE'], pe, row['ASE'], row['AOE'], row['yaw_med'], row['mDR'],
        ' '.join('%s AP %.2f UDS %.2f' % (nm, row['AP_' + nm], row['UDS_' + nm]) for nm, _, _ in CE.PT.BINS)), flush=True)
json.dump(res, open('%s/results_%s.json' % (os.path.dirname(NLA), ARM), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
