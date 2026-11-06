# -*- coding: utf-8 -*-
"""第 2 步：推理时实例原型修正（在线，只用过去帧）。第 t 帧：在 [t−W, t] 窗口内，以最终输出距离为绝对值（σd）、
实例剪影帧间尺度比为相对约束（σi，只用两端都 ≤ t 的帧对），最小二乘取 x_t → 沿视线缩放最终位置。参数沿用验证集定的 σd 5% / σi 1.2% / 质量≥0.8。
帧对来自 pairs2_<split>.pkl（裁窗只用检测器输出）。输出 UDS 对比（真值只评测）。用法 CMP_SPEC 同 compare_gen；ONLINE_W 窗口。"""
import os, sys, pickle
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/ablation'); sys.path.insert(0, G + '/cmpeval')
import ablation as AB, paper_table as PT
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
SD, SI, QM, W = float(os.environ.get('ONLINE_SD', 0.05)), float(os.environ.get('ONLINE_SI', 0.012)), float(os.environ.get('ONLINE_QM', 0.8)), int(os.environ.get('ONLINE_W', 60))
LAG = int(os.environ.get('ONLINE_LAG', 0))   # 允许延迟的帧数（0 = 严格只用过去帧）
def solve(ld, pairs, n):
    rows, cols, vals, b = [], [], [], []
    for t in range(n):
        rows.append(len(b)); cols.append(t); vals.append(1 / SD); b.append(ld[t] / SD)
    for a, c, ls, q in pairs:
        r = len(b); rows += [r, r]; cols += [a, c]; vals += [1 / SI, -1 / SI]; b.append(ls / SI)
    return lsqr(coo_matrix((vals, (rows, cols)), shape=(len(b), n)).tocsr(), np.array(b), atol=1e-10, btol=1e-10)[0]
ARMS = [tuple(a.split(':', 1)) for a in os.environ['CMP_SPEC'].split(',')]
out = []
for sp in ('val', 'test'):
    M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    valid = M['n_gt'] > 0; dist = np.linalg.norm(M['gt'][:, :3], axis=1)
    # 复原 pairs2 的帧顺序（与 inst2.py 相同：st2plus_ens 里有预测且有真值框的帧，按时间戳排序）
    seqs = {}
    for r in pickle.load(open('%s/%s/ens_%s.pkl' % (G, os.environ.get('INST_ENS', 'st2plus_ens'), sp), 'rb'))['records']:
        if len(r['pred']) and len(r['gt']): seqs.setdefault(str(r['seq_id']), []).append(str(r['frame_id']))
    S = {s: (sorted(f, key=lambda x: int(x.split('.')[0]))) for s, f in seqs.items()}
    PR = {x[0]: [p for p in x[3] if p[3] >= QM] for x in pickle.load(open('pairs2_%s%s.pkl' % (sp, os.environ.get('INST_SFX', '')), 'rb'))}
    KS_ = os.environ.get('ONLINE_KS')   # 11-06 设计分析：只用这些帧间隔（逗号分隔），缺省全用
    if KS_:
        KS_ = {int(v) for v in KS_.split(',')}; PR = {k: [p for p in v if p[1] - p[0] in KS_] for k, v in PR.items()}
    for n_, name in ARMS:
        for strength, sfx in (('强', ''), ('弱', 'wq')):
            fac = n_ + sfx
            if not os.path.exists('%s/finalads/unc_%s_%s.pkl' % (G, sp, fac)): continue
            for k in ('UAV_ENSDIR', 'UAV_POSTAG', 'UAV_SHAPESFX'): os.environ.pop(k, None)
            if n_ != 'v2rref': os.environ.update({'UAV_ENSDIR': n_ + '_ens', 'UAV_POSTAG': n_, 'UAV_SHAPESFX': '_' + n_})
            recs, T, F = AB.build(sp, os.environ.get('ONLINE_ORIENT', 'kzt7q'), fac, os.environ.get('ONLINE_HEIGHT', 'kps')); N = len(M['seq'])
            P = {'pt': np.full((N, 3), np.nan), 'plwh': np.full((N, 3), np.nan), 'peul': np.full((N, 3), np.nan), 'conf': np.full(N, np.nan)}
            for r in recs:
                i = idx.get((str(r['seq_id']), str(r['frame_id'])))
                if i is None or not len(r['pred']): continue
                j = int(np.argmax(r['conf'])); q = np.asarray(r['pred'], np.float64).reshape(-1, 9)[j]
                P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = q[:3], q[3:6], q[6:9], float(np.asarray(r['conf']).reshape(-1)[j])
            Q = {k: v.copy() for k, v in P.items()}; nfix = 0
            for s, frs in S.items():
                ii = [idx.get((s, f)) for f in frs]
                ld = np.array([np.log(np.linalg.norm(P['pt'][i])) if i is not None and np.isfinite(P['pt'][i]).all() else np.nan for i in ii])
                pr = PR.get(s, [])
                if os.environ.get('ONLINE_GTREL') == '1':   # 11-06 上限：帧对相对尺度换成真值（只作参考，读真值）
                    lg = np.array([np.log(np.linalg.norm(M['gt'][i][:3])) if i is not None and valid[i] else np.nan for i in ii])
                    pr = [(a, c, lg[a] - lg[c], q) for a, c, ls, q in pr if np.isfinite(lg[a]) and np.isfinite(lg[c])]
                if os.environ.get('ONLINE_MODE') == 'offline':
                    loc = [k for k in range(len(frs)) if np.isfinite(ld[k])]; pos = {k: j for j, k in enumerate(loc)}
                    sub = [(pos[a], pos[c], ls, q) for a, c, ls, q in pr if a in pos and c in pos]
                    if sub:
                        x = solve(ld[loc], sub, len(loc))
                        for k in loc: Q['pt'][ii[k]] = P['pt'][ii[k]] * np.exp(x[pos[k]] - ld[k]); nfix += 1
                    continue
                for t in range(len(frs)):
                    if not np.isfinite(ld[t]): continue
                    lo = max(0, t - W)
                    loc = [k for k in range(lo, min(len(frs), t + LAG + 1)) if np.isfinite(ld[k])]
                    pos = {k: j for j, k in enumerate(loc)}
                    sub = [(pos[a], pos[c], ls, q) for a, c, ls, q in pr if a in pos and c in pos]
                    if not sub: continue
                    x = solve(ld[loc], sub, len(loc))
                    Q['pt'][ii[t]] = P['pt'][ii[t]] * np.exp(x[pos[t]] - ld[t]); nfix += 1
            for tag, PP in (('原样', P), ('+在线实例原型', Q)):
                row = {'all': PT.metrics(PP, M, np.where(valid)[0])}
                for b, lo_, hi in PT.BINS: row[b] = PT.metrics(PP, M, np.where(valid & (dist >= lo_) & (dist < hi))[0])
                pe = float(np.nanmedian(np.linalg.norm(PP['pt'] - M['gt'][:, :3], axis=1)[valid])); a = row['all']
                out.append(dict(split=sp, strength=strength, arm=n_, name=name, online=tag, metrics=row, pos_median=pe))
                print('%s 测距%s %-16s %-10s UDS %.2f AP %.2f ATE %.3f | 位置中位 %.3f m | AP 近/中/远 %.1f/%.1f/%.1f%s' % (
                    sp, strength, name, tag, a['UDS'], a['AP'], a['ATE'], pe, *[row[b[0]]['AP'] for b in PT.BINS], '' if tag == '原样' else '  （修了 %d 帧）' % nfix), flush=True)
pickle.dump(out, open('online_%s.pkl' % os.environ.get('CMP_OUT', 'x'), 'wb'))
