# -*- coding: utf-8 -*-
"""11-02 原型约束自训练三组 vs 部署版 vs 重训噪声对照，按论文表 2 指标（val + test）。只用 CPU。
其余完全相同，按论文表 2 指标比较（val + test）。参照组用同一读法（BE.factors）读部署版测距结果的副本，保证只差这一项。只用 CPU。"""
import os, shutil, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/ablation'); sys.path.insert(0, G + '/cmpeval')
import ablation as AB
import paper_table as PT
O = G + '/finalads'
for sp in ('test', 'val'):           # 参照组：部署版测距结果换个名字，走同一读法
    shutil.copy('%s/unc_%s.pkl' % (O, sp), '%s/unc_%s_v2rref.pkl' % (O, sp))
import bg_eval as BE
out = []
for sp in ('val', 'test'):
    M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
    idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    dist = np.linalg.norm(M['gt'][:, :3], axis=1); valid = M['n_gt'] > 0
    if os.environ.get('CMP3'):
        ARMS5 = [(n, l, {'UAV_ENSDIR': n + '_ens', 'UAV_POSTAG': n, 'UAV_SHAPESFX': '_' + n}) for n, l in (('dep3', '部署版（3 种子）'), ('nd3', '同代码重训（噪声对照，3 种子）'), ('orc', '上限：用真值滤掉完全不重叠的伪标签'), ('cmb', '置信度+时序+原型 去掉最可疑 36%'))]
    else:
      ARMS5 = [('v2rref', '部署版', {}), ('detnd', '同代码重训（噪声对照）', {'UAV_ENSDIR': 'detnd_ens', 'UAV_POSTAG': 'detnd', 'UAV_SHAPESFX': '_detnd'})]
    if not os.environ.get('CMP3'):
      ARMS5 += [(n, l, {'UAV_ENSDIR': n + '_ens', 'UAV_POSTAG': n, 'UAV_SHAPESFX': '_' + n}) for n, l in (('pf', '只筛伪标签'), ('pl', '只加原型loss'), ('pfl', '筛选+原型loss'), ('cpb', '共识原型库（不加仿真监督）'), ('cpbs', '共识原型库（加仿真监督）')) if n in ('pf', 'pl', 'pfl') or os.path.exists('%s/finalads/unc_test_%s.pkl' % (G, n))]
    for tag, name, env in ARMS5:
        for k in ('UAV_ENSDIR', 'UAV_POSTAG', 'UAV_SHAPESFX'):
            os.environ.pop(k, None)
        os.environ.update(env)
        recs, T, F = AB.build(sp, 'kzt7q', tag, 'kps')
        N = len(M['seq'])
        P = {'pt': np.full((N, 3), np.nan), 'plwh': np.full((N, 3), np.nan), 'peul': np.full((N, 3), np.nan), 'conf': np.full(N, np.nan)}
        for r in recs:
            i = idx.get((str(r['seq_id']), str(r['frame_id'])))
            if i is None or not len(r['pred']):
                continue
            j = int(np.argmax(r['conf']))
            q = np.asarray(r['pred'], np.float64).reshape(-1, 9)[j]
            P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = q[:3], q[3:6], q[6:9], float(np.asarray(r['conf']).reshape(-1)[j])
        row = {'all': PT.metrics(P, M, np.where(valid)[0])}
        for b, lo, hi in PT.BINS:
            row[b] = PT.metrics(P, M, np.where(valid & (dist >= lo) & (dist < hi))[0])
        pe = np.linalg.norm(P['pt'] - M['gt'][:, :3], axis=1)[valid]
        out.append((sp, name, row, float(np.nanmedian(pe))))
        a = row['all']
        print('%s %-14s AOE %.2f ATE %.3f ASE %.2f AP %.2f UDS %.2f | 位置中位 %.3f m | AP 近/中/远 %.2f/%.2f/%.2f' % (
            sp, name, a['AOE'], a['ATE'], a['ASE'], a['AP'], a['UDS'], out[-1][3], *[row[b[0]]['AP'] for b in PT.BINS]), flush=True)
import json
json.dump([{'split': s, 'arm': n, 'metrics': r, 'pos_median': m} for s, n, r, m in out], open(G + '/proto/compare_%s.json' % ('flt3' if os.environ.get('CMP3') else 'proto'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
