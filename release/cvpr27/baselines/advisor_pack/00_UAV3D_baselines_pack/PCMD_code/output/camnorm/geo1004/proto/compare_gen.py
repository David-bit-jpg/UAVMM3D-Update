# -*- coding: utf-8 -*-
"""11-05 通用对比：CMP_SPEC='组名:中文名,...'，每组 × 测距强（部署红外剪影）/弱（去红外剪影，后缀 wq）× val/test。
组名 v2rref = 5 种子部署版（默认读法）。输出 compare_gen_<CMP_OUT>.json。只用 CPU。"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/ablation'); sys.path.insert(0, G + '/cmpeval')
import ablation as AB
import paper_table as PT
ARMS = [tuple(a.split(':', 1)) for a in os.environ['CMP_SPEC'].split(',')]
out = []
for sp in ('val', 'test'):
    M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
    idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    dist = np.linalg.norm(M['gt'][:, :3], axis=1); valid = M['n_gt'] > 0
    for strength, sfx in (('强', ''), ('弱', 'wq')):
        for n, name in ARMS:
            fac = n + sfx
            if not os.path.exists('%s/finalads/unc_%s_%s.pkl' % (G, sp, fac)):
                print('缺', sp, fac); continue
            for k in ('UAV_ENSDIR', 'UAV_POSTAG', 'UAV_SHAPESFX'):
                os.environ.pop(k, None)
            if n != 'v2rref':
                os.environ.update({'UAV_ENSDIR': n + '_ens', 'UAV_POSTAG': n, 'UAV_SHAPESFX': '_' + n})
            recs, T, F = AB.build(sp, 'kzt7q', fac, 'kps')
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
            pe = float(np.nanmedian(np.linalg.norm(P['pt'] - M['gt'][:, :3], axis=1)[valid]))
            out.append(dict(split=sp, strength=strength, arm=n, name=name, metrics=row, pos_median=pe))
            a = row['all']
            print('%s 测距%s %-18s UDS %.2f AP %.2f ATE %.3f ASE %.2f AOE %.2f | 位置中位 %.3f m | AP 近/中/远 %.1f/%.1f/%.1f' % (
                sp, strength, name, a['UDS'], a['AP'], a['ATE'], a['ASE'], a['AOE'], pe, *[row[b[0]]['AP'] for b in PT.BINS]), flush=True)
json.dump(out, open(G + '/proto/compare_gen_%s.json' % os.environ.get('CMP_OUT', 'x'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
