# -*- coding: utf-8 -*-
"""配对差的序列整簇 bootstrap 区间（只用 CPU；真值只用于打分）。
读 final_eval.py 存的逐帧预测 fe_<标签>_<split>_P.pkl，两臂在同一批帧上比较；每次按序列有放回抽样（val 16 段 / test 10 段），
两臂用同一组抽样帧算 cmpeval/paper_table.metrics，报 后者 − 前者 的中位与 95% 区间。
用法：python boot_ci.py <前者标签> <后者标签> [次数=500]"""
import sys, pickle, json
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
HERE = G + '/gproto/wf23/mirror'   # 副本：wf21/orient_uda/boot_ci.py，只改读写目录
sys.path.insert(0, G + '/cmpeval')
import paper_table as PT   # noqa: E402
a, b = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 500
out = {}
for sp in ('val', 'test'):
    M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
    Pa = pickle.load(open('%s/fe_%s_%s_P.pkl' % (HERE, a, sp), 'rb'))
    Pb = pickle.load(open('%s/fe_%s_%s_P.pkl' % (HERE, b, sp), 'rb'))
    valid = Pa['valid']
    assert np.array_equal(valid, Pb['valid'])
    seq = np.array([str(s) for s in M['seq']])[valid]
    us = np.unique(seq)
    groups = [valid[seq == s] for s in us]
    rng = np.random.default_rng(0)
    D = {k: [] for k in ('UDS', 'AP', 'ATE', 'ASE', 'AOE')}
    base_a, base_b = PT.metrics(Pa['P'], M, valid), PT.metrics(Pb['P'], M, valid)
    for _ in range(B):
        pick = rng.integers(0, len(us), len(us))
        sel = np.concatenate([groups[i] for i in pick])
        ma, mb = PT.metrics(Pa['P'], M, sel), PT.metrics(Pb['P'], M, sel)
        for k in D:
            D[k].append(mb[k] - ma[k])
    row = {}
    for k, v in D.items():
        v = np.array(v)
        row[k] = dict(diff=float(base_b[k] - base_a[k]), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
    out[sp] = row
    print('%s %s → %s：' % (sp, a, b) + '；'.join('%s %+.2f [%+.2f, %+.2f]' % (k, r['diff'], r['lo'], r['hi']) for k, r in row.items()), flush=True)
json.dump(out, open('%s/boot_%s__%s.json' % (HERE, a, b), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
