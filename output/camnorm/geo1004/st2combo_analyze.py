# -*- coding: utf-8 -*-
"""PREREG_st2combo.md（sha256 2a30bf58）判定 → ST2COMBO_RESULTS.md / st2combo/analysis.json（由 v2_analyze.py 改：第一段 = ST2 检测器，只评真实）。"""
import bisect
import json
import pickle
import sys

import numpy as np

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
O = G + '/st2combo'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
import softkd_analyze as ska  # noqa: E402
import confirm_analyze as ca  # noqa: E402
import twohead_fit as tf  # noqa: E402
from anchor_eval import fnum, deploy  # noqa: E402

S5 = range(5)
GUARD = float(np.log(1.4))
ARMS = ('UH', 'V2Ap', 'V2Bp', 'V2Cp')
tf.SRC = {'test': G + '/st2_recs/test_s%d.pkl', 'val': G + '/st2_recs/val_s%d.pkl'}
NAME = {'test': '真实测试', 'val': '真实验证', 'sim': '仿真全量测试'}
LABEL = {'P0': 'ST2 检测器单独', 'UH': '第二段：剪影硬标签（参照）', 'V2A': 'A 从头外观增广', 'V2B': 'B = A + 两种蒸馏', 'V2C': 'C = B + 剪影几何支路', 'V2Ap': 'A+（再真实适应）', 'V2Bp': 'B+（再真实适应）', 'V2Cp': 'C+（主方案）'}


def causal_global(recs, pr, c0, warm=50):
    """⑤：按帧时间戳跨飞行片段排序；c = 更早的全部检测中通过门限的 δ̂ 中位；前 warm 个用 c0"""
    order = sorted(range(len(recs)), key=lambda i: fnum(recs[i]['frame_id']))
    cs = [np.full(len(r['pred']), c0) for r in recs]
    acc = []
    n = 0
    for i in order:
        if n >= warm and acc:
            m = len(acc)
            cs[i][:] = acc[m // 2] if m % 2 else 0.5 * (acc[m // 2 - 1] + acc[m // 2])
        for v in pr[i][0]:
            if np.isfinite(v) and abs(v) <= GUARD:
                bisect.insort(acc, float(v))
        n += len(recs[i]['pred'])
    return cs


def main():
    R = {}
    for k in S5:
        F = {sp: pickle.load(open('%s/feats_%s_s%d.pkl' % (O, sp, k), 'rb')) for sp in ('test', 'val')}
        RC = {sp: pickle.load(open(tf.SRC[sp] % k, 'rb'))['records'] for sp in ('test', 'val')}
        for sp in ('test', 'val'):
            recs = RC[sp]
            r = R.setdefault(sp, {})
            other = {'test': 'val', 'val': 'test'}.get(sp)
            rows = {'P0': ska.top1(recs)}
            p0 = ska.summ(recs)
            r.setdefault('P0_pos', []).append(p0[1])
            r.setdefault('P0_det', []).append(p0[0])
            for a in ARMS:
                pr = ca.pr_from_feats(F[sp], recs, a)
                if sp == 'sim':
                    rows[a] = ska.top1(deploy(recs, pr, 0.0))
                    continue
                c_other = ska.cval(ca.pr_from_feats(F[other], RC[other], a))
                c_same = ska.cval(pr)
                rows[a] = ska.top1(deploy(recs, pr, c_other))
                for nm, c in (('A0', c_other), ('A1', c_same), ('C5', causal_global(recs, pr, c_other))):
                    s = ska.summ(deploy(recs, pr, c))
                    r.setdefault('%s_%s_pos' % (nm, a), []).append(s[1])
                    r.setdefault('%s_%s_det' % (nm, a), []).append(s[0])
            keys = sorted(set.intersection(*[set(v) for v in rows.values()]))
            for a in rows:
                r.setdefault(a + '_deb', []).append(ska.debiased(rows[a], keys))
            ty = np.array([q[0].split('/')[0] for q in keys])
            if sp != 'sim':
                for t in ('mavic2', 'phantom4'):
                    for a in ARMS:
                        r.setdefault('%s_deb_%s' % (a, t), []).append(ska.debiased(rows[a], keys, ty == t))
            if sp == 'sim':
                # 深度修正误差：|δ̂ − log(真值深度 / P0 深度)| 中位
                truth = {}
                for ri, rr in enumerate(recs):
                    gt, pred, conf = rr['gt'], rr['pred'], rr['conf']
                    if len(gt) == 0 or len(pred) == 0:
                        continue
                    used = np.zeros(len(gt), bool)
                    for jj in np.argsort(-conf):
                        dd = np.linalg.norm(gt[:, :3] - pred[jj, :3], axis=1)
                        dd[used] = np.inf
                        g = int(np.argmin(dd))
                        if dd[g] < 1.0:
                            used[g] = True
                            truth[(ri, int(jj))] = np.log(gt[g, 2] / pred[jj, 2])
                yt = np.array([truth.get((int(ri), int(pi)), np.nan) for ri, pi in F[sp]['items']])
                ok = np.isfinite(yt)
                for a in ARMS:
                    r.setdefault(a + '_derr', []).append(100 * float(np.median(np.abs(F[sp][a]['d'][ok] - yt[ok]))))
        print('种子 %d：' % k + '；'.join('%s 逐帧 ST2 %.1f C+ %.1f' % (sp, R[sp]['P0_deb'][-1], R[sp]['V2Cp_deb'][-1]) for sp in ('test', 'val')), flush=True)
    json.dump(R, open(O + '/analysis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    report(R)


def report(R):
    A = lambda v: np.array(v)
    L = ['# ST2 自训练检测器 + 红外蒸馏第二段（PREREG_st2combo.md，sha256 2a30bf58）', '',
         '第一段 = release_20260924/st2_s{k} 单模型；第二段 = PREREG_v2 的网络（种子 j = 2 + k mod 3，不重训）。', '']
    res = {}
    for sp in ('test', 'val'):
        r = R[sp]
        L += ['## ' + NAME[sp], '']
        if sp == 'sim':
            L += ['| 模型 | 扣全局倍数逐帧位置 cm | 深度修正误差 % |', '|---|---|---|', '| 原模型 | %.1f | — |' % A(r['P0_deb']).mean()]
            for a in ARMS:
                L.append('| %s | %.1f | %.2f |' % (LABEL[a], A(r[a + '_deb']).mean(), A(r[a + '_derr']).mean()))
            L.append('')
            continue
        L += ['| 模型 | 扣全局倍数逐帧位置 cm | 部署：因果全局滚动中位 | 部署：另一集合中位 | 部署：本集合中位 | 检出 %（因果） |', '|---|---|---|---|---|---|',
              '| 原模型 | %.1f | %.1f | — | — | %.1f |' % (A(r['P0_deb']).mean(), A(r['P0_pos']).mean(), A(r['P0_det']).mean())]
        for a in ARMS:
            L.append('| %s | %.1f | %.1f | %.1f | %.1f | %.1f |' % (LABEL[a], A(r[a + '_deb']).mean(), A(r['C5_%s_pos' % a]).mean(),
                                                               A(r['A0_%s_pos' % a]).mean(), A(r['A1_%s_pos' % a]).mean(), A(r['C5_%s_det' % a]).mean()))
        L += ['', '| 比较（扣全局倍数逐帧位置，cm） | 平均 | 各种子 | 变好种子 |', '|---|---|---|---|']
        cmp = {}
        for nm, x, y in (('ST2 + C+ − ST2', 'V2Cp', 'P0'), ('ST2 + B+ − ST2', 'V2Bp', 'P0'), ('ST2 + A+ − ST2', 'V2Ap', 'P0'), ('ST2 + UH − ST2', 'UH', 'P0')):
            d = A(r[x + '_deb']) - A(r[y + '_deb'])
            cmp[nm] = d
            L.append('| %s | %+.2f | %s | %d/5 |' % (nm, d.mean(), ' '.join('%+.2f' % v for v in d), (d < 0).sum()))
        for t in ('mavic2', 'phantom4'):
            for x in ('V2Cp',):
                d = A(r['%s_deb_%s' % (x, t)]) - A(r['UH_deb_' + t])
                L.append('| （按机型 %s）%s − UH | %+.2f | %s | %d/5 |' % (t, LABEL[x], d.mean(), ' '.join('%+.2f' % v for v in d), (d < 0).sum()))
        dep = A(r['C5_V2Cp_pos']) - A(r['P0_pos'])
        ddet = A(r['C5_V2Cp_det']) - A(r['P0_det'])
        L += ['', '部署（因果全局滚动中位）主方案 − 原模型：%+.2f cm（%s；%d/5），检出 %+.2f 点' % (
            dep.mean(), ' '.join('%+.2f' % v for v in dep), (dep < 0).sum(), ddet.mean()), '']
        ok = lambda d, thr: bool(d.mean() <= thr and (d < 0).sum() >= 4)
        res[sp] = dict(main=bool(ok(dep, -1.5) and ddet.mean() >= -1.0), perframe=ok(cmp['ST2 + C+ − ST2'], -1.0))
    L += ['## 判定（两个真实集合都要成立）', '']
    for key, nm in (('main', '主判据：ST2 + C+（因果全局滚动中位部署）比 ST2 单独好 ≥ 1.5 cm（≥ 4/5）'), ('perframe', '次判据：逐帧（扣全局倍数）ST2 + C+ 比 ST2 好 ≥ 1 cm')):
        L.append('**%s：%s**（测试 %s / 验证 %s）' % (nm, '成立' if res['test'][key] and res['val'][key] else '不成立',
                                               '✓' if res['test'][key] else '✗', '✓' if res['val'][key] else '✗'))
    open(G + '/ST2COMBO_RESULTS.md', 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
