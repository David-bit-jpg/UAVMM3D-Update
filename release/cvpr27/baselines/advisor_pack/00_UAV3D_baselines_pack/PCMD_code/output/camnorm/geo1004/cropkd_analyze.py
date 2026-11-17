# -*- coding: utf-8 -*-
"""PREREG_cropkd.md（sha256 8cb381dc）判定 → CROPKD_RESULTS.md / cropkd/analysis.json。特征：cropkd/feats_{split}_s{k}.pkl。"""
import bisect
import json
import pickle
import sys

import numpy as np

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
O = G + '/cropkd'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
import softkd_analyze as ska  # noqa: E402
import confirm_analyze as ca  # noqa: E402
import twohead_fit as tf  # noqa: E402
from anchor_eval import fnum, deploy  # noqa: E402

S5 = range(5)
GUARD = float(np.log(1.4))
ARMS = ('U0', 'UH', 'UHF', 'UHR', 'UHFR')
NAME = {'test': '真实测试', 'val': '真实验证', 'sim': '仿真全量测试'}
LABEL = {'P0': '原模型', 'U0': '第二段：不学剪影', 'UH': '第二段：剪影硬标签（基线）', 'UHF': '+ 对齐特征蒸馏', 'UHR': '+ 相对深度蒸馏', 'UHFR': '+ 两种蒸馏'}


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
        F = {sp: pickle.load(open('%s/feats_%s_s%d.pkl' % (O, sp, k), 'rb')) for sp in ('sim', 'test', 'val')}
        RC = {sp: pickle.load(open(tf.SRC[sp] % k, 'rb'))['records'] for sp in ('sim', 'test', 'val')}
        for sp in ('sim', 'test', 'val'):
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
                    for a in ('UH', 'UHFR'):
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
        print('种子 %d：' % k + '；'.join('%s 逐帧 UH %.1f UHFR %.1f' % (sp, R[sp]['UH_deb'][-1], R[sp]['UHFR_deb'][-1]) for sp in ('sim', 'test', 'val')), flush=True)
    json.dump(R, open(O + '/analysis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    report(R)


def report(R):
    A = lambda v: np.array(v)
    L = ['# 第二段裁窗内蒸馏（PREREG_cropkd.md，sha256 8cb381dc）', '',
         '第二段种子 3 / 4（检测种子偶数用 3、奇数用 4）。基线 = 剪影硬标签版（已训好的 UH）；新臂加对齐特征蒸馏（F）、相对深度蒸馏（R）或两者。', '']
    res = {}
    for sp in ('test', 'val', 'sim'):
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
        for x in ('UHFR', 'UHF', 'UHR'):
            d = A(r[x + '_deb']) - A(r['UH_deb'])
            cmp[x] = d
            L.append('| %s − 基线 | %+.2f | %s | %d/5 |' % (LABEL[x], d.mean(), ' '.join('%+.2f' % v for v in d), (d < 0).sum()))
        for t in ('mavic2', 'phantom4'):
            d = A(r['UHFR_deb_' + t]) - A(r['UH_deb_' + t])
            L.append('| （按机型）%s：两种蒸馏 − 基线 | %+.2f | %s | %d/5 |' % (t, d.mean(), ' '.join('%+.2f' % v for v in d), (d < 0).sum()))
        dep = A(r['C5_UHFR_pos']) - A(r['P0_pos'])
        dep_h = A(r['C5_UH_pos']) - A(r['P0_pos'])
        ddet = A(r['C5_UHFR_det']) - A(r['P0_det'])
        L += ['', '部署（因果全局滚动中位）两种蒸馏 − 原模型：%+.2f cm（%s；%d/5），检出 %+.2f 点；基线 − 原模型：%+.2f cm（%d/5）' % (
            dep.mean(), ' '.join('%+.2f' % v for v in dep), (dep < 0).sum(), ddet.mean(), dep_h.mean(), (dep_h < 0).sum()), '']
        ok = lambda d, thr: bool(d.mean() <= thr and (d < 0).sum() >= 4)
        res[sp] = dict(main=ok(cmp['UHFR'], -1.0), F=ok(cmp['UHF'], -1.0), R=ok(cmp['UHR'], -1.0),
                       deploy=bool(ok(dep, -1.5) and ddet.mean() >= -1.0), deploy_base=bool(ok(dep_h, -1.5)))
    L += ['## 判定（两个真实集合都要成立）', '']
    for key, nm in (('main', '主判据：两种蒸馏让逐帧位置比基线好 ≥ 1 cm（≥ 4/5）'), ('F', '次判据：只加对齐特征蒸馏'), ('R', '次判据：只加相对深度蒸馏'),
                    ('deploy', '次判据：两种蒸馏 + 因果全局滚动中位部署比原模型好 ≥ 1.5 cm'), ('deploy_base', '（描述）基线 + 因果全局滚动中位部署比原模型好 ≥ 1.5 cm')):
        L.append('**%s：%s**（测试 %s / 验证 %s）' % (nm, '成立' if res['test'][key] and res['val'][key] else '不成立',
                                               '✓' if res['test'][key] else '✗', '✓' if res['val'][key] else '✗'))
    open(G + '/CROPKD_RESULTS.md', 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
