# -*- coding: utf-8 -*-
"""PREREG_confirm.md（sha256 b920fe53）判定 → CONFIRM_RESULTS.md / confirm/analysis.json。
特征来自 confirm/feats_{split}_s{k}.pkl（第二段全新种子 j = 3 / 4）。"""
import json
import pickle
import sys

import numpy as np

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
O = G + '/confirm'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
import twohead_fit as tf  # noqa: E402
import twohead_v2 as v2  # noqa: E402
import softkd_analyze as ska  # noqa: E402

S5 = range(5)
OUT_MD = G + '/CONFIRM_RESULTS.md'
NAME = {'test': '真实测试', 'val': '真实验证'}


def pr_from_feats(F, recs, arm):
    per = [(np.full(len(r['pred']), np.nan), np.full(len(r['pred']), np.nan)) for r in recs]
    for n, (ri, pi) in enumerate(F['items']):
        per[int(ri)][0][int(pi)] = F[arm]['d'][n]
        per[int(ri)][1][int(pi)] = F[arm]['lb'][n]
    return per


def main():
    D = {sp: {k: pickle.load(open('%s/feats_%s_s%d.pkl' % (O, sp, k), 'rb')) for k in S5} for sp in ('sim', 'test', 'val')}
    RC = {sp: {k: pickle.load(open(tf.SRC[sp] % k, 'rb'))['records'] for k in S5} for sp in ('sim', 'test', 'val')}
    T = {sp: {k: tf.targets(RC[sp][k], D[sp][k]['items']) for k in S5} for sp in ('sim', 'test', 'val')}
    R = {}
    for k in S5:
        ys = T['sim'][k][0]
        for a in ('UT', 'UH', 'UC'):
            cv = v2.Conv(D['sim'][k], ys, a, 'P')
            for sp in ('test', 'val'):
                y, typ, _ = T[sp][k]
                yh, e2, g = cv.fuse(D[sp][k], True)
                ok = np.isfinite(y)
                r = R.setdefault('H1', {}).setdefault(sp, {})
                if a == 'UT':
                    r.setdefault('h1_deb', []).append(100 * np.median(np.abs(y[ok] - np.median(y[ok]))))
                    r.setdefault('h1_raw', []).append(100 * np.median(np.abs(y[ok])))
                r.setdefault(a + '_deb', []).append(100 * np.median(np.abs(y[ok] - yh[ok] - np.median(y[ok] - yh[ok]))))
                r.setdefault(a + '_raw', []).append(100 * np.median(np.abs(y[ok] - yh[ok])))
                for t in ('mavic2', 'phantom4'):
                    mt = ok & (typ == t)
                    if a == 'UT':
                        r.setdefault('h1_deb_' + t, []).append(100 * np.median(np.abs(y[mt] - np.median(y[mt]))))
                    r.setdefault(a + '_deb_' + t, []).append(100 * np.median(np.abs(y[mt] - yh[mt] - np.median(y[mt] - yh[mt]))))
        for sp, other in (('test', 'val'), ('val', 'test')):
            recs = RC[sp][k]
            r = R.setdefault('H2', {}).setdefault(sp, {})
            rows = {'P0': ska.top1(recs)}
            p0 = ska.summ(recs)
            r.setdefault('P0_pos', []).append(p0[1])
            r.setdefault('P0_det', []).append(p0[0])
            for a in ('U0', 'UH', 'UC', 'UT'):
                pr = pr_from_feats(D[sp][k], recs, a)
                c = ska.cval(pr_from_feats(D[other][k], RC[other][k], a))
                dep = ska.deploy(recs, pr, c, 'w1')
                s = ska.summ(dep)
                r.setdefault(a + '_pos', []).append(s[1])
                r.setdefault(a + '_det', []).append(s[0])
                rows[a] = ska.top1(dep)
            keys = sorted(set.intersection(*[set(v) for v in rows.values()]))
            for a in rows:
                r.setdefault(a + '_deb', []).append(ska.debiased(rows[a], keys))
        print('种子 %d 完成' % k, flush=True)
    json.dump(R, open(O + '/analysis_confirm.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    report(R)


def report(R):
    A = lambda d: np.array(d)
    L = ['# 两个描述性正结果的复核（PREREG_confirm.md，sha256 b920fe53）', '',
         '第二段全新种子（j = 3 / 4，从未在真实上评测过）；检测种子 k 偶数用 j = 3、奇数用 j = 4。学生：U0 不学剪影 / UH 红外剪影硬标签 / UC 红外剪影软标签 / UT 红外教师软剪影。', '']
    res = {}
    for sp in ('test', 'val'):
        h = R['H1'][sp]
        L += ['## %s' % NAME[sp], '', '### H1 框宽（两头等权 + 门控，h2 = 剪影整体宽）', '',
              '| 方案 | 扣整体偏差框宽误差 % | 原始框宽误差 % | mavic2（扣偏差） | phantom4（扣偏差） |', '|---|---|---|---|---|',
              '| h1（原检测头） | %.2f | %.2f | %.2f | %.2f |' % (A(h['h1_deb']).mean(), A(h['h1_raw']).mean(), A(h['h1_deb_mavic2']).mean(), A(h['h1_deb_phantom4']).mean())]
        for a, nm in (('UT', '融合：教师软剪影'), ('UC', '融合：软标签剪影'), ('UH', '融合：硬标签剪影')):
            L.append('| %s | %.2f | %.2f | %.2f | %.2f |' % (nm, A(h[a + '_deb']).mean(), A(h[a + '_raw']).mean(), A(h[a + '_deb_mavic2']).mean(), A(h[a + '_deb_phantom4']).mean()))
        d1a = A(h['UT_deb']) - A(h['h1_deb'])
        d1b = A(h['UT_deb']) - A(h['UH_deb'])
        L += ['', 'H1a 融合（教师软剪影）− h1：%+.2f 点（各种子 %s；%d/5 变好）' % (d1a.mean(), ' '.join('%+.2f' % x for x in d1a), (d1a < 0).sum()),
              'H1b 教师软剪影 − 硬标签剪影：%+.2f 点（各种子 %s；%d/5 变好）' % (d1b.mean(), ' '.join('%+.2f' % x for x in d1b), (d1b < 0).sum())]
        q = R['H2'][sp]
        L += ['', '### H2 深度（第二段，按另一真实集合无标签对齐）', '', '| 模型 | 扣全局倍数位置中位 cm | 部署位置中位 cm | 检出 % |', '|---|---|---|---|']
        for a, nm in (('P0', '原模型'), ('U0', '第二段：不学剪影'), ('UH', '第二段：硬标签剪影'), ('UC', '第二段：软标签剪影'), ('UT', '第二段：教师软剪影')):
            L.append('| %s | %.1f | %.1f | %.1f |' % (nm, A(q[a + '_deb']).mean(), A(q[a + '_pos']).mean(), A(q[a + '_det']).mean()))
        d2a = A(q['UH_deb']) - A(q['U0_deb'])
        d2b = A(q['UH_pos']) - A(q['P0_pos'])
        ddet = A(q['UH_det']) - A(q['P0_det'])
        dc = A(q['UC_deb']) - A(q['U0_deb'])
        L += ['', 'H2a 硬标签剪影 − 不学剪影（扣全局倍数）：%+.2f cm（各种子 %s；%d/5 变好）' % (d2a.mean(), ' '.join('%+.2f' % x for x in d2a), (d2a < 0).sum()),
              '（描述）软标签剪影 − 不学剪影：%+.2f cm（%d/5）' % (dc.mean(), (dc < 0).sum()),
              'H2b 硬标签剪影部署 − 原模型：%+.2f cm（各种子 %s；%d/5 变好）；检出 %+.2f 个百分点' % (d2b.mean(), ' '.join('%+.2f' % x for x in d2b), (d2b < 0).sum(), ddet.mean()), '']
        res[sp] = dict(H1a=bool(d1a.mean() <= -0.3 and (d1a < 0).sum() >= 4), H1b=bool(d1b.mean() <= -0.3 and (d1b < 0).sum() >= 4),
                       H2a=bool(d2a.mean() <= -1.0 and (d2a < 0).sum() >= 4), H2b=bool(d2b.mean() <= -1.5 and (d2b < 0).sum() >= 4 and ddet.mean() >= -1.0))
    L += ['## 判定（两个真实集合都要成立）', '']
    for hk, nm in (('H1a', 'H1a 教师软剪影两头融合让框宽逐帧更准'), ('H1b', 'H1b 教师软剪影胜过硬标签剪影'), ('H2a', 'H2a 剪影标签监督让第二段逐帧深度更准'), ('H2b', 'H2b 第二段 + 无标签对齐部署比原模型好')):
        ok = res['test'][hk] and res['val'][hk]
        L.append('**%s：%s**（测试 %s / 验证 %s）' % (nm, '成立' if ok else '不成立', '✓' if res['test'][hk] else '✗', '✓' if res['val'][hk] else '✗'))
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
