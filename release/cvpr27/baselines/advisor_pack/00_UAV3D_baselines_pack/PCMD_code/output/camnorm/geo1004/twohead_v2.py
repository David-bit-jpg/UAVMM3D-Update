# -*- coding: utf-8 -*-
"""双头尺寸融合 v2（PREREG_twohead_v2.md，sha256 b13b1959）：h2 = 学生剪影直接量宽 × 仿真定的换算常数；融合 = 等权 / 物理一致性门控。
判据只用检测种子 1–4；种子 0 照报。输出 TWOHEAD_V2_RESULTS.md 与 twohead/analysis_v2.json。"""
import json
import pickle
import sys

import numpy as np

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
O = G + '/twohead'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
import twohead_fit as tf  # noqa: E402

S0 = float(np.sqrt(3.0))
SEEDS = range(5)
JUDGE = (1, 2, 3, 4)
SRCS = ('UT', 'UC', 'UH')


def ratios(F, a):
    c = F[a]['colx']
    xl, xr, tl, tr = c[:, 0], c[:, 1], c[:, 4], c[:, 5]
    hl, hr = 63.5 - xl, xr - 63.5
    bad = ~np.isfinite(xl) | (hl < 0.5) | (hr < 0.5)
    rP = (xr - xl) * S0 / 128
    best = np.where(tl < tr, 2 * hl, np.where(tr < tl, 2 * hr, hl + hr))
    rY = best * S0 / 128
    asym = np.abs(np.log(np.maximum(hl, 0.5) / np.maximum(hr, 0.5)))
    rP[bad] = np.nan
    rY[bad] = np.nan
    asym[bad] = np.nan
    return rP, rY, asym


class Conv:
    """仿真上定换算常数（可按一个欧拉角分 8 档）与门限"""

    def __init__(self, Fs, ys, a, mode):
        rP, rY, asym = ratios(Fs, a)
        r = rP if mode == 'P' else rY
        z = ys - np.log(r)
        ok = np.isfinite(z)
        self.mode, self.a = mode, a
        self.kappa = float(np.median(z[ok]))
        self.ang, self.edges, self.kb = None, None, None
        if mode == 'O':
            best = None
            for i in range(3):
                ang = Fs['geo'][:, 7 + i]
                edges = np.quantile(ang[ok], np.linspace(0, 1, 9)[1:-1])
                b = np.digitize(ang, edges)
                kb = np.array([np.median(z[ok & (b == j)]) if (ok & (b == j)).sum() > 20 else self.kappa for j in range(8)])
                resid = np.median(np.abs(z[ok] - kb[b[ok]]))
                if best is None or resid < best[0]:
                    best = (resid, i, edges, kb)
            _, self.ang, self.edges, self.kb = best
        e2 = self.e2(Fs)
        self.q_asym = float(np.nanquantile(asym, 0.9))
        self.q_dis = float(np.nanquantile(np.abs(e2), 0.9))

    def e2(self, F):
        rP, rY, _ = ratios(F, self.a)
        r = rP if self.mode == 'P' else rY
        if self.mode == 'O':
            b = np.digitize(F['geo'][:, 7 + self.ang], self.edges)
            return np.log(r) + self.kb[b]
        return np.log(r) + self.kappa

    def fuse(self, F, gated):
        e2 = self.e2(F)
        yh = 0.5 * e2
        if gated:
            _, _, asym = ratios(F, self.a)
            g = np.isfinite(e2) & (asym <= self.q_asym) & (np.abs(e2) <= self.q_dis)
        else:
            g = np.isfinite(e2)
        return np.where(g, yh, 0.0), e2, g


def main():
    D = {sp: {k: pickle.load(open('%s/feats_%s_s%d.pkl' % (O, sp, k), 'rb')) for k in SEEDS} for sp in ('sim', 'test', 'val')}
    RC = {sp: {k: pickle.load(open(tf.SRC[sp] % k, 'rb'))['records'] for k in SEEDS} for sp in ('sim', 'test', 'val')}
    T = {sp: {k: tf.targets(RC[sp][k], D[sp][k]['items']) for k in SEEDS} for sp in ('sim', 'test', 'val')}
    CONF = [(a, m, f) for a in SRCS for m in ('P', 'Y', 'O') for f in ('F0', 'F1')]
    R = {}
    for k in SEEDS:
        ys = T['sim'][k][0]
        for a, m, f in CONF:
            key = '%s_%s_%s' % (a, m, f)
            cv = Conv(D['sim'][k], ys, a, m)
            yh_o = {}
            for sp in ('test', 'val'):
                yh_o[sp] = cv.fuse(D[sp][k], f == 'F1')
            for sp, other in (('test', 'val'), ('val', 'test')):
                y, typ, _ = T[sp][k]
                yh, e2, g = yh_o[sp]
                ok = np.isfinite(y)
                r = R.setdefault(key, {}).setdefault(sp, {})
                r.setdefault('h1', {})[k] = 100 * np.median(np.abs(y[ok]))
                r.setdefault('fused', {})[k] = 100 * np.median(np.abs(y[ok] - yh[ok]))
                o2 = ok & np.isfinite(e2)
                r.setdefault('h2', {})[k] = 100 * np.median(np.abs(y[o2] - e2[o2]))
                r.setdefault('fused_debiased', {})[k] = 100 * np.median(np.abs(y[ok] - yh[ok] - np.median(y[ok] - yh[ok])))
                r.setdefault('h1_debiased', {})[k] = 100 * np.median(np.abs(y[ok] - np.median(y[ok])))
                r.setdefault('bias_fused', {})[k] = 100 * np.median(y[ok] - yh[ok])
                r.setdefault('bias_h1', {})[k] = 100 * np.median(y[ok])
                r.setdefault('gate_on', {})[k] = float(g.mean())
                for t in ('mavic2', 'phantom4'):
                    mt = ok & (typ == t)
                    r.setdefault('h1_' + t, {})[k] = 100 * np.median(np.abs(y[mt]))
                    r.setdefault('fused_' + t, {})[k] = 100 * np.median(np.abs(y[mt] - yh[mt]))
                if (a, m, f) in (('UT', 'Y', 'F1'), ('UT', 'Y', 'F0'), ('UT', 'P', 'F1'), ('UT', 'O', 'F1'), ('UC', 'Y', 'F1'), ('UH', 'Y', 'F1')):
                    c = float(np.median(yh_o[other][0]))
                    items = D[sp][k]['items']
                    r.setdefault('pos_p0', {})[k] = tf.pos_med(RC[sp][k])[0]
                    r.setdefault('pos_cal', {})[k] = tf.pos_med(tf.deploy(RC[sp][k], items, yh, c))[0]
            ysim = T['sim'][k][0]
            yh_s, _, _ = cv.fuse(D['sim'][k], f == 'F1')
            oks = np.isfinite(ysim)
            R[key].setdefault('sim', {}).setdefault('fused', {})[k] = 100 * np.median(np.abs(ysim[oks] - yh_s[oks]))
            R[key]['sim'].setdefault('h1', {})[k] = 100 * np.median(np.abs(ysim[oks]))
        print('种子 %d：UT_Y_F1 真实测试 h1 %.2f → 融合 %.2f；真实验证 h1 %.2f → 融合 %.2f' % (
            k, R['UT_Y_F1']['test']['h1'][k], R['UT_Y_F1']['test']['fused'][k], R['UT_Y_F1']['val']['h1'][k], R['UT_Y_F1']['val']['fused'][k]), flush=True)
    json.dump(R, open(O + '/analysis_v2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    report(R)


def arr(R, key, sp, m, seeds=JUDGE):
    return np.array([R[key][sp][m][k] for k in seeds])


def report(R):
    L = ['# 双头尺寸融合 v2（PREREG_twohead_v2.md，sha256 b13b1959）', '',
         'h1 = 原检测网络框宽；h2 = 学生剪影直接量宽 × 仿真换算常数；融合在 log 框宽上。框宽误差 = |log(框宽/真框宽)| 中位（%）。判据只用种子 1–4；括号里是种子 0。', '']
    for sp, nm in (('test', '真实测试'), ('val', '真实验证')):
        L += ['## ' + nm, '', '| 剪影来源_量法_融合 | h2 单独 | 融合 | h1 | 融合 − h1 | 变好（种子1–4） | 去偏 融合/h1 | 偏差 融合/h1 | 门控通过 | mavic2 融合/h1 | phantom4 融合/h1 | 种子 0 融合/h1 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for key in R:
            r = R[key][sp]
            d = arr(R, key, sp, 'fused') - arr(R, key, sp, 'h1')
            L.append('| %s | %.2f | %.2f | %.2f | %+.2f | %d/4 | %.2f / %.2f | %+.1f / %+.1f | %.0f%% | %.1f / %.1f | %.1f / %.1f | %.2f / %.2f |' % (
                key, arr(R, key, sp, 'h2').mean(), arr(R, key, sp, 'fused').mean(), arr(R, key, sp, 'h1').mean(), d.mean(), (d < 0).sum(),
                arr(R, key, sp, 'fused_debiased').mean(), arr(R, key, sp, 'h1_debiased').mean(), arr(R, key, sp, 'bias_fused').mean(), arr(R, key, sp, 'bias_h1').mean(),
                100 * arr(R, key, sp, 'gate_on').mean(), arr(R, key, sp, 'fused_mavic2').mean(), arr(R, key, sp, 'h1_mavic2').mean(),
                arr(R, key, sp, 'fused_phantom4').mean(), arr(R, key, sp, 'h1_phantom4').mean(), r['fused'][0], r['h1'][0]))
        L += ['', '位置中位（cm，④ 校准，种子 1–4 平均；P0 → 方案）：' + '；'.join(
            '%s %.1f → %.1f' % (key, arr(R, key, sp, 'pos_p0').mean(), arr(R, key, sp, 'pos_cal').mean()) for key in R if 'pos_cal' in R[key][sp]), '']
    L += ['## 仿真（换算常数与门限在仿真上定，属样本内）', '', '| 方案 | 融合 | h1 |', '|---|---|---|']
    for key in R:
        L.append('| %s | %.2f | %.2f |' % (key, arr(R, key, 'sim', 'fused', SEEDS).mean(), arr(R, key, 'sim', 'h1', SEEDS).mean()))

    def ok(d, thr):
        return d.mean() <= thr and (d < 0).sum() >= 3

    def dd(a, b, sp):
        return arr(R, a, sp, 'fused') - arr(R, b, sp, 'fused')
    main_ok = all(ok(arr(R, 'UT_Y_F1', sp, 'fused') - arr(R, 'UT_Y_F1', sp, 'h1'), -0.5) for sp in ('test', 'val'))
    ir = all(ok(dd('UT_Y_F1', 'UC_Y_F1', sp), -0.5) and ok(dd('UT_Y_F1', 'UH_Y_F1', sp), -0.5) for sp in ('test', 'val'))
    s1 = all(ok(dd('UT_Y_F1', 'UT_P_F1', sp), -0.3) for sp in ('test', 'val'))
    s3 = all(ok(dd('UT_O_F1', 'UT_Y_F1', sp), -0.3) for sp in ('test', 'val'))
    sg = all(ok(dd('UT_Y_F1', 'UT_Y_F0', sp), -0.3) for sp in ('test', 'val'))
    guard = all(arr(R, 'UT_Y_F1', sp, 'pos_cal').mean() - arr(R, 'UT_Y_F1', sp, 'pos_p0').mean() <= 0.5 for sp in ('test', 'val'))
    L += ['', '## 判定（种子 1–4）', '']
    for sp in ('test', 'val'):
        for nm, a, b in (('② UT vs UC', 'UT_Y_F1', 'UC_Y_F1'), ('② UT vs UH', 'UT_Y_F1', 'UH_Y_F1'), ('① 对称 Y vs 整体 P', 'UT_Y_F1', 'UT_P_F1'),
                         ('③ 朝向 O vs Y', 'UT_O_F1', 'UT_Y_F1'), ('门控 F1 vs 等权 F0', 'UT_Y_F1', 'UT_Y_F0')):
            d = dd(a, b, sp)
            L.append('%s %s：%+.2f 点（%d/4 变好）' % (sp, nm, d.mean(), (d < 0).sum()))
    L += ['', '**主判据（UT_Y_F1 框宽误差比 h1 低 ≥ 0.5 点且 ≥ 3/4，两个真实集合）：%s**' % ('成立' if main_ok else '不成立'),
          '**② 红外教师软剪影胜过两种标签（≥ 0.5 点，≥ 3/4，两个集合）：%s**' % ('成立' if ir else '不成立'),
          '**次判据 ①：%s；③：%s；门控：%s**' % tuple('成立' if x else '不成立' for x in (s1, s3, sg)),
          '**护栏（位置不比 P0 差超过 0.5 cm）：%s**' % ('守住' if guard else '没守住')]
    open(G + '/TWOHEAD_V2_RESULTS.md', 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
