# -*- coding: utf-8 -*-
"""PREREG_softkd.md 判定 → SOFTKD_RESULTS.md / softkd/eval/analysis.json。"""
import json
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
O = G + '/softkd/eval'
sys.path.insert(0, ROOT)
from uavdet3d.utils import pose_eval as pe  # noqa: E402

S5 = range(5)
ARMS = ('U0', 'UH', 'UC', 'UT')
V1 = 0.01145
GUARD = float(np.log(1.4))
SRC = {'test': G + '/irloc_recs/P0_s%d.pkl', 'val': G + '/irloc_val_recs/P0_s%d.pkl', 'sim': G + '/sim_full_recs/P0_s%d.pkl'}
NAME = {'test': '真实测试', 'val': '真实验证', 'sim': '仿真全量测试'}


def load(split, k):
    return pickle.load(open(SRC[split] % k, 'rb'))['records'], pickle.load(open('%s/pred_%s_s%d.pkl' % (O, split, k), 'rb'))


def cval(pr):
    d = np.concatenate([x[0] for x in pr])
    d = d[np.isfinite(d) & (np.abs(d) <= GUARD)]
    return float(np.median(d))


def deploy(recs, pr, c, mode):
    new = []
    for r, (d, lb) in zip(recs, pr):
        p = r['pred'].copy()
        if len(d):
            ok = np.isfinite(d) & (np.abs(d) <= GUARD)
            dd = np.where(ok, np.nan_to_num(d) - c, 0.0)
            if mode == 'fuse':
                dd = dd * V1 / (V1 + 2 * np.exp(2 * np.nan_to_num(lb)))
            p[:, :3] *= np.exp(dd)[:, None]
        new.append(dict(r, pred=p))
    return new


def summ(recs):
    s = pe.summarize(pe.match_records(recs, seq='xyz', match='greedy', dist_thresh=1.0), n_frames=len(recs))
    return 100 * s['det_rate'], 100 * s.get('pos_median')


def top1(recs):
    out = {}
    for r in recs:
        if len(r['gt']) and len(r['pred']):
            j = int(np.argmax(r['conf']))
            g, q = r['gt'][0], r['pred'][j]
            if np.linalg.norm(g[:3] - q[:3]) < 1.0:
                out[(r['seq_id'], r['frame_id'])] = (g[:3], q[:3])
    return out


def debiased(rows, keys, sel=None):
    g = np.array([rows[q][0] for q in keys])
    p = np.array([rows[q][1] for q in keys])
    if sel is not None:
        g, p = g[sel], p[sel]
    s = 1.0 / np.median(p[:, 2] / g[:, 2])
    return 100 * float(np.median(np.linalg.norm(p * s - g, axis=1)))


def boot_debiased(rowsA, rowsB, keysk, n=1000):
    """按飞行片段重抽样（真实按机型分层）：5 个种子平均的 扣全局倍数位置中位 A − B 的 95% 区间"""
    seqs = sorted(set(q[0] for keys in keysk for q in keys))
    strata = {}
    for s in seqs:
        strata.setdefault(s.split('/')[0] if s.split('/')[0] in ('mavic2', 'phantom4') else 'all', []).append(s)
    pre = []
    for k, keys in enumerate(keysk):
        sid = np.array([seqs.index(q[0]) for q in keys])
        ga = np.array([rowsA[k][q][0] for q in keys]); pa = np.array([rowsA[k][q][1] for q in keys])
        pb = np.array([rowsB[k][q][1] for q in keys])
        pre.append((sid, ga, pa, pb))
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        pick = []
        for _, ss in strata.items():
            pick += list(rng.choice([seqs.index(s) for s in ss], size=len(ss), replace=True))
        cnt = np.bincount(pick, minlength=len(seqs))
        v = []
        for sid, g, pa, pb in pre:
            wgt = cnt[sid]
            idx = np.repeat(np.arange(len(sid)), wgt)
            if len(idx) == 0:
                continue
            gg, aa, bb = g[idx], pa[idx], pb[idx]
            sa = 1.0 / np.median(aa[:, 2] / gg[:, 2]); sb = 1.0 / np.median(bb[:, 2] / gg[:, 2])
            v.append(np.median(np.linalg.norm(aa * sa - gg, axis=1)) - np.median(np.linalg.norm(bb * sb - gg, axis=1)))
        vals.append(100 * np.mean(v))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    L = ['# 严格软剪影蒸馏 + 不确定度后融合（PREREG_softkd.md，sha256 fb284f2a）', '',
         'U0 不监督剪影 / UH 红外剪影硬标签 / UC 红外剪影覆盖率软标签 / UT 红外教师网络软剪影蒸馏（不看标签）。都带不确定度。P0 = 普通彩色检测模型（第一段）。', '']
    res = {}
    for split in ('test', 'val', 'sim'):
        other = {'test': 'val', 'val': 'test'}.get(split)
        D = {}       # 部署原始位置、检出
        DB = {a: [] for a in ('P0',) + ARMS}
        DBT = {t: {a: [] for a in ('U0', 'UT')} for t in ('mavic2', 'phantom4')}
        rho = {a: [] for a in ARMS}
        rows_k = {a: [] for a in ('P0',) + ARMS}
        keys_k = []
        cs = {a: [] for a in ARMS}
        for k in S5:
            recs, pr = load(split, k)
            D.setdefault('P0', []).append(summ(recs))
            rows = {'P0': top1(recs)}
            for a in ARMS:
                c = cval(load(other, k)[1][a]) if other else 0.0
                cs[a].append(c)
                for mode in ('w1', 'fuse'):
                    D.setdefault((a, mode), []).append(summ(deploy(recs, pr[a], c, mode)))
                rows[a] = top1(deploy(recs, pr[a], c, 'w1'))
                # 不确定度排序：P0 最高分检测（真值 3D 距离 < 1 m）上，b 与 |需要的修正 − (δ̂ − c)| 的秩相关
                need, est, b = [], [], []
                for r, (d, lb) in zip(recs, pr[a]):
                    if len(r['gt']) and len(r['pred']):
                        j = int(np.argmax(r['conf']))
                        g, q = r['gt'][0], r['pred'][j]
                        if np.linalg.norm(g[:3] - q[:3]) < 1.0 and np.isfinite(d[j]) and abs(d[j]) <= GUARD:
                            need.append(np.log(g[2] / q[2])); est.append(d[j] - c); b.append(lb[j])
                rho[a].append(float(spearmanr(b, np.abs(np.array(need) - np.array(est)))[0]))
            keys = sorted(set.intersection(*[set(v) for v in rows.values()]))
            keys_k.append(keys)
            for a in rows:
                DB[a].append(debiased(rows[a], keys))
                rows_k[a].append(rows[a])
            ty = np.array([q[0].split('/')[0] for q in keys])
            for t in DBT:
                if (ty == t).sum() > 20:
                    for a in ('U0', 'UT'):
                        DBT[t][a].append(debiased(rows[a], keys, ty == t))
        r = dict(c=cs, rho=rho)
        L += ['## ' + NAME[split], '']
        if other:
            L.append('对齐量 c（用%s算）：' % NAME[other] + '；'.join('%s %s' % (a, ' '.join('%.3f' % x for x in cs[a])) for a in ARMS))
        else:
            L.append('仿真：第二段原样（c = 0）')
        L += ['', '| 模型 | 扣全局倍数后位置中位 cm（每帧精度） | 部署位置中位 cm（权重 1） | 部署位置中位 cm（不确定度融合） | 检出 %（权重 1） | 不确定度秩相关 |', '|---|---|---|---|---|---|']
        L.append('| P0 | %.1f | %.1f | — | %.1f | — |' % (np.mean(DB['P0']), np.mean([x[1] for x in D['P0']]), np.mean([x[0] for x in D['P0']])))
        for a in ARMS:
            L.append('| %s | %.1f | %.1f | %.1f | %.1f | %+.3f |' % (a, np.mean(DB[a]), np.mean([x[1] for x in D[(a, 'w1')]]),
                                                               np.mean([x[1] for x in D[(a, 'fuse')]]), np.mean([x[0] for x in D[(a, 'w1')]]), np.mean(rho[a])))
        L += ['', '| 比较（扣全局倍数口径，cm） | 平均 | 各种子 | 变好的种子 |', '|---|---|---|---|']
        cmp = {}
        for x, y in (('UT', 'U0'), ('UT', 'UC'), ('UT', 'UH'), ('UC', 'U0'), ('UH', 'U0'), ('U0', 'P0'), ('UT', 'P0')):
            d = [a - b for a, b in zip(DB[x], DB[y])]
            cmp['%s-%s' % (x, y)] = d
            L.append('| %s − %s | %+.2f | %s | %d/5 |' % (x, y, np.mean(d), ' '.join('%+.2f' % v for v in d), sum(v < 0 for v in d)))
        ci = boot_debiased(rows_k['UT'], rows_k['U0'], keys_k)
        L += ['', '按飞行片段 95%% 区间（扣全局倍数口径）UT − U0：[%+.2f, %+.2f] cm' % tuple(ci)]
        for t in DBT:
            if DBT[t]['UT']:
                d = [a - b for a, b in zip(DBT[t]['UT'], DBT[t]['U0'])]
                L.append('按机型（扣该机型子集的全局倍数）%s UT − U0：%+.2f cm（%d/%d 变好）' % (t, np.mean(d), sum(v < 0 for v in d), len(d)))
        dfu = [a[1] - b[1] for a, b in zip(D[('UT', 'fuse')], D[('UT', 'w1')])]
        dp = {a: [x[1] - y[1] for x, y in zip(D[(a, 'w1')], D['P0'])] for a in ARMS}
        L.append('部署（权重 1）相对 P0：' + '；'.join('%s %+.2f cm（%d/5）' % (a, np.mean(dp[a]), sum(v < 0 for v in dp[a])) for a in ARMS))
        L.append('UT 不确定度融合 − 权重 1：%+.2f cm（%d/5 变好）' % (np.mean(dfu), sum(v < 0 for v in dfu)))
        L.append('')
        r.update(debiased=DB, cmp=cmp, ci_ut_u0=ci, fuse_minus_w1=dfu, deploy_minus_p0=dp,
                 deploy={str(kk): v for kk, v in D.items()})
        res[split] = r
    ok = lambda s, key: np.mean(res[s]['cmp'][key]) <= -1.0 and sum(v < 0 for v in res[s]['cmp'][key]) >= 4
    main_ok = ok('test', 'UT-U0') and ok('val', 'UT-U0')
    sec1 = all(ok(s, kk) for s in ('test', 'val') for kk in ('UT-UC', 'UT-UH'))
    sec2 = all(np.mean(res[s]['rho']['UT']) >= 0.10 and sum(v <= 0 for v in res[s]['fuse_minus_w1']) >= 4 for s in ('test', 'val'))
    L += ['## 判定', '',
          '**主判据（扣全局倍数口径 UT − U0 ≤ −1.0 cm 且 ≥ 4/5，真实测试与验证都成立）：%s**' % ('成立' if main_ok else '不成立'),
          '**次判据 1（教师软剪影胜过硬标签与覆盖率软标签）：%s**' % ('成立' if sec1 else '不成立'),
          '**次判据 2（不确定度秩相关 ≥ 0.10 且融合不差于权重 1）：%s**' % ('成立' if sec2 else '不成立')]
    res['verdict'] = dict(main=bool(main_ok), sec1=bool(sec1), sec2=bool(sec2))
    open(G + '/SOFTKD_RESULTS.md', 'w', encoding='utf-8').write('\n'.join(L))
    json.dump(res, open(O + '/analysis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
