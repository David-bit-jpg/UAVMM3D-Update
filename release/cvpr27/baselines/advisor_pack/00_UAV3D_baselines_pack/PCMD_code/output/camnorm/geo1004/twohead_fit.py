# -*- coding: utf-8 -*-
"""双头尺寸融合（PREREG_twohead.md，sha256 e69ff3e1）第 2 步：拟合 h2 / h1 置信度小网络（仿真全量测试），在真实测试 / 真实验证上评测框宽与位置。
输出 TWOHEAD_RESULTS.md 与 twohead/analysis.json。"""
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
O = G + '/twohead'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
from uavdet3d.utils import pose_eval as pe  # noqa: E402
import size_vs_center as svc  # noqa: E402

S5 = [int(x) for x in os.environ.get("TH_SEEDS", "0,1,2,3,4").split(",")]
S0 = float(np.sqrt(3.0))
SRC = {'test': G + '/irloc_recs/P0_s%d.pkl', 'val': G + '/irloc_val_recs/P0_s%d.pkl', 'sim': G + '/sim_full_recs/P0_s%d.pkl'}
VARIANTS = [('A1', 'U0'), ('A2', 'UT'), ('A3', 'UT'), ('A4', 'UT'), ('A2', 'UC'), ('A3', 'UC'), ('A4', 'UC'), ('A2', 'UH'), ('A3', 'UH'), ('A4', 'UH')]


def targets(recs, items, seq='xyz'):
    """每个检测：真值框宽 / 预测框宽 的对数（用记录自己的 K），没配上的为 nan；另返回机型"""
    y = np.full(len(items), np.nan)
    pos = {}
    for ri, r in enumerate(recs):
        gt, pred, conf, K = r['gt'], r['pred'], r['conf'], r['K']
        if len(gt) == 0 or len(pred) == 0:
            continue
        used = np.zeros(len(gt), bool)
        for j in np.argsort(-conf):
            d = np.linalg.norm(gt[:, :3] - pred[j, :3], axis=1)
            d[used] = np.inf
            k = int(np.argmin(d))
            if d[k] < 1.0:
                used[k] = True
                wp, wg = svc.uspan(svc.corners(pred[j], seq), K), svc.uspan(svc.corners(gt[k], seq), K)
                if wp > 0 and wg > 0:
                    pos[(ri, int(j))] = np.log(wg / wp)
    for n, (ri, pi) in enumerate(items):
        y[n] = pos.get((int(ri), int(pi)), np.nan)
    typ = np.array([recs[ri]['seq_id'].split('/')[0] for ri, _ in items])
    seqs = np.array([recs[ri]['seq_id'] for ri, _ in items])
    return y, typ, seqs


def feats(F, var, src):
    g = F['geo']
    w, f, z = g[:, 0], g[:, 2], g[:, 3]
    cols = [np.log(w / f), np.log(z)] + [np.log(np.maximum(g[:, 4 + i], 1e-3)) for i in range(3)]
    cols += [F['U0']['d'], F['U0']['lb']]
    if var in ('A2', 'A3', 'A4'):
        a = F[src]
        c = a['colx']
        sw = (c[:, 1] - c[:, 0]) * S0 / 128
        cols += [np.log(np.maximum(sw, 1e-2)), np.log(np.maximum(a['area'], 1e-4)), a['ent']]
    if var in ('A3', 'A4'):
        a = F[src]
        c, rr = a['colx'], a['rowy']
        hl = np.maximum(63.5 - c[:, 0], 0.5) * 2 * S0 / 128
        hr = np.maximum(c[:, 1] - 63.5, 0.5) * 2 * S0 / 128
        cols += [np.log(hl), np.log(hr), np.abs(np.log(hl / hr)), c[:, 2], c[:, 3], np.log1p(c[:, 4]), np.log1p(c[:, 5]),
                 np.log(np.maximum((rr[:, 1] - rr[:, 0]) * S0 / 128, 1e-2)), rr[:, 2], rr[:, 3]]
    if var == 'A4':
        e = g[:, 7:10]
        cols += [np.sin(e[:, 0]), np.cos(e[:, 0]), np.sin(e[:, 1]), np.cos(e[:, 1]), np.sin(e[:, 2]), np.cos(e[:, 2])]
    X = np.stack(cols, 1).astype(np.float32)
    return X


class MLP(nn.Module):
    def __init__(self, d, out):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out))

    def forward(self, x):
        return self.f(x)


def fit(X, y, seed):
    """→ 预测函数：返回 (μ2, log b2, log b1)"""
    torch.manual_seed(seed)
    m = np.isfinite(y) & np.isfinite(X).all(1)
    mu, sd = np.nanmedian(X[m], 0), np.nanstd(X[m], 0) + 1e-6
    Xt = torch.from_numpy(np.nan_to_num((X[m] - mu) / sd)).float()
    yt = torch.from_numpy(y[m]).float()
    h2, h1 = MLP(X.shape[1], 2), MLP(X.shape[1], 1)
    opt = torch.optim.Adam(list(h2.parameters()) + list(h1.parameters()), lr=3e-3, weight_decay=1e-4)
    for _ in range(400):
        o = h2(Xt)
        lb2 = o[:, 1].clamp(-7, 2)
        lb1 = h1(Xt)[:, 0].clamp(-7, 2)
        loss = ((o[:, 0] - yt).abs() * torch.exp(-lb2) + lb2).mean() + (yt.abs() * torch.exp(-lb1) + lb1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def pred(Xn):
        x = torch.from_numpy(np.nan_to_num((Xn - mu) / sd)).float()
        o = h2(x)
        return o[:, 0].numpy(), o[:, 1].clamp(-7, 2).numpy(), h1(x)[:, 0].clamp(-7, 2).numpy()
    return pred


def fuse(mu2, lb2, lb1):
    w2, w1 = np.exp(-2 * lb2), np.exp(-2 * lb1)
    return mu2 * w2 / (w1 + w2)


def deploy(recs, items, yhat, c):
    new = [dict(r, pred=r['pred'].copy()) for r in recs]
    for (ri, pi), v in zip(items, yhat):
        new[int(ri)]['pred'][int(pi), :3] *= np.exp(-(v - c))
    return new


def pos_med(recs):
    s = pe.summarize(pe.match_records(recs, seq='xyz', match='greedy', dist_thresh=1.0), n_frames=len(recs))
    return 100 * s.get('pos_median'), 100 * s['det_rate']


def main():
    D = {sp: {k: pickle.load(open('%s/feats_%s_s%d.pkl' % (O, sp, k), 'rb')) for k in S5} for sp in ('sim', 'test', 'val')}
    RC = {sp: {k: pickle.load(open(SRC[sp] % k, 'rb'))['records'] for k in S5} for sp in ('sim', 'test', 'val')}
    T = {sp: {k: targets(RC[sp][k], D[sp][k]['items']) for k in S5} for sp in ('sim', 'test', 'val')}
    R = {}
    for var, src in VARIANTS:
        key = '%s_%s' % (var, src)
        R[key] = {}
        for k in S5:
            Xs = feats(D['sim'][k], var, src)
            ys, _, seqs = T['sim'][k]
            pred = fit(Xs, ys, k)
            # 仿真：按飞行片段两折交叉拟合
            us = np.array(sorted(set(seqs)))
            half = set(us[np.random.default_rng(k).permutation(len(us))[:len(us) // 2]])
            fa = np.array([s in half for s in seqs])
            yh_sim = np.full(len(ys), np.nan)
            for tr in (fa, ~fa):
                pr = fit(Xs[tr], ys[tr], k)
                m2, b2, b1 = pr(Xs[~tr])
                yh_sim[~tr] = fuse(m2, b2, b1)
            ok = np.isfinite(ys)
            R[key].setdefault('sim', {}).setdefault('fused', []).append(100 * np.median(np.abs(ys[ok] - yh_sim[ok])))
            R[key]['sim'].setdefault('h1', []).append(100 * np.median(np.abs(ys[ok])))
            out = {}
            for sp in ('test', 'val'):
                X = feats(D[sp][k], var, src)
                m2, b2, b1 = pred(X)
                out[sp] = (m2, b2, b1, fuse(m2, b2, b1))
            for sp, other in (('test', 'val'), ('val', 'test')):
                m2, b2, b1, yh = out[sp]
                y, typ, _ = T[sp][k]
                ok = np.isfinite(y)
                r = R[key].setdefault(sp, {})
                r.setdefault('h1', []).append(100 * np.median(np.abs(y[ok])))
                r.setdefault('h2', []).append(100 * np.median(np.abs(y[ok] - m2[ok])))
                r.setdefault('fused', []).append(100 * np.median(np.abs(y[ok] - yh[ok])))
                r.setdefault('w_h2', []).append(float(np.median((np.exp(-2 * b2) / (np.exp(-2 * b1) + np.exp(-2 * b2)))[ok])))
                for t in ('mavic2', 'phantom4'):
                    mt = ok & (typ == t)
                    r.setdefault('h1_' + t, []).append(100 * np.median(np.abs(y[mt])))
                    r.setdefault('fused_' + t, []).append(100 * np.median(np.abs(y[mt] - yh[mt])))
                c = float(np.median(out[other][3]))
                items = D[sp][k]['items']
                r.setdefault('pos_p0', []).append(pos_med(RC[sp][k])[0])
                r.setdefault('pos_cal', []).append(pos_med(deploy(RC[sp][k], items, yh, c))[0])
                r.setdefault('pos_nocal', []).append(pos_med(deploy(RC[sp][k], items, yh, 0.0))[0])
                r.setdefault('c', []).append(c)
            print(key, k, ' | '.join('%s 框宽 h1 %.2f h2 %.2f 融合 %.2f 位置 P0 %.1f 校准 %.1f' % (
                sp, R[key][sp]['h1'][-1], R[key][sp]['h2'][-1], R[key][sp]['fused'][-1], R[key][sp]['pos_p0'][-1], R[key][sp]['pos_cal'][-1])
                for sp in ('test', 'val')), flush=True)
    json.dump(R, open(O + '/analysis%s.json' % os.environ.get('TH_TAG', ''), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    report(R)


def report(R):
    L = ['# 双头尺寸融合（PREREG_twohead.md，sha256 e69ff3e1）', '',
         'h1 = 原检测网络的框宽；h2 = 放大裁窗 + 学生剪影 → 框宽；按各自置信度融合。框宽误差 = |log(框宽 / 真框宽)| 的中位（%，P0 配对检测上）。', '']
    for sp, nm in (('test', '真实测试'), ('val', '真实验证')):
        L += ['## ' + nm, '', '| 方案 | h2 单独 | 融合后 | h1 | 融合 − h1 | 变好种子 | h2 权重中位 | mavic2 融合/h1 | phantom4 融合/h1 | 位置 P0 → 校准④ / 不校准 cm |', '|---|---|---|---|---|---|---|---|---|---|']
        for key in R:
            r = R[key][sp]
            d = np.array(r['fused']) - np.array(r['h1'])
            L.append('| %s | %.2f | %.2f | %.2f | %+.2f | %d/5 | %.2f | %.1f / %.1f | %.1f / %.1f | %.1f → %.1f / %.1f |' % (
                key, np.mean(r['h2']), np.mean(r['fused']), np.mean(r['h1']), d.mean(), (d < 0).sum(), np.mean(r['w_h2']),
                np.mean(r['fused_mavic2']), np.mean(r['h1_mavic2']), np.mean(r['fused_phantom4']), np.mean(r['h1_phantom4']),
                np.mean(r['pos_p0']), np.mean(r['pos_cal']), np.mean(r['pos_nocal'])))
        L.append('')
    L += ['## 仿真（按飞行片段两折交叉拟合）', '', '| 方案 | 融合后 | h1 |', '|---|---|---|']
    for key in R:
        L.append('| %s | %.2f | %.2f |' % (key, np.mean(R[key]['sim']['fused']), np.mean(R[key]['sim']['h1'])))

    def cmpf(a, b, sp):
        return np.array(R[a][sp]['fused']) - np.array(R[b][sp]['fused'])

    def ok(d, thr):
        return d.mean() <= thr and (d < 0).sum() >= 4
    main_ok = all(ok(np.array(R['A4_UT'][sp]['fused']) - np.array(R['A4_UT'][sp]['h1']), -0.5) for sp in ('test', 'val'))
    guard = all(np.mean(R['A4_UT'][sp]['pos_cal']) - np.mean(R['A4_UT'][sp]['pos_p0']) <= 0.5 for sp in ('test', 'val'))
    s1 = all(ok(cmpf('A3_UT', 'A2_UT', sp), -0.3) for sp in ('test', 'val'))
    s2 = all(ok(cmpf('A4_UT', 'A4_UC', sp), -0.3) and ok(cmpf('A4_UT', 'A4_UH', sp), -0.3) for sp in ('test', 'val'))
    s3 = all(ok(cmpf('A4_UT', 'A3_UT', sp), -0.3) for sp in ('test', 'val'))
    L += ['', '## 判定', '']
    for sp in ('test', 'val'):
        for nm, (a, b) in (('① 对称半宽 A3−A2', ('A3_UT', 'A2_UT')), ('② 教师软剪影 vs 覆盖率标签', ('A4_UT', 'A4_UC')), ('② 教师软剪影 vs 硬标签', ('A4_UT', 'A4_UH')), ('③ 朝向 A4−A3', ('A4_UT', 'A3_UT'))):
            d = cmpf(a, b, sp)
            L.append('%s %s：%+.2f 点（%d/5 变好）' % (sp, nm, d.mean(), (d < 0).sum()))
    L += ['', '**主判据（A4(UT) 框宽误差比 h1 低 ≥ 0.5 点且 ≥ 4/5，两个真实集合）：%s**' % ('成立' if main_ok else '不成立'),
          '**护栏（位置不比 P0 差超过 0.5 cm）：%s**' % ('守住' if guard else '没守住'),
          '**次判据 ①：%s；②：%s；③：%s**' % tuple('成立' if x else '不成立' for x in (s1, s2, s3))]
    open(G + '/TWOHEAD_RESULTS%s.md' % os.environ.get('TH_TAG', ''), 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
