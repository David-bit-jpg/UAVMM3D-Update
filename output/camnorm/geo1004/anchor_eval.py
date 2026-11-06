# -*- coding: utf-8 -*-
"""PREREG_anchor.md（sha256 02184ce6）判定 → ANCHOR_RESULTS.md / confirm/anchor.json。
第二段 = PREREG_confirm 的全新种子（confirm/feats_*），检测 = P0_s{k} 标准口径记录；逐帧原始尺寸来自 irloc_raw_recs。"""
import json
import pickle
import sys

import numpy as np

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
import softkd_analyze as ska  # noqa: E402
import confirm_analyze as ca  # noqa: E402

S5 = range(5)
GUARD = float(np.log(1.4))
CAL = {'test': G + '/irloc_recs/P0_s%d.pkl', 'val': G + '/irloc_val_recs/P0_s%d.pkl'}
NAME = {'test': '真实测试', 'val': '真实验证'}


def fnum(f):
    return int(''.join(ch for ch in f if ch.isdigit()))


def flights(recs):
    out = {}
    for i, r in enumerate(recs):
        out.setdefault(r['seq_id'], []).append(i)
    return {s: sorted(ix, key=lambda i: fnum(recs[i]['frame_id'])) for s, ix in out.items()}


def size_scale(recs, raw):
    """A3：每帧的 L_t / L_cal"""
    s = np.ones(len(recs))
    for _, ix in flights(recs).items():
        acc = []
        for i in ix:
            r, q = recs[i], raw[i]
            for j in range(len(q['pred'])):
                if q['conf'][j] >= 0.7:
                    acc.append(np.linalg.norm(q['pred'][j, 3:6]))
            if len(acc) >= 10 and len(r['pred']):
                s[i] = np.median(acc) / np.linalg.norm(r['pred'][0, 3:6])
    return s


def causal_c(recs, pr, c0):
    """A2：每个检测的 c = 该飞行片段此前（不含当前）通过门限的 δ̂ 中位；前 20 个用 c0"""
    cs = [np.full(len(r['pred']), c0) for r in recs]
    for _, ix in flights(recs).items():
        acc = []
        for i in ix:
            d = pr[i][0]
            if len(acc) >= 20:
                cs[i][:] = np.median(acc)
            for v in d:
                if np.isfinite(v) and abs(v) <= GUARD:
                    acc.append(v)
    return cs


def deploy(recs, pr, c, scale=None):
    new = []
    for i, (r, (d, _)) in enumerate(zip(recs, pr)):
        p = r['pred'].copy()
        if len(d):
            ci = c[i] if isinstance(c, list) else c
            ok = np.isfinite(d) & (np.abs(d) <= GUARD)
            p[:, :3] *= np.exp(np.where(ok, np.nan_to_num(d) - ci, 0.0))[:, None]
        if scale is not None:
            p[:, :3] *= scale[i]
        new.append(dict(r, pred=p))
    return new


def depth_bias(recs):
    out = {}
    for r in recs:
        if len(r['gt']) and len(r['pred']):
            g, q = r['gt'][0], r['pred'][0]
            if np.linalg.norm(g[:3] - q[:3]) < 1.0:
                out.setdefault(r['seq_id'].split('/')[0], []).append(np.log(q[2] / g[2]))
    return {t: 100 * float(np.median(v)) for t, v in out.items()}


def main():
    R = {}
    for k in S5:
        F = {sp: pickle.load(open('%s/confirm/feats_%s_s%d.pkl' % (G, sp, k), 'rb')) for sp in ('test', 'val')}
        RC = {sp: pickle.load(open(CAL[sp] % k, 'rb'))['records'] for sp in ('test', 'val')}
        for sp, other in (('test', 'val'), ('val', 'test')):
            recs = RC[sp]
            raw = pickle.load(open('%s/irloc_raw_recs/%s_P0_s%d.pkl' % (G, sp, k), 'rb'))['records']
            assert len(raw) == len(recs)
            for a, b in zip(recs, raw):
                assert a['seq_id'] == b['seq_id'] and a['frame_id'] == b['frame_id'] and len(a['pred']) == len(b['pred'])
                assert np.allclose(a['conf'], b['conf'], atol=1e-5), '检测不一致'
            sc = size_scale(recs, raw)
            r = R.setdefault(sp, {})
            base = ska.summ(recs)
            r.setdefault('P0', []).append(base)
            r.setdefault('P0_bias', []).append(depth_bias(recs))
            a3 = deploy(recs, [(np.zeros(0), np.zeros(0)) if len(x['pred']) == 0 else (np.full(len(x['pred']), np.nan), np.full(len(x['pred']), np.nan)) for x in recs], 0.0, sc)
            r.setdefault('A3only', []).append(ska.summ(a3))
            r.setdefault('A3only_bias', []).append(depth_bias(a3))
            r.setdefault('scale_med', []).append({t: float(np.median([sc[i] for i, x in enumerate(recs) if x['seq_id'].startswith(t)])) for t in ('mavic2', 'phantom4')})
            for arm in ('UH', 'UT'):
                pr = ca.pr_from_feats(F[sp], recs, arm)
                c_other = ska.cval(ca.pr_from_feats(F[other], RC[other], arm))
                c_same = ska.cval(pr)
                variants = {'A0': deploy(recs, pr, c_other), 'A1': deploy(recs, pr, c_same), 'A2': deploy(recs, pr, causal_c(recs, pr, c_other)),
                            'A3': deploy(recs, pr, c_other, sc)}
                for v, rr in variants.items():
                    r.setdefault('%s_%s' % (v, arm), []).append(ska.summ(rr))
                    if arm == 'UH':
                        r.setdefault('%s_%s_bias' % (v, arm), []).append(depth_bias(rr))
        print('种子 %d：' % k + '；'.join('%s P0 %.1f A3单独 %.1f A0 %.1f A3+UH %.1f' % (sp, R[sp]['P0'][-1][1], R[sp]['A3only'][-1][1], R[sp]['A0_UH'][-1][1], R[sp]['A3_UH'][-1][1]) for sp in ('test', 'val')), flush=True)
    json.dump(R, open(G + '/confirm/anchor.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    report(R)


def report(R):
    L = ['# 部署整体尺度的无标签锚定（PREREG_anchor.md，sha256 02184ce6）', '',
         'A0 另一集合中位（现做法）/ A1 同集合中位 / A2 按飞行片段因果滑动中位 / A3 按飞行片段因果尺寸自标定（主）。第二段 = PREREG_confirm 全新种子；UH 硬标签剪影（主）、UT 教师软剪影。', '']
    res = {}
    for sp in ('test', 'val'):
        r = R[sp]
        pos = lambda key: np.array([x[1] for x in r[key]])
        det = lambda key: np.array([x[0] for x in r[key]])
        L += ['## ' + NAME[sp], '', '| 方案 | 部署位置中位 cm | − P0 | 变好种子 | 检出 % | 深度偏差 mavic2 / phantom4 % |', '|---|---|---|---|---|---|']
        bias = lambda key: (np.mean([b.get('mavic2', np.nan) for b in r[key]]), np.mean([b.get('phantom4', np.nan) for b in r[key]]))
        L.append('| P0 | %.1f | — | — | %.1f | %+.1f / %+.1f |' % (pos('P0').mean(), det('P0').mean(), *bias('P0_bias')))
        L.append('| A3 单独（只改尺寸） | %.1f | %+.2f | %d/5 | %.1f | %+.1f / %+.1f |' % (pos('A3only').mean(), (pos('A3only') - pos('P0')).mean(), ((pos('A3only') - pos('P0')) < 0).sum(), det('A3only').mean(), *bias('A3only_bias')))
        for v in ('A0', 'A1', 'A2', 'A3'):
            for arm in ('UH', 'UT'):
                key = '%s_%s' % (v, arm)
                d = pos(key) - pos('P0')
                bs = ' %+.1f / %+.1f |' % bias(key + '_bias') if arm == 'UH' else ' — |'
                L.append('| %s + 第二段 %s | %.1f | %+.2f | %d/5 | %.1f |%s' % (v, arm, pos(key).mean(), d.mean(), (d < 0).sum(), det(key).mean(), bs))
        L.append('')
        L.append('A3 尺寸倍数中位（L_t / L_cal，按机型，5 种子平均）：mavic2 %.3f / phantom4 %.3f' % (
            np.mean([x['mavic2'] for x in r['scale_med']]), np.mean([x['phantom4'] for x in r['scale_med']])))
        L.append('')
        ok = lambda key, thr: bool((pos(key) - pos('P0')).mean() <= thr and ((pos(key) - pos('P0')) < 0).sum() >= 4 and (det(key) - det('P0')).mean() >= -1.0)
        res[sp] = dict(main=ok('A3_UH', -1.5), a3only=ok('A3only', -1.0), A1=ok('A1_UH', -1.5), A2=ok('A2_UH', -1.5), A0=ok('A0_UH', -1.5))
    L += ['## 判定（两个真实集合都要成立）', '']
    for key, nm in (('main', '主判据 A3 + 第二段 UH ≤ −1.5 cm 且 ≥ 4/5'), ('a3only', '次判据 A3 单独 ≤ −1.0 cm 且 ≥ 4/5'),
                    ('A0', 'A0 + UH（现做法）'), ('A1', 'A1 + UH'), ('A2', 'A2 + UH')):
        L.append('**%s：%s**（测试 %s / 验证 %s）' % (nm, '成立' if res['test'][key] and res['val'][key] else '不成立', '✓' if res['test'][key] else '✗', '✓' if res['val'][key] else '✗'))
    open(G + '/ANCHOR_RESULTS.md', 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
