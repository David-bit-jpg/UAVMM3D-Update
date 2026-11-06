# -*- coding: utf-8 -*-
"""PREREG_final.md（sha256 141c8496）判定 → FINAL_RESULTS.md / final/analysis.json。
第一段 = ST2 五模型集成 + 翻转（final/ens_{split}.pkl）；第二段 = V2Cp_s2/s3/s4 平均 δ̂；整体尺度 = 因果全局滚动中位。"""
import json
import pickle
import sys

import numpy as np
import torch

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
from scipy.spatial.transform import Rotation as R  # noqa: E402
from uavdet3d.utils import pose_eval as pe  # noqa: E402
import cropw_extract as ce  # noqa: E402
import silw_train as sw  # noqa: E402
import cropdist_eval as cde  # noqa: E402
import dcalign_eval as dae  # noqa: E402
import secondstage_v2 as v2  # noqa: E402
import softkd_analyze as ska  # noqa: E402
from anchor_eval import deploy  # noqa: E402
from cropkd_analyze import causal_global  # noqa: E402

ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
GUARD = float(np.log(1.4))
NAME = {'test': '真实测试', 'val': '真实验证'}


@torch.no_grad()
def deltas(recs, imgs):
    nets = []
    for j in (2, 3, 4):
        n = v2.make('V2Cp').cuda()
        n.load_state_dict(torch.load('%s/softkd/V2Cp_s%d.pt' % (G, j), map_location='cuda'))
        nets.append(n.eval())
    items, views, scal = [], [], []
    for ri, r in enumerate(recs):
        if len(r['pred']) == 0:
            continue
        img, K, _ = imgs.get(r['seq_id'], r['frame_id'])
        for pi, g in enumerate(r['pred']):
            if g[2] <= 1e-3:
                continue
            c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            q = (K @ c.T).T
            if (q[:, 2] <= 1e-3).any():
                continue
            uv = q[:, :2] / q[:, 2:3]
            u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
            w = u1 - u0
            if not (6.0 <= w <= 170.0):
                continue
            views.append(sw.view_img(ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), ce.KW * w), S0, 0.0, 0.0, 1.0, w, False))
            scal.append([np.log(w / K[0, 0]), np.log(g[2])] + list(np.log(np.maximum(g[3:6], 1e-3))))
            items.append((ri, pi))
    scal = np.array(scal, np.float32)
    out = []
    for b in range(0, len(views), 256):
        x = v2.skt.dt.prep(np.stack(views[b:b + 256]), None, False).cuda()
        s = torch.from_numpy(scal[b:b + 256]).cuda()
        ds = []
        for n in nets:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dh, _, _ = n(x, s)
            ds.append(dh.float().cpu().numpy())
        out.append(np.mean(ds, 0))
    d = np.concatenate(out) if out else np.zeros(0)
    pr = [(np.full(len(r['pred']), np.nan), np.zeros(len(r['pred']))) for r in recs]
    for n, (ri, pi) in enumerate(items):
        pr[ri][0][pi] = d[n]
    return pr


def frame_rows(recs):
    out = {}
    for r in recs:
        if len(r['gt']) and len(r['pred']):
            j = int(np.argmax(r['conf']))
            e = np.linalg.norm(r['pred'][j, :3] - r['gt'][0, :3])
            out[(r['seq_id'], r['frame_id'])] = (e < 1.0, e)
    return out


def boot(base, new, n=2000):
    """按飞行片段（按机型分层）重抽样：位置中位（配上的帧）新 − 旧 的 95% 区间"""
    seqs = sorted(set(k[0] for k in base))
    by = {s: [k for k in base if k[0] == s] for s in seqs}
    strata = {}
    for s in seqs:
        strata.setdefault(s.split('/')[0], []).append(s)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        pick = []
        for _, ss in strata.items():
            pick += list(rng.choice(ss, size=len(ss), replace=True))
        ks = [k for s in pick for k in by[s]]
        b = [base[k][1] for k in ks if base[k][0]]
        w = [new[k][1] for k in ks if k in new and new[k][0]]
        vals.append(100 * (np.median(w) - np.median(b)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    RC = {sp: pickle.load(open('%s/final/ens_%s.pkl' % (G, sp), 'rb'))['records'] for sp in ('test', 'val')}
    PR = {sp: deltas(RC[sp], dae.ImagesVal() if sp == 'val' else cde.Images('real')) for sp in ('test', 'val')}
    res, L = {}, ['# 最终部署管线（PREREG_final.md，sha256 141c8496）', '',
                  '第一段 = ST2 五模型集成 + 翻转；第二段 = 红外蒸馏第二段 C+（V2Cp_s2/s3/s4）平均修正量；整体尺度 = 因果全局滚动中位（只用过去的帧、不用标签）。', '']
    for sp, other in (('test', 'val'), ('val', 'test')):
        recs, pr = RC[sp], PR[sp]
        c_other = ska.cval(PR[other])
        variants = {'集成单独': recs, '+ 第二段（因果全局滚动中位）': deploy(recs, pr, causal_global(recs, pr, c_other)),
                    '+ 第二段（另一集合中位）': deploy(recs, pr, c_other), '+ 第二段（本集合中位，非因果）': deploy(recs, pr, ska.cval(pr))}
        L += ['## ' + NAME[sp], '', '| 管线 | 检出 % | 位置中位 cm | 扣全局倍数逐帧位置 cm | mavic2 位置中位 | phantom4 位置中位 |', '|---|---|---|---|---|---|']
        rows = {k: ska.top1(v) for k, v in variants.items()}
        keys = sorted(set.intersection(*[set(v) for v in rows.values()]))
        r = {}
        for nm, rr in variants.items():
            det, pos = ska.summ(rr)
            fr = frame_rows(rr)
            tm = {t: 100 * np.median([e for k, (m, e) in fr.items() if m and k[0].startswith(t)]) for t in ('mavic2', 'phantom4')}
            deb = ska.debiased(rows[nm], keys)
            r[nm] = dict(det=det, pos=pos, deb=deb, **tm)
            L.append('| %s | %.1f | %.1f | %.1f | %.1f | %.1f |' % (nm, det, pos, deb, tm['mavic2'], tm['phantom4']))
        ci = boot(frame_rows(recs), frame_rows(variants['+ 第二段（因果全局滚动中位）']))
        d = r['+ 第二段（因果全局滚动中位）']['pos'] - r['集成单独']['pos']
        dd = r['+ 第二段（因果全局滚动中位）']['det'] - r['集成单独']['det']
        ok = bool(d <= -1.5 and ci[1] < 0 and dd >= -1.0)
        L += ['', '因果全局滚动中位部署 − 集成单独：%+.2f cm，按飞行片段 95%% 区间 [%+.2f, %+.2f]，检出 %+.2f 点 → %s' % (d, ci[0], ci[1], dd, '成立' if ok else '不成立'), '']
        res[sp] = dict(rows=r, diff=d, ci=ci, ddet=dd, ok=ok)
    both = res['test']['ok'] and res['val']['ok']
    L += ['## 判定', '', '**主判据（两集合都：位置中位降 ≥ 1.5 cm 且飞行片段区间上界 < 0，检出降 ≤ 1 点）：%s**' % ('成立' if both else '不成立'), '',
          '对照：09-24 发布（5 集成 + 翻转 + KZ 旋转）真实测试位置中位 24.3 cm（README）。']
    open(G + '/FINAL_RESULTS.md', 'w', encoding='utf-8').write('\n'.join(L))
    json.dump(res, open(G + '/final/analysis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
