# -*- coding: utf-8 -*-
"""PREREG_xpaste.md（sha256 f82012af）描述性：最终管线（ST2 五集成 + 翻转，final/ens_{split}.pkl）+ 第二段 5 网平均 δ̂，
A+ ×5 / P+ ×5（贴图）/ Q+ ×5（贴图去红外），因果全局滚动中位 → XPASTE_FINAL.md / xpaste/final.json。"""
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
from final_eval import frame_rows  # noqa: E402

ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
NAME = {'test': '真实测试', 'val': '真实验证'}
SETS = {'A+ ×5（红外，不贴图）': [('V2Ap', j) for j in range(5)], 'P+ ×5（红外 + 贴图）': [('V2Pp', j) for j in range(5)], 'Q+ ×5（贴图，去红外）': [('V2Qp', j) for j in range(5)]}


@torch.no_grad()
def deltas(recs, imgs, members):
    nets = []
    for a, j in members:
        n = v2.make(a).cuda()
        n.load_state_dict(torch.load('%s/softkd/%s_s%d.pt' % (G, a, j), map_location='cuda'))
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


def boot(base, new, n=2000):
    """按飞行片段（按机型分层）重抽样：配上的帧位置中位 new − base 的 95% 区间；键 = (集合, 片段, 帧)"""
    seqs = sorted(set(k[:2] for k in base))
    by = {s: [k for k in base if k[:2] == s] for s in seqs}
    strata = {}
    for s in seqs:
        strata.setdefault(s[1].split('/')[0], []).append(s)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        pick = []
        for _, ss in strata.items():
            pick += [ss[i] for i in rng.integers(0, len(ss), len(ss))]
        ks = [k for s in pick for k in by[s]]
        b = [base[k][1] for k in ks if base[k][0]]
        w = [new[k][1] for k in ks if k in new and new[k][0]]
        vals.append(100 * (np.median(w) - np.median(b)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    RC = {sp: pickle.load(open('%s/final/ens_%s.pkl' % (G, sp), 'rb'))['records'] for sp in ('test', 'val')}
    IM = {'test': cde.Images('real'), 'val': dae.ImagesVal()}
    PR = {nm: {sp: deltas(RC[sp], IM[sp], mem) for sp in ('test', 'val')} for nm, mem in SETS.items()}
    L = ['# 最终管线 + 跨域贴图（PREREG_xpaste.md 描述性，sha256 f82012af）', '',
         '第一段 = ST2 五集成 + 翻转；第二段 = 5 个网络平均修正量；整体尺度 = 因果全局滚动中位。', '',
         '| 集合 | 管线 | 检出 % | 位置中位 cm | mavic2 | phantom4 |', '|---|---|---|---|---|---|']
    FR, res = {}, {}
    for sp, other in (('test', 'val'), ('val', 'test')):
        recs = RC[sp]
        variants = {'集成单独': recs}
        for nm in SETS:
            variants[nm] = deploy(recs, PR[nm][sp], causal_global(recs, PR[nm][sp], ska.cval(PR[nm][other])))
        for nm, rr in variants.items():
            det, pos = ska.summ(rr)
            fr = frame_rows(rr)
            FR.setdefault(nm, {}).update({(sp,) + k: v for k, v in fr.items()})
            tm = {t: 100 * np.median([e for k, (m, e) in fr.items() if m and k[0].startswith(t)]) for t in ('mavic2', 'phantom4')}
            res.setdefault(sp, {})[nm] = dict(det=det, pos=pos, **tm)
            L.append('| %s | %s | %.1f | %.1f | %.1f | %.1f |' % (NAME[sp], nm, det, pos, tm['mavic2'], tm['phantom4']))
    a, p, q = list(SETS)
    L.append('')
    sub = lambda nm, sp: {k: v for k, v in FR[nm].items() if k[0] == sp}
    for x, y, lab in ((p, a, 'P+ − A+（贴图的作用）'), (p, q, 'P+ − Q+（贴图后红外的作用）')):
        for sp in ('test', 'val'):
            ci = boot(sub(y, sp), sub(x, sp))
            d = res[sp][x]['pos'] - res[sp][y]['pos']
            res[sp][lab] = dict(d=d, ci=ci)
            L.append('%s %s = %+.2f cm，飞行片段区间 [%+.2f, %+.2f]' % (NAME[sp], lab, d, ci[0], ci[1]))
        med = lambda nm: 100 * np.median([e for (m, e) in FR[nm].values() if m])
        ci = boot(FR[y], FR[x])
        res['pooled_' + lab] = dict(d=med(x) - med(y), ci=ci)
        L.append('两集合合并（26 个飞行片段）%s = %+.2f cm [%+.2f, %+.2f]' % (lab, med(x) - med(y), ci[0], ci[1]))
    open(G + '/XPASTE_FINAL.md', 'w', encoding='utf-8').write('\n'.join(L))
    json.dump(res, open(G + '/xpaste/final.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
