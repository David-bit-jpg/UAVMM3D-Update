import os
# -*- coding: utf-8 -*-
"""最终管线的第二段（V2Rp_s5..9 五网）逐检测输出：δ̂ 均值、对数 Laplace 尺度 log b 均值、五网 δ̂ 标准差。
在 st2plus_ens（检测器自己的朝向，与开发同口径）上算 → finalads/unc_{split}.pkl。用法：python unc_dump.py test val"""
import pickle
import sys

import numpy as np
import torch

G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G)
import xpaste_final as XF  # noqa: E402  (导入时设 ce.P/KW，与开发一致)

MEM = [('V2Rp', j) for j in range(5, 10)]


@torch.no_grad()
def deltas_full(recs, imgs, members):
    nets = []
    for a, j in members:
        n = XF.v2.make(a).cuda()
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
            c = (XF.pe.PROTO8 * g[3:6]) @ XF.R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            q = (K @ c.T).T
            if (q[:, 2] <= 1e-3).any():
                continue
            uv = q[:, :2] / q[:, 2:3]
            u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
            w = u1 - u0
            if not (6.0 <= w <= 170.0):
                continue
            views.append(XF.sw.view_img(XF.ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), XF.ce.KW * w), XF.S0, 0.0, 0.0, 1.0, w, False))
            scal.append([np.log(w / K[0, 0]), np.log(g[2])] + list(np.log(np.maximum(g[3:6], 1e-3))))
            items.append((ri, pi))
    scal = np.array(scal, np.float32)
    D, B = [], []
    for b in range(0, len(views), 256):
        x = XF.v2.skt.dt.prep(np.stack(views[b:b + 256]), None, False).cuda()
        s = torch.from_numpy(scal[b:b + 256]).cuda()
        ds, lbs = [], []
        for n in nets:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dh, lb, _ = n(x, s)
            ds.append(dh.float().cpu().numpy())
            lbs.append(lb.float().cpu().numpy())
        D.append(np.stack(ds, 1))
        B.append(np.stack(lbs, 1))
    D = np.concatenate(D) if D else np.zeros((0, len(nets)))
    B = np.concatenate(B) if B else np.zeros((0, len(nets)))
    out = [dict(d=np.full(len(r['pred']), np.nan), lb=np.full(len(r['pred']), np.nan), sd=np.full(len(r['pred']), np.nan)) for r in recs]
    for n, (ri, pi) in enumerate(items):
        out[ri]['d'][pi] = D[n].mean()
        out[ri]['lb'][pi] = B[n].mean()
        out[ri]['sd'][pi] = D[n].std()
    return out


if __name__ == '__main__':
    for sp in sys.argv[1:]:
        recs = pickle.load(open('%s/%s/ens_%s.pkl' % (G, os.environ.get('UAV_ENSDIR', 'st2plus_ens'), sp), 'rb'))['records']
        imgs = XF.cde.Images('real') if sp == 'test' else XF.dae.ImagesVal()
        u = deltas_full(recs, imgs, MEM)
        pickle.dump(dict(keys=[(r['seq_id'], r['frame_id']) for r in recs], xyz=[r['pred'][:, :3].copy() for r in recs], unc=u),
                    open('%s/finalads/unc_%s.pkl' % (G, sp), 'wb'))
        n = sum(np.isfinite(x['d']).sum() for x in u)
        print(sp, '检测数', sum(len(r['pred']) for r in recs), '有效', n, flush=True)
