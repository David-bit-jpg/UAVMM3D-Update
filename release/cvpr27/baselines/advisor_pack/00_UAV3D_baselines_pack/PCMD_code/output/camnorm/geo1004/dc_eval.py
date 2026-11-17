# -*- coding: utf-8 -*-
"""「第二段直接修正深度」（PREREG_dc.md）第 3 步：部署评测。检测网络 P0_s{k} 不动，改它已有的逐帧结果（仿真全量 / 真实标准口径）。
每个检测：投影预测 3D 框 → 外接框中心与宽 w（只在 w ∈ [6, 170] 像素时处理），与训练完全相同的取窗 → 128×128 彩色裁窗；
标量 = log(w / 焦距)、log(预测深度)、log(预测长宽高)（真实标准口径里长宽高是自标定的固定尺寸，与检测当时解深度用的一致）；
第二段（种子 j = k mod 3）给出 δ̂，位置沿视线缩放 p ← p × exp(δ̂)；倍数超出 [1/1.4, 1.4] 时保留原值。
    DCS：红外轮廓版；DC0：对照版。输出 dc/eval/{DCS,DC0}_{sim|real}_s{k}.pkl 与 dc/eval/summary.json。"""
import json
import os
import pickle
import sys
import time

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
import dc_train as dt  # noqa: E402

ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
GUARD = 1.4
OUT = G + '/dc/eval'


@torch.no_grad()
def process(recs, imgs, nets):
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
            patch = ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), ce.KW * w)
            views.append(sw.view_img(patch, S0, 0.0, 0.0, 1.0, w, False))
            scal.append([np.log(w / K[0, 0]), np.log(g[2])] + list(np.log(np.maximum(g[3:6], 1e-3))))
            items.append((ri, pi))
    scal = np.array(scal, np.float32)
    deltas = {a: [] for a in nets}
    for b in range(0, len(views), 256):
        x = dt.prep(np.stack(views[b:b + 256]), None, False).cuda()
        s = torch.from_numpy(scal[b:b + 256]).cuda()
        for a, net in nets.items():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dh, _ = net(x, s)
            deltas[a].append(dh.float().cpu().numpy())
    outs, st = {}, dict(n_det=sum(len(r['pred']) for r in recs), n_try=len(items))
    for a in nets:
        d = np.concatenate(deltas[a]) if deltas[a] else np.zeros(0)
        new = [dict(r, pred=r['pred'].copy()) for r in recs]
        ch = 0
        for n, (ri, pi) in enumerate(items):
            f = float(np.exp(d[n]))
            if 1.0 / GUARD <= f <= GUARD:
                new[ri]['pred'][pi, :3] *= f
                ch += 1
        outs[a] = new
        st[a + '_changed'] = ch
        st[a + '_delta_median'] = float(np.median(d)) if len(d) else float('nan')
    return outs, st


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = json.load(open(OUT + '/summary.json', encoding='utf-8')) if os.path.exists(OUT + '/summary.json') else {}
    srcs = {'sim': G + '/sim_full_recs/P0_s%d.pkl', 'real': G + '/irloc_recs/P0_s%d.pkl'}
    for which in ('real', 'sim'):
        imgs = cde.Images(which)
        for k in range(5):
            key = '%s_s%d' % (which, k)
            if key in summary:
                continue
            t0 = time.time()
            nets = {}
            for a in ('DCS', 'DC0'):
                n = dt.DCNet().cuda()
                n.load_state_dict(torch.load('%s/dc/%s_s%d.pt' % (G, a, k % 3), map_location='cuda'))
                nets[a] = n.eval()
            z = pickle.load(open(srcs[which] % k, 'rb'))
            outs, st = process(z['records'], imgs, nets)
            for a, recs in outs.items():
                with open('%s/%s_%s_s%d.pkl' % (OUT, a, which, k), 'wb') as f:
                    pickle.dump(dict(z, records=recs), f)
            summary[key] = dict(stats=st, P0=cde.summ(z['records']), **{a: cde.summ(outs[a]) for a in outs})
            print(key, json.dumps(summary[key], ensure_ascii=False), '%.0fs' % (time.time() - t0), flush=True)
            json.dump(summary, open(OUT + '/summary.json.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            os.replace(OUT + '/summary.json.tmp', OUT + '/summary.json')
            del nets
            torch.cuda.empty_cache()
    print('全部完成', flush=True)


if __name__ == '__main__':
    main()
