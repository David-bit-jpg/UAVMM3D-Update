# -*- coding: utf-8 -*-
"""PREREG_bgdepth 评测：现最终版（朝向 Q + 关键点高度模板，在线一致）里只换第二段测距：V2Rp（现）/ V2Mp（10-12 只有机身 LiDAR 点）/ V2Jp（加背景点）。
    python bg_eval.py unc V2Jp          → finalads/unc_{test,val}_v2j.pkl（st2plus_ens 上逐检测 δ̂ / σ）
    python bg_eval.py eval <val|test>"""
import json, os, pickle, sys
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
for p in (G + '/finalads', G + '/lidar', G + '/lidar/kpsize', G):
    sys.path.insert(0, p)
O = G + '/finalads'
TAGS = {'V2Mp': 'v2m', 'V2Jp': 'v2j'}


def unc(arm):
    import unc_dump as UD
    for sp in ('test', 'val'):
        recs = pickle.load(open('%s/%s/ens_%s.pkl' % (G, os.environ.get('UAV_ENSDIR', 'st2plus_ens'), sp), 'rb'))['records']
        imgs = UD.XF.cde.Images('real') if sp == 'test' else UD.XF.dae.ImagesVal()
        u = UD.deltas_full(recs, imgs, [(arm, j) for j in range(5, 10)])
        pickle.dump(dict(keys=[(r['seq_id'], r['frame_id']) for r in recs], xyz=[r['pred'][:, :3].copy() for r in recs], unc=u),
                    open('%s/unc_%s_%s.pkl' % (O, sp, TAGS[arm]), 'wb'))
        print(arm, sp, '完成', flush=True)


def factors(sp, tag):
    import final_eval_ads2 as F
    U = {s: pickle.load(open('%s/unc_%s_%s.pkl' % (O, s, tag), 'rb')) for s in ('test', 'val')}
    pr = {s: [(u['d'], np.zeros_like(u['d'])) for u in U[s]['unc']] for s in U}
    other = 'val' if sp == 'test' else 'test'
    recs = pickle.load(open('%s/%s/ens_%s.pkl' % (G, os.environ.get('UAV_ENSDIR', 'st2plus_ens'), sp), 'rb'))['records']
    assert [(r['seq_id'], r['frame_id']) for r in recs] == U[sp]['keys']
    new = F.deploy(recs, pr[sp], [x * F.LAM for x in F.causal_global(recs, pr[sp], F.ska.cval(pr[other]))])
    sig = [np.sqrt(2 * np.exp(u['lb']) ** 2 + u['sd'] ** 2) for u in U[sp]['unc']]
    med = float(np.nanmedian(np.concatenate(sig)))
    out = {}
    for r0, r1, s in zip(recs, new, sig):
        if len(r0['pred']):
            out[(r0['seq_id'], r0['frame_id'])] = (r0['pred'][:, :3].copy(), r1['pred'][:, 2] / r0['pred'][:, 2], r1['pred'][:, 2].copy(), np.where(np.isfinite(s), s, med))
    return out


def build(split, fac):
    """现最终版（release_20261015kps）：KZ-ST7Q 5 成员 + 关键点高度比（0.843×，片段内过去帧滚动中位）定 PnP 模板高度 + 第二段 fac"""
    import temporal_filter as T
    import kps_shape_chain as KC
    F, RC, C, KFO = T.F, T.RC, T.C, T.KFO
    rk = pickle.load(open(G + '/lidar/kpsize/shape_%s.pkl' % split, 'rb'))
    P = RC.tag_paths('st2p_%s_kzt7q' % split)
    D, info = C.load_npz_info(P['npz'])
    frames, _ = RC.load_frames(P['pkl'], with_gt=False)
    rr = KC.running(rk, list(frames.keys()), 'causal')
    for k, f in frames.items():
        p = np.asarray(f['pred'], np.float64).copy()
        if len(p) and k in rr:
            p[:, 5] = KC.A_MAP * rr[k] * p[:, 3:5].mean(1)
        f['pred'] = p
    keys, B = RC.compute_arms(D, info, frames, members=list(range(5)))
    frames_gt, _ = RC.load_frames(P['pkl'], P['sealed'], with_gt=True)
    Fr = KFO.build_arm_frames(frames_gt, keys, B['SATM'], B['top'])
    votes = T.member_votes(D, info, frames, keys, B['SATM'])
    recs = [{'gt': np.asarray(f['gt'], np.float64).reshape(-1, 9), 'pred': np.asarray(f['pred'], np.float64).reshape(-1, 9),
             'conf': np.asarray(f['conf'], np.float64), 'K': f['K_in'], 'seq_id': f['seq'], 'frame_id': f['frame']} for f in Fr]
    recs = F.apply(recs, fac if fac is not None else F.factors(split), T.TAU, alpha=T.ALPHA)
    for r, v in zip(recs, votes):
        r['votes'] = v
    return T.do(recs, 'hmm_causal_g3_w9'), T, F


def evaluate(sp):
    rows, D = {}, {}
    for name, tag in (('现最终版（第二段 V2Rp）', None), ('第二段 V2Mp（只有机身 LiDAR 点，10-12）', 'v2m'), ('第二段 V2Jp（加背景点，本实验）', 'v2j')):
        if tag and not os.path.exists('%s/unc_%s_%s.pkl' % (O, sp, tag)):
            continue
        r, T, F = build(sp, factors(sp, tag) if tag else None)
        rows[name] = F.summ(r, 'bgdepth_%s_%d' % (sp, len(rows)), ('indoor', 'laa'))
        pe, _, dz = T.errs(r)
        D[name] = {'depth_bad35': float(np.nanmean(np.abs(dz) > 0.3)), 'gt1m': float(np.nanmean(pe > 1.0))}
        x = rows[name]
        print('%s %s：ADS indoor %.2f laa %.2f 位置中位 %.3f 深度错>35%% %.1f%%' % (sp, name, x['ads_indoor'], x['ads_laa'], x['pos'], 100 * D[name]['depth_bad35']), flush=True)
    json.dump(dict(rows=rows, desc=D), open(G + '/lidar/bgdepth/eval_%s.json' % sp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    L = ['', '## 第二段测距加"相对背景深度"（LiDAR）（PREREG_bgdepth b4778586）· 真实%s · 现最终版链路只换第二段' % {'val': '验证', 'test': '测试'}[sp], '',
         '| 第二段 | 检出 | 位置中位 | <20cm | <50cm | 角度中位 | ADS indoor | ADS laa | 深度错>35% | >1 m |', '|---|---|---|---|---|---|---|---|---|---|']
    for k, x in rows.items():
        L.append('| %s | %.1f%% | %.3f m | %.1f%% | %.1f%% | %.1f° | %.2f | %.2f | %.1f%% | %.1f%% |' % (
            k, x['det'], x['pos'], x['lt20'], x['lt50'], x['ang'], x['ads_indoor'], x['ads_laa'], 100 * D[k]['depth_bad35'], 100 * D[k]['gt1m']))
    open(G + '/FINAL_ADS.md', 'a', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'unc':
        unc(sys.argv[2])
    else:
        evaluate(sys.argv[2])
