# -*- coding: utf-8 -*-
"""用户 10-19：「告诉我最新最好的模型结构，以及和不用蒸馏的、只用红外的、都用的对比」。同一条管线逐项开关蒸馏，用户口径评测。
    unc <arm> <tag>    第二段测距网络（种子 5–9）在真实 val/test 上推理 → finalads/unc_<split>_<tag>.pkl（GPU）
    eval <split>       6 行对照（CPU）"""
import json, os, pickle, sys
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
for p in (G + '/finalads', G + '/lidar', G + '/lidar/kpsize', G + '/lidar/bgdepth', G):
    sys.path.insert(0, p)
O = G + '/finalads'


def unc(arm, tag):
    import unc_dump as UD
    for sp in ('test', 'val'):
        recs = pickle.load(open('%s/%s/ens_%s.pkl' % (G, os.environ.get('UAV_ENSDIR', 'st2plus_ens'), sp), 'rb'))['records']
        imgs = UD.XF.cde.Images('real') if sp == 'test' else UD.XF.dae.ImagesVal()
        u = UD.deltas_full(recs, imgs, [(arm, j) for j in range(5, 10)])
        pickle.dump(dict(keys=[(r['seq_id'], r['frame_id']) for r in recs], xyz=[r['pred'][:, :3].copy() for r in recs], unc=u), open('%s/unc_%s_%s.pkl' % (O, sp, tag), 'wb'))
        print(arm, sp, '完成', flush=True)


def build(split, orient, fac_tag, height):
    """orient：朝向 5 成员的推理记录（kzt4 = 不用辅助答案；kzt6 = 红外+LiDAR 辅助答案；kzt7q = 同 kzt6 但自训练不教前后）
    fac_tag：None = 部署的测距（用红外剪影 + 真实伪标签那一步）；'v2q' / 'v2p' = 同配方的 不用 / 用 红外剪影
    height：'fixed' = 固定框；'kps' = LiDAR 关键点高度比 × 平均(长, 宽)（朝向模板 + 输出框，系数 1.0）"""
    import temporal_filter as T
    import kps_shape_chain as KC
    import bg_eval as BE
    F, RC, C, KFO = T.F, T.RC, T.C, T.KFO
    rk = pickle.load(open(G + '/lidar/kpsize/shape_%s%s.pkl' % (split, os.environ.get('UAV_SHAPESFX', '')), 'rb')) if height == 'kps' else None   # 11-06 固定框时不读 LiDAR 形状
    P = RC.tag_paths('%s_%s_%s' % (os.environ.get('UAV_POSTAG', 'st2p'), split, orient))
    D, info = C.load_npz_info(P['npz'])
    frames, _ = RC.load_frames(P['pkl'], with_gt=False)
    if height == 'kps':
        rr = KC.running(rk, list(frames.keys()), 'causal')
        for k, f in frames.items():
            p = np.asarray(f['pred'], np.float64).copy()
            if len(p) and k in rr:
                p[:, 5] = rr[k] * p[:, 3:5].mean(1)
            f['pred'] = p
    keys, B = RC.compute_arms(D, info, frames, members=list(range(5)))
    frames_gt, _ = RC.load_frames(P['pkl'], P['sealed'], with_gt=True)
    Fr = KFO.build_arm_frames(frames_gt, keys, B['SATM'], B['top'])
    votes = T.member_votes(D, info, frames, keys, B['SATM'])
    recs = [{'gt': np.asarray(f['gt'], np.float64).reshape(-1, 9), 'pred': np.asarray(f['pred'], np.float64).reshape(-1, 9),
             'conf': np.asarray(f['conf'], np.float64), 'K': f['K_in'], 'seq_id': f['seq'], 'frame_id': f['frame']} for f in Fr]
    recs = F.apply(recs, BE.factors(split, fac_tag) if fac_tag else F.factors(split), T.TAU, alpha=T.ALPHA)
    for r, v in zip(recs, votes):
        r['votes'] = v
    recs = T.do(recs, 'hmm_causal_g3_w9')
    if height == 'kps':
        rr2 = KC.running(rk, [(str(r['seq_id']), str(r['frame_id'])) for r in recs if (str(r['seq_id']), str(r['frame_id'])) in rk], 'causal')
        for r in recs:
            k = (str(r['seq_id']), str(r['frame_id']))
            if len(r['pred']) and k in rr2:
                r['pred'] = r['pred'].copy()
                r['pred'][0, 5] = rr2[k] * r['pred'][0, 3:5].mean()
    return recs, T, F


ROWS = [('①', '完全不用蒸馏', 'kzt4', 'v2q', 'fixed'),
        ('②', '只用红外：测距用红外剪影（与①同配方）', 'kzt4', 'v2p', 'fixed'),
        ('②′', '只用红外：测距用红外剪影（部署配方，多一步真实伪标签）', 'kzt4', None, 'fixed'),
        ('③′', '红外 + LiDAR：再加朝向的红外+LiDAR 辅助答案', 'kzt6', None, 'fixed'),
        ('③', '红外 + LiDAR：再加 LiDAR 关键点定高度', 'kzt6', None, 'kps'),
        ('④', '现最终版：③ + 朝向自训练不教前后', 'kzt7q', None, 'kps')]


def evaluate(sp):
    out = {}
    for code, name, orient, fac, height in ROWS:
        recs, T, F = build(sp, orient, fac, height)
        s = F.summ(recs, 'ablation_%s_%s' % (sp, code), ('indoor', 'laa'))
        pe, ang, dz = T.errs(recs)
        cls = np.array([r['seq_id'].split('/')[0] for r in recs])
        P_ = np.array([r['pred'][0, 3:6] if len(r['pred']) else [np.nan] * 3 for r in recs]); Gt = np.array([r['gt'][0, 3:6] for r in recs])
        s.update(name=name, flip90=float(np.nanmean(ang > 90)), lt20=float(np.nanmean(pe < 0.2)), lt50=float(np.nanmean(pe < 0.5)),
                 hgt_err={c: float(100 * np.nanmedian(np.abs(P_[cls == c, 2] - Gt[cls == c, 2]))) for c in ('mavic2', 'phantom4')})
        out[code] = s
        print('%s %s %s：ADS indoor %.2f laa %.2f | 位置中位 %.3f m <20cm %.1f%% | 角度中位 %.1f° 机头反 %.1f%% | 高度误差 mavic2 %.1f / phantom4 %.1f cm' % (
            sp, code, name, s['ads_indoor'], s['ads_laa'], s['pos'], 100 * s['lt20'], s['ang'], 100 * s['flip90'], s['hgt_err']['mavic2'], s['hgt_err']['phantom4']), flush=True)
    json.dump(out, open(G + '/ablation/ablation_%s.json' % sp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'unc':
        unc(sys.argv[2], sys.argv[3])
    else:
        evaluate(sys.argv[2])
