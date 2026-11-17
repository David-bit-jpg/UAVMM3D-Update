# -*- coding: utf-8 -*-
"""最终对比（用户 10-12：只给最终的，用自己代码里的 metric）：release_20260924 README 同口径（eval_camnorm 摘要 + 仓库原版 LAA3D_ADS）。
python final_eval_ads2.py select  → 真实验证上按 PREREG_unc.md（sha 8685fb8e）选不确定性置信度的 τ → finalads/unc_select.json
python final_eval_ads2.py final   → 真实测试最终表 → G/FINAL_ADS.md + finalads/final_rows.json
第二段修正量（V2Rp_s5..9）与不确定性来自 unc_dump.py（st2plus_ens 上算，检测器自己的朝向，开发同口径），按帧套到带 KZ 朝向的记录上（先核对位置逐位一致）。"""
import json, os, pickle, re, sys
import numpy as np
ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
REPL = ROOT + '/output/camnorm/night1003/scratch/repl'
for p in (REPL, G, ROOT, ROOT + '/tools'):
    sys.path.insert(0, p)
os.environ.setdefault('MPLBACKEND', 'Agg')
import repl_common as RC          # noqa: E402
import kzst_common as C           # noqa: E402
KFO = C.KFO
from uavdet3d.utils import pose_eval as pe   # noqa: E402
import eval_camnorm as EC          # noqa: E402
import softkd_analyze as ska       # noqa: E402
from anchor_eval import deploy     # noqa: E402
from st2combo_analyze import causal_global   # noqa: E402

O = G + '/finalads'
LAM = 1.0
W, H = 512.0, 288.0
TAUS = [None, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]


def summ(recs, tag, profiles=('indoor', 'laa')):
    S = pe.summarize(pe.match_records(recs, seq='xyz', match='greedy', dist_thresh=1.0), n_frames=len(recs))
    rep = EC.ads_eval(recs, [(1920, 1080)] * len(recs), W, H, 'xyz', list(profiles), O + '/ads_final', tag)
    out = dict(det=100 * S['det_rate'], pos=S['pos_median'], lt20=100 * S['ACC_pos_0.2'], lt50=100 * S['ACC_pos_0.5'], ang=S['ang_median'], score=S['score'])
    for p in rep:
        out['ads_' + p] = float(re.search(r'LAA3D_ADS_\S+ \(%\) : ([0-9.]+)', rep[p]).group(1))
        out['ap3d_' + p] = float(re.search(r'detection 3D AP:\s*\n\s*AP_R40_eval_distance_\S+\(%\): ([0-9.eE+-]+)', rep[p]).group(1))
        out['ap2d_' + p] = float(re.search(r'detection 2D AP: ?\s*\n\s*AP_R40_eval_distance_\S+\(%\): ([0-9.eE+-]+)', rep[p]).group(1))
    return out


def factors(sp):
    """(seq, frame) → (xyz0, 位置倍数, 校正后深度 z, 对数深度总标准差 σ)"""
    U = {s: pickle.load(open('%s/unc_%s.pkl' % (O, s), 'rb')) for s in ('test', 'val')}
    pr = {s: [(u['d'], np.zeros_like(u['d'])) for u in U[s]['unc']] for s in U}
    other = 'val' if sp == 'test' else 'test'
    recs = pickle.load(open('%s/%s/ens_%s.pkl' % (G, os.environ.get('UAV_ENSDIR', 'st2plus_ens'), sp), 'rb'))['records']
    assert [(r['seq_id'], r['frame_id']) for r in recs] == U[sp]['keys']
    new = deploy(recs, pr[sp], [x * LAM for x in causal_global(recs, pr[sp], ska.cval(pr[other]))])
    sig = [np.sqrt(2 * np.exp(u['lb']) ** 2 + u['sd'] ** 2) for u in U[sp]['unc']]
    allsig = np.concatenate(sig)
    med = float(np.nanmedian(allsig))
    out = {}
    for r0, r1, s in zip(recs, new, sig):
        if len(r0['pred']):
            out[(r0['seq_id'], r0['frame_id'])] = (r0['pred'][:, :3].copy(), r1['pred'][:, 2] / r0['pred'][:, 2], r1['pred'][:, 2].copy(),
                                                   np.where(np.isfinite(s), s, med))
    return out


def apply(recs, fac, tau=None, oracle=False, alpha=0.0):
    out = []
    for r in recs:
        r = dict(r)
        k = (r['seq_id'], r['frame_id'])
        if len(r['pred']):
            xyz0, f, z, s = fac[k]
            assert r['pred'].shape[0] == xyz0.shape[0] and np.allclose(r['pred'][:, :3], xyz0, atol=1e-4), ('位置不一致', k)
            p = r['pred'].copy()
            p[:, :3] *= f[:, None]
            p[:, 3:6] *= (f ** alpha)[:, None]          # PREREG_size.md：尺寸 ×f^α
            r['pred'] = p
            if tau is not None:
                key = np.log(np.maximum(r['conf'], 1e-6)) - z * s / tau
                r['conf'] = 1.0 / (1.0 - key)          # 单调、不下溢
            if oracle:
                e = np.linalg.norm(p[:, None, :3] - r['gt'][None, :, :3], axis=2).min(1) if len(r['gt']) else np.full(len(p), 9.9)
                r['conf'] = 1.0 / (1.0 + e)
        out.append(r)
    return out


def select():
    fac = factors('val')
    newv = pickle.load(open(O + '/new_val.pkl', 'rb'))['records']
    res = {'newdet_check': summ(newv, 'val_newdet', ('indoor',))}   # 应等于 new_val.log 的 45.32
    print('核对', res['newdet_check']['ads_indoor'], flush=True)
    for t in TAUS:
        res[str(t)] = summ(apply(newv, fac, t), 'val_tau_%s' % t, ('indoor',))
        print('τ', t, json.dumps(res[str(t)], ensure_ascii=False), flush=True)
    res['oracle'] = summ(apply(newv, fac, oracle=True), 'val_oracle', ('indoor',))
    # 描述：z·σ 与真实位置误差的秩相关
    from scipy.stats import spearmanr
    a, b = [], []
    for r in apply(newv, fac):
        if len(r['pred']) and len(r['gt']):
            xyz0, f, z, s = fac[(r['seq_id'], r['frame_id'])]
            a.append(z[0] * s[0])
            b.append(np.linalg.norm(r['pred'][0, :3] - r['gt'][0, :3]))
    rho = float(spearmanr(a, b).correlation)
    best = max([t for t in TAUS], key=lambda t: res[str(t)]['ads_indoor'])
    json.dump(dict(res=res, rho=rho, best=best, prereg='8685fb8e'), open(O + '/unc_select.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('秩相关 z·σ vs 位置误差', rho, '选中 τ', best, flush=True)


def final():
    sel = json.load(open(O + '/unc_select.json', encoding='utf-8'))
    tau = sel['best']
    Pk, Pb = RC.tag_paths('st2p_test_kz'), RC.tag_paths('st2p_test_adabn')
    Dk, ik = C.load_npz_info(Pk['npz'])
    Db, ib = C.load_npz_info(Pb['npz'])
    assert float(ik['W']) == W and float(ik['H']) == H
    fac = factors('test')
    rows = {}
    rows['09-24 发布版'] = summ(pickle.load(open(O + '/base_test.pkl', 'rb'))['records'], 'base')
    newr = pickle.load(open(O + '/new_test.pkl', 'rb'))['records']
    rows['① 换去偏自训练检测器'] = summ(newr, 'newdet')
    rows['② ① + 第二段裁窗测距'] = summ(apply(newr, fac), 'final_kz')
    tau_dep = None
    if tau is not None:
        r3 = rows['③ ② + 不确定性置信度'] = summ(apply(newr, fac, tau), 'final_kz_unc')
        r2 = rows['② ① + 第二段裁窗测距']
        ok = r3['ads_indoor'] > r2['ads_indoor'] and r3['ap3d_indoor'] > r2['ap3d_indoor'] and r3['ads_laa'] > r2['ads_laa'] - 0.05
        print('预登记判据', '成立' if ok else '判负', flush=True)
        tau_dep = tau if ok else None
    alpha = json.load(open(O + '/size_select.json'))['best']
    alpha_dep = 0.0
    if alpha:
        ref = rows['③ ② + 不确定性置信度'] if tau_dep is not None else rows['② ① + 第二段裁窗测距']
        r4 = rows['③b + 尺寸随测距联动'] = summ(apply(newr, fac, tau_dep, alpha=alpha), 'final_kz_size')
        ok = r4['ads_indoor'] > ref['ads_indoor'] and r4['ads_laa'] > ref['ads_laa'] - 0.05
        print('尺寸联动判据', '成立' if ok else '判负', flush=True)
        alpha_dep = alpha if ok else 0.0
    frames, _ = RC.load_frames(Pk['pkl'], with_gt=False)
    keys, A = RC.compute_arms(Dk, ik, frames)
    keys_b, B = RC.compute_arms(Db, ib, frames, members=list(range(5)))
    assert keys_b == keys and np.array_equal(A['top'], B['top'])
    frames_gt, _ = RC.load_frames(Pk['pkl'], Pk['sealed'], with_gt=True)
    for nm, Rarm, tg in (('④ + 朝向五网融合（SA-TM）', A['SATM'], 'satm'), ('⑤ + 朝向网络真实域归一化（AdaBN，待定）', B['SATM'], 'bnsatm')):
        F = KFO.build_arm_frames(frames_gt, keys, Rarm, A['top'])
        recs = [{'gt': np.asarray(f['gt'], np.float64).reshape(-1, 9), 'pred': np.asarray(f['pred'], np.float64).reshape(-1, 9), 'conf': np.asarray(f['conf'], np.float64),
                 'K': f['K_in'], 'seq_id': f['seq'], 'frame_id': f['frame']} for f in F]
        rows[nm] = summ(apply(recs, fac, tau_dep, alpha=alpha_dep), 'final_' + tg)
    L = ['# 最终对比：MAV6D 真实测试（4800 帧），release_20260924 README 同口径（eval_camnorm + LAA3D_ADS）', '',
         '| 配置 | 检出 | 位置中位 | <20cm | <50cm | 角度中位 | score | 3D AP(室内) | 2D AP(室内) | ADS indoor | ADS laa |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for nm, r in rows.items():
        L.append('| %s | %.1f%% | %.3f m | %.1f%% | %.1f%% | %.1f° | %.4f | %.2f | %.2f | %.2f | %.2f |' % (
            nm, r['det'], r['pos'], r['lt20'], r['lt50'], r['ang'], r['score'], r['ap3d_indoor'], r['ap2d_indoor'], r['ads_indoor'], r['ads_laa']))
    L += ['', '核对：09-24 发布版应复现 README（95.0% / 0.243 m / 39.6% / 78.7% / 41.8° / 0.3323 / 50.54 / 75.73）。',
          '不确定性置信度 τ = %s（验证集按 ADS indoor 选，PREREG_unc.md sha 8685fb8e）；测试判据%s，④⑤ 用 τ = %s。' % (tau, '成立' if tau_dep is not None else '判负/未用', tau_dep),
          '尺寸联动 α = %s（验证选，PREREG_size.md sha f0ffb188），④⑤ 用 α = %s。' % (alpha, alpha_dep)]
    open(G + '/FINAL_ADS.md', 'w', encoding='utf-8').write('\n'.join(L))
    json.dump(rows, open(O + '/final_rows.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    {'select': select, 'final': final}[sys.argv[1]]()
