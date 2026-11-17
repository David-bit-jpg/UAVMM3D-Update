# -*- coding: utf-8 -*-
"""11-09 精简实验：老师组减到 2 个、去掉纯彩色对照组时，伪朝向质量如何（只用 CPU）。
与 pseudo_build.py 逐行同法（同一份老师读数 teach/teach_train.npz、同一部署记录、同一 SA-TM 融合与 PnP），只把"出伪朝向的成员"和"互验规则"换成下列几种：
  full   ：伪朝向 = 红外+LiDAR×5 + 红外蒸馏×2 + LiDAR×2；通过 = 红外⇄LiDAR ≤10° 且 老师⇄纯彩色×5 ≤10°（现最终版，应复现 1088 帧）
  t9il   ：同上 9 个出伪朝向；通过 = 只看红外⇄LiDAR ≤10°（去掉纯彩色对照组）
  t6rgb2 ：伪朝向 = 红外+LiDAR×2（s0,s1）+ 红外蒸馏×2 + LiDAR×2；通过 = 红外⇄LiDAR 且 老师⇄纯彩色×2（s0,s1）
  t6il   ：同上 6 个；通过 = 只看红外⇄LiDAR
  t4il   ：伪朝向 = 红外蒸馏×2 + LiDAR×2；通过 = 只看红外⇄LiDAR
输出 pseudo/pseudo_train_<名>.npz（格式与 pseudo_train.npz 相同，无真值字段）；--diag 另读封存真值只做诊断 → pseudo/sets_diag.json。"""
import sys
sys.dont_write_bytecode = True
import os, json, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pseudo_build as PB   # noqa: E402  （同一套路径、工具函数与阈值）
C1, S2, KFO = PB.C1, PB.S2, PB.KFO
TH, SIG_W = PB.TH, PB.SIG_W

Z = dict(np.load(HERE + '/teach/teach_train.npz', allow_pickle=False))
J = json.load(open(HERE + '/teach/teach_train.json', encoding='utf-8'))
names = J['member_names']
ix = {n: i for i, n in enumerate(names)}
G = {'RGB5': ['RGB_s%d' % k for k in range(5)], 'RGB2': ['RGB_s0', 'RGB_s1'], 'IRL5': ['IRLiD_s%d' % k for k in range(5)], 'IRL2': ['IRLiD_s0', 'IRLiD_s1'],
     'IR': ['IR_s0', 'IR_s1'], 'LiD': ['LiD_s0', 'LiD_s1']}
G = {k: [ix[n] for n in v] for k, v in G.items()}
G['T9'] = G['IRL5'] + G['IR'] + G['LiD']
G['T6'] = G['IRL2'] + G['IR'] + G['LiD']
G['T4'] = G['IR'] + G['LiD']
CFGS = {'full': ('T9', 'RGB5'), 't9il': ('T9', None), 't6rgb2': ('T6', 'RGB2'), 't6il': ('T6', None), 't4il': ('T4', None)}
ks = float(J.get('KP_SCALE') or 64.0)
frames, _ = C1.load_dump_nogt('%s/tags/%s/real_ens5flip_%s.pkl' % (PB.REPRO, PB.TAG, PB.TAG))
keys = [(str(s), str(f)) for s, f in zip(Z['seq'], Z['frame'])]
rows = {c: [] for c in CFGS}
t0 = time.time()
for i, k in enumerate(keys):
    fr = frames[k]
    t = C1.hook_top(Z, i, fr)
    if t < 0:
        continue
    lwh = np.asarray(fr['pred'][t, 3:6], np.float64)
    ctx = C1.make_ctx(Z, i, lwh, ks)
    Rdep = KFO.rot_from_eul(fr['pred'][t, 6:9])
    out = {}
    for g in ('RGB5', 'RGB2', 'IR', 'LiD', 'T9', 'T6', 'T4'):
        mem = G[g]
        cell = ctx.cell_for(mem)
        o, fb, plan, (xs, Rs) = KFO.fuse_subset(ctx, mem, cell, ['C0', 'SATM'], Rdep)
        vec = KFO.satm_vec(xs, plan) if plan is not None else KFO.mean_vec(xs)
        out[g] = dict(R=o['SATM'], fb=bool(fb['SATM']), vec=vec)
    d_il = PB.sym(out['IR']['R'], out['LiD']['R'])
    x0, y0, w, h, s = [float(v) for v in Z['crop'][i]]
    Kc = S2.crop_K(Z['K_raw_src'][i], x0, y0, s)
    for c, (tg, rg) in CFGS.items():
        T = out[tg]
        rt = KFO.pnp(KFO.f32(T['vec']), ctx.g, return_rt=True)
        if rt is None or T['fb']:
            continue
        Rh, rv, tv = rt
        d_tr = PB.sym(Rh, out[rg]['R']) if rg else 0.0
        rows[c].append(dict(i=i, seq=k[0], frame=k[1], t_pos=np.asarray(fr['pred'][t, :3], np.float64), lwh=lwh, R=Rh, t_pnp=tv.reshape(3),
                            K_crop=Kc, d_il=d_il, d_tr=d_tr, conf=float(fr['conf'][t]), R_ir=out['IR']['R'], R_lid=out['LiD']['R'],
                            R_rgb=out[rg]['R'] if rg else Rh, R_irlid=out['T6']['R'] if tg != 'T4' else Rh))
    if i % 500 == 0:
        print('[sets] %d / %d  %.0f s' % (i, len(keys), time.time() - t0), flush=True)

info = {}
for c, rr in rows.items():
    d_il = np.array([r['d_il'] for r in rr]); d_tr = np.array([r['d_tr'] for r in rr])
    gate = (d_il <= TH) & (d_tr <= TH)
    wgt = np.exp(-(d_il ** 2 + d_tr ** 2) / (2 * SIG_W ** 2))
    eul = np.array([Rot.from_matrix(r['R']).as_euler('xyz') for r in rr])
    view = np.array([PB.view_dir(r['R'], r['t_pnp']) for r in rr])
    np.savez(HERE + '/pseudo/pseudo_train_%s.npz' % c, idx=np.array([r['i'] for r in rr]), seq=np.array([r['seq'] for r in rr]),
             frame=np.array([r['frame'] for r in rr]), K_crop=np.stack([r['K_crop'] for r in rr]), t_pnp=np.stack([r['t_pnp'] for r in rr]),
             t_pos=np.stack([r['t_pos'] for r in rr]), lwh=np.stack([r['lwh'] for r in rr]), R=np.stack([r['R'] for r in rr]), eul=eul,
             view=view, d_il=d_il, d_tr=d_tr, gate=gate, w=wgt, conf=np.array([r['conf'] for r in rr]),
             R_rgb=np.stack([r['R_rgb'] for r in rr]), R_ir=np.stack([r['R_ir'] for r in rr]), R_lid=np.stack([r['R_lid'] for r in rr]),
             R_irlid=np.stack([r['R_irlid'] for r in rr]))
    info[c] = dict(n_pseudo=len(rr), n_gate=int(gate.sum()), frac_gate=float(gate.mean()), teacher=CFGS[c][0], rgb_check=CFGS[c][1])
    print('[sets]', c, info[c], flush=True)
json.dump(info, open(HERE + '/pseudo/sets_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

if '--diag' in sys.argv:   # 只诊断：读封存真值
    import repl_common as RC
    fr_gt, _ = RC.load_frames('%s/tags/%s/real_ens5flip_%s.pkl' % (PB.REPRO, PB.TAG, PB.TAG), '%s/tags/%s/sealed_gt_%s.pkl' % (PB.REPRO, PB.TAG, PB.TAG), with_gt=True)
    res = {}
    for c, rr in rows.items():
        E, H, gi = [], [], []
        d_il = np.array([r['d_il'] for r in rr]); d_tr = np.array([r['d_tr'] for r in rr]); gate = (d_il <= TH) & (d_tr <= TH)
        for r, gg in zip(rr, gate):
            g = np.asarray(fr_gt[(r['seq'], r['frame'])]['gt'], np.float64).reshape(-1, 9)
            if not len(g):
                continue
            g = g[np.argmin(np.linalg.norm(g[:, :3] - r['t_pos'], axis=1))]
            if np.linalg.norm(g[:3] - r['t_pos']) > 1.0:
                continue
            Rg = Rot.from_euler('xyz', g[6:9]).as_matrix()
            E.append(PB.sym(r['R'], Rg)); H.append(PB.sym_head(r['R'], Rg)); gi.append(bool(gg))
        E, H, gi = np.array(E), np.array(H), np.array(gi)
        res[c] = dict(n_scored=len(E), err_all=float(E.mean()), err_pass=float(E[gi].mean()), n_pass_scored=int(gi.sum()), frac_pass=float(gi.mean()),
                      head_med_pass=float(np.median(H[gi])), head_med_all=float(np.median(H)))
        print('[diag]', c, res[c], flush=True)
    json.dump(res, open(HERE + '/pseudo/sets_diag.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
