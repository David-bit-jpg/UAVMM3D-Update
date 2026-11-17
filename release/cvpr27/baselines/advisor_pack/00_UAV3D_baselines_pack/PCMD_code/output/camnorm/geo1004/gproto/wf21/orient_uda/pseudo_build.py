# -*- coding: utf-8 -*-
"""目标域伪朝向（只用 CPU）：教师知识进伪标签 + 红外⇄LiDAR 互验 + 教师⇄纯彩色互验。规则在看任何真值之前写死（README §2）。
输入 teach/teach_train.npz（teach_dump.py：四组朝向网络在真实 train 部署裁窗上的 5×5 窗口热图 / 关键点）+ 部署记录（位置网络的框）。
每帧（顶部检测在钩子格）：
  组内融合：每组在自己的平均热图峰值格读关键点 → 部署解码器同款 PnP；组内 5（或 2）个成员在 180° 商空间对齐、截掉离群、平均关键点再 PnP（SA-TM，同部署）。
  伪朝向 R̂（带教师知识）：红外 / LiDAR 蒸馏过的 9 个成员（KZ-IRLiD×5 + KZ-KDir×2 + KZ-LiD×2）一起 SA-TM；
                       同时得到 PnP 平移 t̂（只用于把关键点目标写成刚体投影，位置不用它）。
  互验（四旋翼对称：转 90° 的倍数不算差）：
     d_IL = 对称测地(红外蒸馏组 R_IR, LiDAR 蒸馏组 R_LiD)      ← 红外⇄LiDAR 互验
     d_TR = 对称测地(R̂, 纯彩色组 R_RGB)                         ← 教师⇄纯彩色互验
  通过 = d_IL ≤ 10° 且 d_TR ≤ 10°（写死）；
  可信度权重 w = exp(−(d_IL² + d_TR²) / (2·10²))（写死，用于原型损失）。
输出 pseudo/pseudo_train.npz（无真值字段）；--diag 另外读封存真值写 pseudo/premise.json（只诊断，不进训练）。"""
import sys
sys.dont_write_bytecode = True
import os, json, pickle, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
SCR1003 = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR1003 + '/kzst1', SCR1003 + '/repl', SCR1003 + '/kzst2', '@@ROOT@@/tools', '@@ROOT@@'):
    if p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = '@@ROOT@@/output/camnorm/geo1004/gproto/wf20/repro'
TAG = 'wanb_train_kz'
import kzst_common as C1     # noqa: E402
import kzst2_common as S2    # noqa: E402
KFO = C1.KFO
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
TH = 10.0          # 互验阈值（写死）
SIG_W = 10.0       # 可信度权重尺度（写死）


def sym(A, B):
    return min(KFO.geodesic(A @ RZ[q], B) for q in range(4))


def sym_head(Rp, Rg):
    out = []
    for q in range(4):
        E = Rg.T @ Rp @ RZ[q]
        out.append(abs(np.degrees(np.arctan2(E[1, 0], E[0, 0]))))
    return min(out)


def tilt(Rp, Rg):
    return float(np.degrees(np.arccos(np.clip(Rp[:, 2] @ Rg[:, 2], -1, 1))))


def view_dir(R, t):
    """机体系里"从机体看相机"的方向 v = Rᵀ(−t/|t|)"""
    t = np.asarray(t, np.float64)
    return R.T @ (-t / max(np.linalg.norm(t), 1e-9))


def main():
    diag = '--diag' in sys.argv
    Z = dict(np.load(HERE + '/teach/teach_train.npz', allow_pickle=False))
    J = json.load(open(HERE + '/teach/teach_train.json', encoding='utf-8'))
    names = J['member_names']
    grp = {g: [names.index(n) for n in v] for g, v in J['groups'].items()}
    teacher = grp['IRLiD'] + grp['IR'] + grp['LiD']
    SFX = ''
    if '--rgb' in sys.argv:          # 消融（事后加）：伪朝向只用纯彩色组（没有任何红外 / LiDAR 知识），互验量照算但不用
        teacher, SFX, diag = list(grp['RGB']), '_rgb', False
    ks = float(J.get('KP_SCALE') or 64.0)
    frames, _ = C1.load_dump_nogt('%s/tags/%s/real_ens5flip_%s.pkl' % (REPRO, TAG, TAG))
    keys = [(str(s), str(f)) for s, f in zip(Z['seq'], Z['frame'])]
    N = len(keys)
    rows = []
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
        for g, mem in list(grp.items()) + [('T', teacher)]:
            cell = ctx.cell_for(mem)
            o, fb, plan, (xs, Rs) = KFO.fuse_subset(ctx, mem, cell, ['C0', 'SATM'], Rdep)
            vec = KFO.satm_vec(xs, plan) if plan is not None else KFO.mean_vec(xs)
            out[g] = dict(R=o['SATM'], fb=bool(fb['SATM']), cell=cell, vec=vec, Rs=Rs, stratum=plan['stratum'] if plan is not None else 'degenerate')
        T = out['T']
        rt = KFO.pnp(KFO.f32(T['vec']), ctx.g, return_rt=True)
        if rt is None or T['fb']:
            continue
        Rh, rv, tv = rt
        d_il = sym(out['IR']['R'], out['LiD']['R'])
        d_tr = sym(Rh, out['RGB']['R'])
        x0, y0, w, h, s = [float(v) for v in Z['crop'][i]]
        Kc = S2.crop_K(Z['K_raw_src'][i], x0, y0, s)
        rows.append(dict(i=i, seq=k[0], frame=k[1], top=t, t_pos=np.asarray(fr['pred'][t, :3], np.float64), lwh=lwh, R=Rh, t_pnp=tv.reshape(3),
                         K_crop=Kc, d_il=d_il, d_tr=d_tr, conf=float(fr['conf'][t]),
                         R_rgb=out['RGB']['R'], R_irlid=out['IRLiD']['R'], R_ir=out['IR']['R'], R_lid=out['LiD']['R'],
                         strat=T['stratum']))
        if i % 500 == 0:
            print('[pseudo] %d / %d  %.0f s' % (i, N, time.time() - t0), flush=True)
    n = len(rows)
    d_il = np.array([r['d_il'] for r in rows]); d_tr = np.array([r['d_tr'] for r in rows])
    gate = (d_il <= TH) & (d_tr <= TH)
    wgt = np.exp(-(d_il ** 2 + d_tr ** 2) / (2 * SIG_W ** 2))
    eul = np.array([Rot.from_matrix(r['R']).as_euler('xyz') for r in rows])
    view = np.array([view_dir(r['R'], r['t_pnp']) for r in rows])
    os.makedirs(HERE + '/pseudo', exist_ok=True)
    np.savez(HERE + '/pseudo/pseudo_train%s.npz' % SFX, idx=np.array([r['i'] for r in rows]), seq=np.array([r['seq'] for r in rows]),
             frame=np.array([r['frame'] for r in rows]), K_crop=np.stack([r['K_crop'] for r in rows]), t_pnp=np.stack([r['t_pnp'] for r in rows]),
             t_pos=np.stack([r['t_pos'] for r in rows]), lwh=np.stack([r['lwh'] for r in rows]), R=np.stack([r['R'] for r in rows]), eul=eul,
             view=view, d_il=d_il, d_tr=d_tr, gate=gate, w=wgt, conf=np.array([r['conf'] for r in rows]),
             R_rgb=np.stack([r['R_rgb'] for r in rows]), R_ir=np.stack([r['R_ir'] for r in rows]), R_lid=np.stack([r['R_lid'] for r in rows]),
             R_irlid=np.stack([r['R_irlid'] for r in rows]))
    info = dict(n_frames=N, n_pseudo=n, n_gate=int(gate.sum()), frac_gate=float(gate.mean()), th=TH, sig_w=SIG_W,
                frac_il=float((d_il <= TH).mean()), frac_tr=float((d_tr <= TH).mean()), teacher_members=[names[j] for j in teacher],
                written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(info, open(HERE + '/pseudo/pseudo_train%s.json' % SFX, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[pseudo]', info, flush=True)
    if not diag:
        return
    # ---------------- 只诊断：读封存真值，不写进训练文件
    import repl_common as RC
    fr_gt, _ = RC.load_frames('%s/tags/%s/real_ens5flip_%s.pkl' % (REPRO, TAG, TAG), '%s/tags/%s/sealed_gt_%s.pkl' % (REPRO, TAG, TAG), with_gt=True)
    E = {kk: [] for kk in ('T', 'RGB', 'IRLiD', 'IR', 'LiD')}
    H, TL, CLS, OK = [], [], [], []
    for r in rows:
        f = fr_gt[(r['seq'], r['frame'])]
        g = np.asarray(f['gt'], np.float64).reshape(-1, 9)
        if not len(g):
            OK.append(False); continue
        g = g[np.argmin(np.linalg.norm(g[:, :3] - r['t_pos'], axis=1))]
        if np.linalg.norm(g[:3] - r['t_pos']) > 1.0:
            OK.append(False); continue
        Rg = Rot.from_euler('xyz', g[6:9]).as_matrix()
        OK.append(True)
        for kk, RR in (('T', r['R']), ('RGB', r['R_rgb']), ('IRLiD', r['R_irlid']), ('IR', r['R_ir']), ('LiD', r['R_lid'])):
            E[kk].append(sym(RR, Rg))
        H.append(sym_head(r['R'], Rg)); TL.append(tilt(r['R'], Rg)); CLS.append(r['seq'].split('/')[0])
    OK = np.array(OK)
    E = {kk: np.array(v) for kk, v in E.items()}; H = np.array(H); TL = np.array(TL); CLS = np.array(CLS)
    gi, il, tr = gate[OK], d_il[OK], d_tr[OK]
    from scipy.stats import spearmanr
    res = dict(n_scored=int(OK.sum()), group_err_mean={kk: float(v.mean()) for kk, v in E.items()},
               group_err_median={kk: float(np.median(v)) for kk, v in E.items()},
               gate=dict(frac=float(gi.mean()), err_mean_pass=float(E['T'][gi].mean()), err_mean_fail=float(E['T'][~gi].mean()),
                         err_med_pass=float(np.median(E['T'][gi])), err_med_fail=float(np.median(E['T'][~gi])),
                         head_med_pass=float(np.median(H[gi])), head_med_fail=float(np.median(H[~gi])),
                         head_mean_pass=float(H[gi].mean()), head_mean_fail=float(H[~gi].mean()),
                         tilt_med_pass=float(np.median(TL[gi])), tilt_med_fail=float(np.median(TL[~gi])),
                         frac_p4=float(np.mean(CLS[gi] == 'phantom4')), frac_p4_all=float(np.mean(CLS == 'phantom4'))),
               only_il=dict(frac=float((il <= TH).mean()), err_pass=float(E['T'][il <= TH].mean()), err_fail=float(E['T'][il > TH].mean())),
               only_tr=dict(frac=float((tr <= TH).mean()), err_pass=float(E['T'][tr <= TH].mean()), err_fail=float(E['T'][tr > TH].mean())),
               spearman_il=float(spearmanr(il, E['T'])[0]), spearman_tr=float(spearmanr(tr, E['T'])[0]),
               spearman_w=float(spearmanr(wgt[OK], E['T'])[0]))
    for c in ('phantom4', 'mavic2'):
        s = CLS == c
        res['cls_' + c] = dict(n=int(s.sum()), err_all=float(E['T'][s].mean()), err_pass=float(E['T'][s & gi].mean()), frac_pass=float(gi[s].mean()),
                               head_med_all=float(np.median(H[s])), head_med_pass=float(np.median(H[s & gi])))
    json.dump(res, open(HERE + '/pseudo/premise.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
