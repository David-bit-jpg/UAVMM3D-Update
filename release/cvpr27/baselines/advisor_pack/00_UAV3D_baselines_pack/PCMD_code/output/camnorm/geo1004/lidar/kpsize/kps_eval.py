# -*- coding: utf-8 -*-
"""KZ-KPS 阶段 A 评测（PREREG_kps 908f298a）。
    adabn <seed>            真实 train 裁窗重估 BN（与 adabn_kzm 同配方）→ kpsize/adabn/kps_s<seed>_adabn.pth
    pilot <split> <seed>    真实部署裁窗上：(a) 原 8 角点 PnP（部署解码）(b) 结构关键点 + 预测形状 PnP (c) 结构关键点 + 固定 phantom4 形状 (d) 参照：原 KZ-IRLiD+AdaBN 种子 0 的角点 PnP
                            → 角度 / 象限准确率 / 按机型 + 预测形状比例与尺寸 → kpsize/pilot_<split>_s<seed>.json，表追加 FINAL_ADS.md"""
import json, os, pickle, sys, time
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
SCR = ROOT + '/output/camnorm/night1003/scratch'
K_ = G + '/lidar/kpsize'
sys.path.insert(0, G + '/lidar/kzbody')
sys.path.insert(0, K_)
import kzbody_eval as E          # noqa: E402  （build / AdaBN 配方与 KZ-Body 相同）
E.CFG = 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_irlid_kps.yaml'
E.CK = ROOT + '/output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_irlid_kps/KPSs%d/ckpt/checkpoint_epoch_8.pth'
E.ADA = K_ + '/adabn/kps_s%d_adabn.pth'
E.B = K_
RZ = E.RZ
T = np.load(K_ + '/templates.npy', allow_pickle=True).item()
PH = T['DJI-phantom4']['kp8'] / float(np.max(T['DJI-phantom4']['lidar_ext']))
TRUE = {'mavic2': np.array([0.284, 0.333, 0.12]), 'phantom4': np.array([0.34, 0.34, 0.23])}


def register():
    import uavdet3d.model.detectors as dets
    from center_det_mm import CenterDetMM
    from center_det_mmk import CenterDetMMK
    dets.__all__['CenterDetMM'] = CenterDetMM
    dets.__all__['CenterDetMMK'] = CenterDetMMK


def adabn(seed):
    register()
    os.makedirs(K_ + '/adabn', exist_ok=True)
    E.adabn(seed)


def geo(R, Gm):
    return float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Gm) - 1) / 2, -1, 1))))


def pnp(obj, img, K, dist=None):
    if not np.isfinite(img).all() or not np.isfinite(obj).all():
        return None, None
    try:
        ok, rv, tv = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None, None
        ok, rv, tv = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), K, dist, rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tv[2, 0] <= 1e-6:
            return None, None
    except cv2.error:
        return None, None
    return cv2.Rodrigues(rv)[0], tv.reshape(3)


def pilot(split, seed):
    import torch
    register()
    import adabn_kz as AB
    C = AB.C
    KFO = C.KFO
    sys.path.insert(0, SCR + '/kzst2')
    import kzst2_common as S2
    import repl_common as RC
    dev = torch.device('cuda')
    P = RC.tag_paths('st2p_%s_kzt7q' % split)
    D, info = C.load_npz_info(P['npz'])
    frames, _ = RC.load_frames(P['pkl'], with_gt=False)
    nm_ = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns_ = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    in_wh = tuple(int(v) for v in info['in_wh'])
    ks = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    R = pickle.load(open(G + '/kzst7/kzt7q_%s_recs_v.pkl' % split, 'rb'))
    pos = {(str(r['seq_id']), str(r['frame_id'])): n for n, r in enumerate(R)}
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    idx = [i for i, k in enumerate(keys) if k in pos and len(R[pos[k]]['pred']) and C.hook_top(D, i, frames[k]) >= 0]
    newm = E.build(E.ADA % seed, torch, dev)
    refm = AB.build_models(torch, dev, [SCR + '/kzst1/adabn_kzm/kz_s%d_adabn_kzm.pth' % seed])[0][0]
    src = AB.NpzSource(D)
    rows = []
    t0 = time.time()
    for c0 in range(0, len(idx), 16):
        ch = idx[c0:c0 + 16]
        x = np.stack([AB.to_input(AB.recut(D, i, src, in_wh), nm_, ns_) for i in ch])
        with torch.no_grad():
            outs = []
            for mdl, kps in ((newm, True), (refm, False)):
                tb = {'image': torch.from_numpy(np.ascontiguousarray(x[:, None])).to(dev), 'batch_size': len(ch)}
                for mod in mdl.module_list:
                    tb = mod(tb)
                kp = tb['pred_center_dict']['kp2d'].float().cpu().numpy()
                kk = None
                if kps:
                    f = tb['features_2d']
                    kk = mdl.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:])).float().cpu().numpy()
                outs.append((kp, kk))
        for b, i in enumerate(ch):
            k = keys[i]
            r = R[pos[k]]
            fr = frames[k]
            t = C.hook_top(D, i, fr)
            rb, cb = (int(v) for v in D['rb_cb'][i])
            g_ = KFO.frame_geo(D, i, fr['pred'][t, 3:6], ks)
            Gm = Rot.from_euler('xyz', r['gt'][0, 6:9]).as_matrix()
            row = {'cls': k[0].split('/')[0], 'seq': k[0]}
            for name, (kp, _) in (('a_corner', outs[0]), ('d_ref', outs[1])):
                Rm = KFO.pnp(KFO.f32(kp[b, :, rb, cb].astype(np.float64) / float(D['z'][i])), g_)
                row[name] = Rm
            x0, y0, _, _, s = [float(v) for v in D['crop'][i]]
            Kc = S2.crop_K(D['K_raw_src'][i], x0, y0, s)
            tp = np.asarray(r['pred'][0, :3], np.float64)
            q = Kc @ tp
            uvc = q[:2] / q[2]
            kk = outs[0][1][b, :, rb, cb].astype(np.float64)
            img = uvc[None] + 64.0 * kk[:16].reshape(8, 2)
            lay = kk[16:].reshape(8, 3)
            Rb, tv = pnp(lay, img, Kc)
            row['b_struct'] = Rb
            Rc, _ = pnp(PH, img, Kc)
            row['c_fixed'] = Rc
            if tv is not None:
                Lm = tp[2] / tv[2]
                ext = np.ptp(lay, 0) * Lm
                row['size'] = ext
                row['ratio'] = (ext[0] / ext[1], ext[2] / ext[:2].mean())
            for nm in ('a_corner', 'b_struct', 'c_fixed', 'd_ref'):
                Rm = row[nm]
                if Rm is None:
                    row[nm] = None
                    continue
                a = [geo(Rm @ RZ[qq], Gm) for qq in range(4)]
                row[nm] = (a[0], int(np.argmin(a)))
            rows.append(row)
        if c0 % 640 == 0:
            print('  %d / %d 帧 %.0f s' % (c0 + len(ch), len(idx), time.time() - t0), flush=True)
    cls = np.array([r['cls'] for r in rows])
    res = {'n': len(rows)}
    NAME = {'a_corner': '(a) 原 8 角点 PnP（同一新网络）', 'b_struct': '(b) LiDAR 结构关键点 + 预测形状 PnP', 'c_fixed': '(c) 结构关键点 + 固定 phantom4 形状', 'd_ref': '(d) 参照：原 KZ-IRLiD+AdaBN 角点 PnP'}
    L = ['', '## LiDAR 结构关键点试点（PREREG_kps 908f298a）· 真实%s · 种子 %d · 单网络逐帧（无集成/自训练/时间滤波）' % ({'val': '验证', 'test': '测试'}[split], seed), '',
         '| 解朝向方式 | 有效 | 角度中位 | <15° | <30° | 象限准确率 | 反向(>90°) | mavic2 角度中位 | phantom4 角度中位 |', '|---|---|---|---|---|---|---|---|---|']
    for nm in ('a_corner', 'b_struct', 'c_fixed', 'd_ref'):
        ok = np.array([r[nm] is not None for r in rows])
        a = np.array([r[nm][0] if r[nm] is not None else np.nan for r in rows])
        qd = np.array([r[nm][1] if r[nm] is not None else -1 for r in rows])
        d = dict(valid=float(ok.mean()), ang=float(np.nanmedian(a)), lt15=float(np.nanmean(a < 15)), lt30=float(np.nanmean(a < 30)),
                 quad=float(np.mean(qd[ok] == 0)), flip=float(np.nanmean(a > 90)),
                 ang_mavic2=float(np.nanmedian(a[cls == 'mavic2'])), ang_phantom4=float(np.nanmedian(a[cls == 'phantom4'])))
        res[nm] = d
        L.append('| %s | %.1f%% | %.1f° | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f° | %.1f° |' % (NAME[nm], 100 * d['valid'], d['ang'], 100 * d['lt15'], 100 * d['lt30'], 100 * d['quad'], 100 * d['flip'], d['ang_mavic2'], d['ang_phantom4']))
    for c in ('mavic2', 'phantom4'):
        sz = np.array([r['size'] for r in rows if r['cls'] == c and 'size' in r])
        rt = np.array([r['ratio'] for r in rows if r['cls'] == c and 'ratio' in r])
        if len(sz):
            res['shape_' + c] = dict(size_med=np.median(sz, 0).tolist(), lw=float(np.median(rt[:, 0])), hr=float(np.median(rt[:, 1])))
            L.append('')
            L.append('预测形状 %s：外廓中位 长 %.3f 宽 %.3f 高 %.3f m；长/宽 %.2f，高/平均边 %.2f（真实框 %s，比例 %.2f / %.2f）' % (
                c, *np.median(sz, 0), np.median(rt[:, 0]), np.median(rt[:, 1]), TRUE[c].round(3).tolist(), TRUE[c][0] / TRUE[c][1], TRUE[c][2] / TRUE[c][:2].mean()))
    a, b = res['a_corner'], res['b_struct']
    res['gate'] = bool(b['ang'] <= a['ang'] - 1.0 and b['quad'] >= a['quad'])
    L.append('')
    L.append('阶段 A 继续条件（(b) 角度中位比 (a) 小 ≥ 1°，象限准确率不低于 (a)）：%s' % ('过' if res['gate'] else '不过'))
    json.dump(res, open(K_ + '/pilot_%s_s%d.json' % (split, seed), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    pickle.dump(rows, open(K_ + '/pilot_%s_s%d_rows.pkl' % (split, seed), 'wb'))
    open(G + '/FINAL_ADS.md', 'a', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
    if sys.argv[1] == 'adabn':
        adabn(int(sys.argv[2]))
    else:
        pilot(sys.argv[2], int(sys.argv[3]))
