# -*- coding: utf-8 -*-
"""11-10 无标签链（PREREG 第 3 条）：两种测距合并权重 w 只用**仿真验证集**的标签定。
  w = σ²_红外 / (σ²_红外 + σ²_角点)，σ = 对数距离误差先减中位后的稳健散布 1.4826·MAD（与真实部署先减整体偏移一致）。
仿真验证集 = full_v2/test（间隔 2 帧）。每帧取纯仿真检测器 P0_s0（irloc 配方，与第二段仿真训练样本同一个检测器）的最高分检测，
置信 ≥0.5、与仿真真值中心相距 < 1 m、投影宽 ∈ [6,170] 像素（与部署同范围）。
  (a) 红外教的测距网络：第二段第 2 步 V2Pp_s{5,6}_st_<arm>（第 3 步从它出发；第 3 步的训练目标要用 w，故不能用第 3 步自己）；
      裁窗 / 标量与第二段仿真训练样本完全相同（softkd_extract.py 取窗）；e_红外 = mean(δ̂) − log(z真 / z检)。
  (b) 角点测距：朝向学生 S6I 种子 0、1（不重训）+ 各自镜像视图 = 4 个读数，部署同法：放大裁窗（目标 2D 宽 95 像素、倍数夹在 [1,4]）
      → 5×5 钩子窗 → SA-TM 融合 8 角点 → PnP 平移深度 z_kp（机身尺寸 = 检测框长宽高）；e_角点 = log(z_kp / z真)。
      仿真是源域，网络用各自训练结束时的权重与 BN 统计（部署时的 AdaBN 统计是真实域的，不用于仿真）。
只读仿真标签；不读任何真实图像或真实标签。输出 nolabel/<arm>/w_sim.json；仿真检测 / 角点读数与臂无关，缓存在 nolabel/w_sim_cache.pkl。
用法：python w_sim.py <arm>"""
import os, sys, json, pickle, time
sys.dont_write_bytecode = True
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import numpy as np
import cv2
import torch
ROOT = '@@ROOT@@'; G = ROOT + '/output/camnorm/geo1004'; NL = G + '/gproto/nolabel'
SCR = ROOT + '/output/camnorm/night1003/scratch'
OU = G + '/gproto/wf21/orient_uda'
for p in (SCR + '/kzst1', SCR + '/repl', ROOT + '/tools', ROOT, G, NL):
    if p not in sys.path:
        sys.path.insert(0, p)
arm = sys.argv[1]
OUTJ = '%s/%s/w_sim.json' % (NL, arm)
CACHE = NL + '/w_sim_cache.pkl'
IV = 2
rob = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))


def sim_detect():
    """P0_s0 在 full_v2/test（间隔 IV）上推理（同 softkd_extract.py 的模型 / 配置），每帧最高分检测与最近仿真真值配对"""
    cwd = os.getcwd()
    from easydict import EasyDict
    import uavdet3d.model.detectors as dets
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils, pose_eval as pe
    from center_det_geoir import CenterDetGeoIR
    dets.__all__['CenterDetGeoIR'] = CenterDetGeoIR
    os.chdir(ROOT + '/tools')
    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_irloc.yaml', cfg)
    dc = cfg.DATA_CONFIG
    dc.DATA_SPLIT['test'] = 'test'
    dc.SAMPLED_INTERVAL['test'] = IV
    ds, loader, _ = build_dataloader(dataset_cfg=dc, batch_size=8, dist=False, workers=0, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    model.load_params_from_file(filename=ROOT + '/output/models/uavdet_3d/camnorm/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_irloc/P0_s0/ckpt/checkpoint_epoch_5.pth', to_cpu=False, logger=logger)
    model.cuda().eval()
    recs = pe.run_inference(model, loader)
    del model
    torch.cuda.empty_cache()
    os.chdir(cwd)
    out = []
    for r in recs:
        if len(r['pred']) == 0 or len(r['gt']) == 0:
            continue
        pi = int(np.argmax(r['conf']))
        if r['conf'][pi] < 0.5:
            continue
        p = np.asarray(r['pred'][pi], np.float64)
        gt = np.asarray(r['gt'], np.float64).reshape(-1, 9)
        k = int(np.argmin(np.linalg.norm(gt[:, :3] - p[:3], axis=1)))
        if np.linalg.norm(gt[k, :3] - p[:3]) >= 1.0 or p[2] <= 1e-3:
            continue
        out.append(dict(seq=str(r['seq_id']), frame=str(r['frame_id']), pred=p, gt=gt[k]))
    return out


def build_cache(t0):
    import cropw_extract as ce
    import silw_train as sw
    from scipy.spatial.transform import Rotation as R
    from uavdet3d.utils import pose_eval as pe
    import cropdist_eval as cde
    import adabn_kz as AB
    import kzst_common as C
    import kp_zoom as KZ
    ce.P, ce.KW = 256, 3.0
    S0 = float(np.sqrt(1.2 * 2.5))
    KFO = C.KFO
    dets_ = sim_detect()
    print('仿真验证集：配对检测 %d 帧，%.0f s' % (len(dets_), time.time() - t0), flush=True)
    imgs = cde.Images('sim')                          # full_v2/test 原分辨率 jpg + K_in（仿真）
    info = json.load(open(G + '/gproto/wf21/strict/tags/wsfnb_train_kz/kz_members_wsfnb_train_kz.json', encoding='utf-8'))
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1); ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    search = int(info.get('search', 2)); KS = float(info.get('KP_SCALE') or 64.0)
    dev = torch.device('cuda')
    cks = ['%s/runs/S6I_s%d/ckpt_last.pth' % (OU, k) for k in (0, 1)]
    ms, kcfg, kds, in_wh = AB.build_models(torch, dev, cks)
    in_w, in_h = in_wh
    kstride = int(kds.dataset_cfg.STRIDE)
    P8 = KFO.PROTO8
    PI = [int(np.argmin(np.abs(P8 - P8[i] * np.array([1, -1, 1])).sum(1))) for i in range(8)]

    def unmirror(kp, hm):              # = recut_mirror.py
        hm2 = torch.flip(hm, dims=[-1])
        k = torch.flip(kp, dims=[-1]).reshape(8, 2, *kp.shape[-2:]).clone()
        k[:, 0] = -k[:, 0]
        k2 = torch.empty_like(k)
        for i in range(8):
            k2[PI[i]] = k[i]
        return k2.reshape(16, *kp.shape[-2:]), hm2
    rows = []
    for d in dets_:
        img, Kr, _ = imgs.get(d['seq'], d['frame'])
        raw_h, raw_w = img.shape[:2]
        p = d['pred']
        c8 = (pe.PROTO8 * p[3:6]) @ R.from_euler('xyz', p[6:9]).as_matrix().T + p[:3]
        q = (Kr @ c8.T).T
        if (q[:, 2] <= 1e-3).any():
            continue
        uv8 = q[:, :2] / q[:, 2:3]
        u0, u1, v0, v1 = uv8[:, 0].min(), uv8[:, 0].max(), uv8[:, 1].min(), uv8[:, 1].max()
        w = u1 - u0
        if not (6.0 <= w <= 170.0):
            continue
        # (a) 第二段裁窗（与 softkd_extract.py 同一取窗；网络在臂相关部分跑）
        view = sw.view_img(ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), ce.KW * w), S0, 0.0, 0.0, 1.0, w, False)
        scal = [np.log(w / Kr[0, 0]), np.log(p[2])] + list(np.log(np.maximum(p[3:6], 1e-3)))
        # (b) 放大裁窗（kp_zoom 钩子同法；输入坐标 = 原图 × in_w/raw_w）
        sc_in = in_w / raw_w
        qc = Kr @ p[:3]
        u_in, v_in = qc[0] / qc[2] * sc_in, qc[1] / qc[2] * sc_in
        w2d = w * sc_in
        z = float(np.clip(95.0 / max(w2d, 1e-3), 1.0, 4.0))
        x0, y0, ww, hh = KZ.crop_window(u_in, v_in, z, raw_w, raw_h, in_w, in_h)
        s = in_w / ww
        Mw = np.array([[s, 0, (0.5 - x0) * s - 0.5], [0, s, (0.5 - y0) * s - 0.5]], np.float64)
        crop = cv2.warpAffine(img, Mw, (in_w, in_h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        X = AB.to_input(crop, nm, ns)[None]
        o1 = AB.forward(ms, X, torch, dev)
        o2 = AB.forward(ms, np.ascontiguousarray(X[..., ::-1]), torch, dev)
        maps = [o[0][0] for o in o1]; hms = [o[1][0] for o in o1]
        for o in o2:
            k_, h_ = unmirror(o[0][0], o[1][0]); maps.append(k_); hms.append(h_)
        uc = (u_in * raw_w / in_w - x0) * s; vc = (v_in * raw_h / in_h - y0) * s
        cc, rc = int(uc / kstride), int(vc / kstride)
        Hh, Ww = hms[0].shape
        r0, r1 = max(0, rc - search), min(Hh, rc + search + 1); c0, c1 = max(0, cc - search), min(Ww, cc + search + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        hm_b = torch.stack(hms).mean(0); sub = hm_b[r0:r1, c0:c1]
        j = int(torch.argmax(sub.reshape(-1))); rb, cb = r0 + j // sub.shape[1], c0 + j % sub.shape[1]
        S_ = 2 * search + 1; Mn = len(maps)
        hm_win = np.full((Mn, S_, S_), np.nan, np.float32); kp_win = np.full((Mn, 16, S_, S_), np.nan, np.float32)
        ar, ac = r0 - (rc - search), c0 - (cc - search)
        hm_win[:, ar:ar + (r1 - r0), ac:ac + (c1 - c0)] = torch.stack(hms)[:, r0:r1, c0:c1].float().cpu().numpy()
        kp_win[:, :, ar:ar + (r1 - r0), ac:ac + (c1 - c0)] = torch.stack(maps)[:, :, r0:r1, c0:c1].float().cpu().numpy()
        cell = (int(rb - (rc - search)), int(cb - (cc - search)))
        K512 = Kr.copy(); K512[:2] *= sc_in                        # 解码器在 512×288 输入坐标里解 PnP（真实部署同：K_in、raw_size = new_size = 512×288、步长 8）
        pr, pc_ = int(v_in / 8), int(u_in / 8)
        geo = KFO.Geo(K512, np.zeros(5), (in_w, in_h), (in_w, in_h), 8, (pr, pc_), (u_in / 8 - pc_, v_in / 8 - pr), p[3:6], KS)
        ctx = KFO.FrameCtx(hm_win, kp_win, z, geo)
        mem = list(range(Mn))
        o_, fb_, plan, _ = KFO.fuse_subset(ctx, mem, cell, ['C0', 'SATM'], KFO.rot_from_eul(p[6:9]))
        zkp = np.nan
        if plan is not None:
            rt = KFO.pnp(KFO.f32(KFO.satm_vec([ctx.x(m, cell) for m in mem], plan)), ctx.g, return_rt=True)
            if rt is not None and rt[2][2, 0] > 0:
                zkp = float(rt[2][2, 0])
        rows.append(dict(seq=d['seq'], frame=d['frame'], view=view, scal=np.array(scal, np.float32), z_det=float(p[2]), z_gt=float(d['gt'][2]), z_kp=zkp))
    del ms
    torch.cuda.empty_cache()
    C0 = dict(rows=rows, n_det=len(dets_), iv=IV, students=cks, note='full_v2/test sim val; P0_s0 top-1 detections paired to sim GT')
    pickle.dump(C0, open(CACHE + '.tmp', 'wb')); os.replace(CACHE + '.tmp', CACHE)
    print('仿真角点读数：%d 帧，有 z_kp %d，%.0f s' % (len(rows), int(np.isfinite([r['z_kp'] for r in rows]).sum()), time.time() - t0), flush=True)
    return C0


def main():
    if os.path.exists(OUTJ):
        print('exists', OUTJ); return
    t0 = time.time()
    C0 = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else build_cache(t0)
    if arm == 'cache':
        return
    rows = C0['rows']
    # ---------- 红外教的测距网络（第二段第 2 步，臂相关）
    import secondstage_v2_nl as v2
    nets = []
    for j in (5, 6):
        n = v2.make('V2Pp').cuda()
        n.load_state_dict(torch.load('%s/softkd/V2Pp_s%d_st_%s.pt' % (G, j, arm), map_location='cuda'))
        nets.append(n.eval())
    V = np.stack([r['view'] for r in rows]); Sc = np.stack([r['scal'] for r in rows])
    dh = []
    with torch.no_grad():
        for b in range(0, len(V), 256):
            x = v2.skt.dt.prep(V[b:b + 256], None, False).cuda(); s = torch.from_numpy(Sc[b:b + 256]).cuda()
            ds = []
            for n in nets:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    d_, _, _ = n(x, s)
                ds.append(d_.float().cpu().numpy())
            dh.append(np.mean(ds, 0))
    dh = np.concatenate(dh)
    zg = np.array([r['z_gt'] for r in rows]); zd = np.array([r['z_det'] for r in rows]); zk = np.array([r['z_kp'] for r in rows])
    ok = np.isfinite(zk) & (zk > 0)
    e_ir = (dh - np.log(zg / zd))[ok]
    e_kp = np.log(zk[ok] / zg[ok])
    s_ir, s_kp = rob(e_ir), rob(e_kp)
    w = s_ir ** 2 / (s_ir ** 2 + s_kp ** 2)
    res = dict(arm=arm, w=float(w), sigma_ir=s_ir, sigma_kp=s_kp, n=int(ok.sum()), n_rows=len(rows), n_det=C0['n_det'],
               corr=float(np.corrcoef(e_ir - np.median(e_ir), e_kp - np.median(e_kp))[0, 1]),
               bias_ir=float(np.median(e_ir)), bias_kp=float(np.median(e_kp)), sigma_det=rob(np.log(zd / zg)[ok]),
               refiner=['softkd/V2Pp_s%d_st_%s.pt' % (j, arm) for j in (5, 6)], students=C0['students'], sim_split='full_v2/test interval %d' % C0['iv'],
               rule='w = s_ir^2/(s_ir^2+s_kp^2), s = 1.4826*MAD after median removal of log-distance error; simulated labels only')
    json.dump(res, open(OUTJ, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('%s：σ_红外 %.4f  σ_角点 %.4f  相关 %.2f  n=%d  → w = %.4f（检测器自身 σ %.4f）' % (arm, s_ir, s_kp, res['corr'], res['n'], w, res['sigma_det']), flush=True)


if __name__ == '__main__':
    main()
