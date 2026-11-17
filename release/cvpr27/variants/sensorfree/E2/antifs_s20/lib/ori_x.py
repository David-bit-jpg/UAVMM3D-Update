# [review1113 copy of srcx/lib/ori_x.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s20/lib] batch = $ORI_BS (default 16 = deployed)
# -*- coding: utf-8 -*-
"""[outdoor->srcx] orientation branch utilities (Module 5 / 6b), compact re-implementation for the outdoor run (no labels read).
  crops <recs.pkl> <split> <every> <name>      orientation crops (MODULES §0.2): zoom = 95 / w2D clipped to [1, ZMAX] around the frame's single
                                               detection, 16:9 window of the raw 1920x1080 image -> 512x288; saves D:/S2MUAV3D/pcmd_out_work/oc_<name>.npy/.pkl
                                               [outdoor] ZMAX = 18.75 (= window >= 512/5 raw px, the same max 5x raw upscaling as the outdoor sim
                                               KP crops; indoor [1,4] assumed indoor apparent sizes)
  bank <recs.pkl> <name>                       orientation-scale real background bank: conf >= 0.6, same-frame window of crop scale avoiding the
                                               detection box enlarged by 20%  -> pcmd_out_work/<name>.npy
  adabn <ckpt> <crops name> <out> [cap]        BN re-estimation on eligible crops (conf >= 0.5); cap = max crops per sequence (0 = all)
  read <crops name> <out.pkl> <ck1,ck2,...>    per network: heatmap (sigmoid) 5x5 around the detection cell, centre offsets, kp2d at those cells
  pseudo <crops name> <reads.pkl> <out name>   teacher pseudo-orientations: symmetric ensemble over the privileged nets (irlid, kdir, lid),
                                               gate d_sym(IR-distilled, LiDAR) <= 10 deg -> pcmd_out_work/<out name>.npz in ou_train format
Symmetric ensemble = MODULES §0.2 steps 1-5 (read cell from member-averaged heatmap; per-member PnP with s_bar; centre member by 180-folded yaw;
renumber members > 90 deg by the 180-deg-about-vertical corner correspondence; drop members > 45 deg folded; average corners; PnP)."""
import os, sys, pickle, json, time
import numpy as np, cv2
os.environ.setdefault('TORCH_HOME', 'E:/torch_home')
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s20/lib'); import sx  # noqa  [srcx] per source/target constants
ROOT = 'E:/Open3DUAVDet'; O = sx.O; W = sx.W
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/tools')
from scipy.spatial.transform import Rotation as Rot  # noqa
PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
FLIP180 = [int(np.where((np.abs(PROTO8 - PROTO8[i] * [-1, -1, 1]).sum(1) < 1e-9))[0][0]) for i in range(8)]
PI_Y = [int(np.argmin(np.abs(PROTO8 - PROTO8[i] * np.array([1, -1, 1])).sum(1))) for i in range(8)]
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
ZMAX = sx.T['zmax']
CFG = sx.S['kp_cfg']          # [srcx] source KP config (network construction only; reads hm / centre offset / kp2d)
REAL = sx.REAL
RN = (np.array(sx.T['rn'][0], np.float32), np.array(sx.T['rn'][1], np.float32))   # real-domain input norm (unlabeled train)


def sym(A, B):
    return min(np.degrees(np.arccos(np.clip((np.trace((A @ RZ[q]).T @ B) - 1) / 2, -1, 1))) for q in range(4))


def proj(box, K):
    c = (PROTO8 * box[3:6]) @ Rot.from_euler('xyz', box[6:9]).as_matrix().T + box[:3]
    q = (K @ c.T).T
    return q[:, :2] / q[:, 2:3]


# ------------------------------------------------------------------ crops
def metas(split):
    return {(m['seq'], m['frame']): m for m in pickle.load(open('%s/%s/index.pkl' % (REAL, split), 'rb'))['metas']}


def window(u_in, v_in, z):
    w = sx.TW / z; h = w * 288 / 512
    cu, cv_ = u_in * sx.TW / 512, v_in * sx.TH / 288
    return float(np.clip(cu - w / 2, 0, sx.TW - w)), float(np.clip(cv_ - h / 2, 0, sx.TH - h)), w, h


def cut(img, x0, y0, w):
    s = 512.0 / w
    M = np.array([[s, 0, (0.5 - x0) * s - 0.5], [0, s, (0.5 - y0) * s - 0.5]], np.float64)
    return cv2.warpAffine(img, M, (512, 288), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), s


def crop_K(K, x0, y0, s):
    Kc = np.asarray(K, np.float64).copy()
    Kc[0, 0] *= s; Kc[1, 1] *= s; Kc[0, 2] = s * (K[0, 2] + 0.5 - x0) - 0.5; Kc[1, 2] = s * (K[1, 2] + 0.5 - y0) - 0.5
    return Kc


def cmd_crops(recs_pkl, split, every, name):
    D = pickle.load(open(recs_pkl, 'rb')); recs = [r for r in D['recs'][split] if r['box'] is not None][::int(every)]
    M = metas(split)
    X = np.lib.format.open_memmap('%s/oc_%s.npy' % (W, name), 'w+', np.uint8, (len(recs), 288, 512, 3))
    info = []
    for n, r in enumerate(recs):
        m = M[(r['seq'], r['frame'])]; K = np.asarray(m['K_in']); Kr = np.asarray(m['K_raw'], np.float64)
        q = K @ r['box'][:3]; u, v = q[0] / q[2], q[1] / q[2]
        z = float(np.clip(95.0 / max(r['w2d'], 1e-3), 1.0, ZMAX))
        x0, y0, w, h = window(u, v, z)
        img = sx.imread_raw(m)
        X[n], s = cut(img, x0, y0, w)
        Kc = crop_K(Kr, x0, y0, s)
        qc = Kc @ r['box'][:3]
        info.append(dict(seq=r['seq'], frame=r['frame'], conf=r['conf'], box=r['box'], K_crop=Kc, uc=qc[0] / qc[2], vc=qc[1] / qc[2], zoom=z, win=(x0, y0, w, h)))
    X.flush(); pickle.dump(dict(info=info, size=np.asarray(D['size'])), open('%s/oc_%s.pkl' % (W, name), 'wb'))
    print('crops %s: %d (%s every %s); zoom median %.2f, clipped at ZMAX %.1f%%' % (name, len(info), split, every, np.median([i['zoom'] for i in info]), 100 * np.mean([i['zoom'] >= ZMAX for i in info])))


def cmd_bank(recs_pkl, name):
    D = pickle.load(open(recs_pkl, 'rb')); M = metas('train'); rng = np.random.default_rng(0); out = []
    for r in D['recs']['train']:
        if r['box'] is None or r['conf'] < 0.6:
            continue
        m = M[(r['seq'], r['frame'])]; K = np.asarray(m['K_in']); Kr = np.asarray(m['K_raw'], np.float64)
        q = K @ r['box'][:3]; u, v = q[0] / q[2], q[1] / q[2]
        z = float(np.clip(95.0 / max(r['w2d'], 1e-3), 1.0, ZMAX)); w = sx.TW / z; h = w * 288 / 512
        uv = proj(r['box'], Kr); bx0, by0 = uv.min(0); bx1, by1 = uv.max(0)
        ex, ey = 0.2 * (bx1 - bx0), 0.2 * (by1 - by0); bx0, bx1, by0, by1 = bx0 - ex, bx1 + ex, by0 - ey, by1 + ey
        if w >= sx.TW:
            continue
        for _ in range(20):
            x0, y0 = rng.uniform(0, sx.TW - w), rng.uniform(0, sx.TH - h)
            if x0 > bx1 or x0 + w < bx0 or y0 > by1 or y0 + h < by0:
                img = sx.imread_raw(m)
                out.append(cut(img, x0, y0, w)[0]); break
    np.save('%s/%s.npy' % (W, name), np.stack(out)); print('orientation bank %s: %d windows' % (name, len(out)))


# ------------------------------------------------------------------ networks
def build(ck, bn=None):
    import torch
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils
    os.chdir(ROOT + '/tools')
    cfg = EasyDict(); cfg_from_yaml_file(CFG, cfg)
    lg = common_utils.create_logger()
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=lg, training=False)
    m = build_network(model_cfg=cfg.MODEL, dataset=ds)
    a, b = m.load_params_from_file(filename=ck, to_cpu=False, logger=lg)
    assert a == b, (ck, a, b)
    return m.cuda().eval()


def batches(X, idx, bs=None):
    bs = bs or int(os.environ.get('ORI_BS', '16'))
    import torch
    for i in range(0, len(idx), bs):
        x = X[idx[i:i + bs]].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        x = (x - RN[0][None, :, None, None]) / RN[1][None, :, None, None]
        yield idx[i:i + bs], torch.from_numpy(np.ascontiguousarray(x[:, None])).cuda()


def head_out(m, x):
    tb = {'image': x, 'batch_size': x.shape[0]}
    for mod in m.module_list:
        tb = mod(tb)
        if mod is m.dense_head_2d:
            break
    return m.dense_head_2d.forward_loss_dict['pred_center_dict'] if 'pred_center_dict' not in tb else tb['pred_center_dict']


def cmd_adabn(ck, name, out, cap=0):
    import torch
    X = np.load('%s/oc_%s.npy' % (W, name), mmap_mode='r'); info = pickle.load(open('%s/oc_%s.pkl' % (W, name), 'rb'))['info']
    el = [i for i, r in enumerate(info) if r['conf'] >= 0.5]
    if cap:
        rng = np.random.default_rng(0); by = {}
        for i in el:
            by.setdefault(info[i]['seq'], []).append(i)
        el = sorted(sum([list(rng.choice(v, min(cap, len(v)), replace=False)) for v in by.values()], []))
    m = build(ck)
    bns = [mm for mm in m.modules() if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats(); b.momentum = None; b.train()
    with torch.no_grad():
        for _, x in batches(X, np.array(el), int(os.environ.get('ORI_BS', '16'))):
            head_out(m, x)
    sd = torch.load(ck, map_location='cpu', weights_only=False); ms = sd.get('model_state', sd)
    new = m.state_dict()
    for k in new:
        if k in ms and ('running_' in k or 'num_batches' in k):
            ms[k] = new[k].detach().cpu()
    torch.save({'model_state': ms, 'epoch': sd.get('epoch', 0) if isinstance(sd, dict) else 0}, out)
    print('adabn %s: %d BN layers on %d crops -> %s' % (os.path.basename(ck), len(bns), len(el), out))


def cmd_read(name, out, cks):
    import torch
    X = np.load('%s/oc_%s.npy' % (W, name), mmap_mode='r'); info = pickle.load(open('%s/oc_%s.pkl' % (W, name), 'rb'))['info']
    res = {}
    jobs = [(ck, False) for ck in cks] + ([(ck, True) for ck in cks] if os.environ.get('ORI_MIRROR') == '1' else [])
    for ck, mir in jobs:
        m = build(ck); H, C, KP = [], [], []
        with torch.no_grad():
            for ii, x in batches(X, np.arange(len(info))):
                pc = head_out(m, torch.flip(x, dims=[-1]) if mir else x)
                hm = torch.sigmoid(pc['hm'].float()).cpu().numpy().reshape(len(ii), -1, 36, 64)[:, 0]
                cr = pc['center_res'].float().cpu().numpy().reshape(len(ii), -1, 36, 64)
                kp = pc['kp2d'].float().cpu().numpy().reshape(len(ii), -1, 36, 64)
                if mir:   # mirrored crop (pixel u -> 511-u): maps back, centre x-offset 1-r-1/8, corner x-offsets negated, corners renumbered by body-y mirror
                    hm = hm[..., ::-1]; cr = cr[..., ::-1].copy(); kp = kp[..., ::-1].reshape(len(ii), 8, 2, 36, 64).copy()
                    cr[:, 0] = 1.0 - cr[:, 0] - 0.125
                    kp[:, :, 0] = -kp[:, :, 0]
                    k2 = np.empty_like(kp)
                    for a_ in range(8):
                        k2[:, PI_Y[a_]] = kp[:, a_]
                    kp = k2.reshape(len(ii), 16, 36, 64)
                for b, i in enumerate(ii):
                    r0, c0 = int(np.clip(info[i]['vc'] // 8, 0, 35)), int(np.clip(info[i]['uc'] // 8, 0, 63))
                    rs, cs = np.clip(np.arange(r0 - 2, r0 + 3), 0, 35), np.clip(np.arange(c0 - 2, c0 + 3), 0, 63)
                    H.append(hm[b][np.ix_(rs, cs)]); C.append(cr[b][:, rs][:, :, cs]); KP.append(kp[b][:, rs][:, :, cs])
                    if b == 0:
                        pass
        res[ck + ('#M' if mir else '')] = dict(hm=np.stack(H), cr=np.stack(C), kp=np.stack(KP))
        del m; torch.cuda.empty_cache()
    for i in range(len(info)):
        r0, c0 = int(np.clip(info[i]['vc'] // 8, 0, 35)), int(np.clip(info[i]['uc'] // 8, 0, 63))
        info[i]['cells'] = (np.clip(np.arange(r0 - 2, r0 + 3), 0, 35), np.clip(np.arange(c0 - 2, c0 + 3), 0, 63))
    pickle.dump(dict(reads=res, order=list(res.keys()), cells=[i['cells'] for i in info]), open(out, 'wb'))
    print('read %d nets on %d crops -> %s' % (len(cks), len(info), out))


# ------------------------------------------------------------------ ensemble
def pnp(c2d, Kc, lwh):
    obj = (PROTO8 * lwh).astype(np.float64)
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
        try:
            ok, rv, tv = cv2.solvePnP(obj, c2d.astype(np.float64), Kc, None, flags=flag)
        except cv2.error:
            ok = False
        if ok and np.isfinite(tv).all() and tv[2, 0] > 0:
            if flag == cv2.SOLVEPNP_EPNP:
                ok, rv, tv = cv2.solvePnP(obj, c2d.astype(np.float64), Kc, None, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
            return cv2.Rodrigues(rv)[0], tv.reshape(3)
    return None


def corners(rd, mem, i, rc):
    r, c = rc
    cr = rd[mem]['cr'][i][:, r, c]; kp = rd[mem]['kp'][i][:, r, c]
    return None, cr, kp


def ensemble(rd, mems, i, cells, Kc, lwh):
    """MODULES §0.2 symmetric ensemble -> (R, t, corners2d) or None"""
    hm = np.mean([rd[m]['hm'][i] for m in mems], 0)
    r, c = np.unravel_index(int(np.argmax(hm)), hm.shape)
    gr, gc = cells[0][r], cells[1][c]
    C, sols = [], []
    for m in mems:
        cr = rd[m]['cr'][i][:, r, c]; kp = rd[m]['kp'][i][:, r, c]
        ctr = (np.array([gc, gr], np.float64) + cr[:2]) * 8.0
        c2 = ctr[None] + kp.reshape(8, 2) * 64.0
        s = pnp(c2, Kc, lwh)
        C.append(c2); sols.append(s)
    ok = [k for k, s in enumerate(sols) if s is not None]
    if not ok:
        return None
    yaw = lambda A, B: abs(np.degrees(np.arctan2((A.T @ B)[1, 0], (A.T @ B)[0, 0])))
    fold = lambda d: min(d, 180 - d)
    if len(ok) > 1:
        sc = [sum(fold(yaw(sols[a][0], sols[b][0])) for b in ok if b != a) for a in ok]
        ce = ok[int(np.argmin(sc))]
    else:
        ce = ok[0]
    kept, nflip = [], 0
    for k in ok:
        d = yaw(sols[ce][0], sols[k][0])
        if fold(d) > 45:
            continue
        cc = C[k][FLIP180] if d > 90 else C[k]
        nflip += d > 90; kept.append(cc)
    avg = np.mean(kept, 0)
    if nflip > len(kept) / 2:
        avg = avg[FLIP180]
    s = pnp(avg, Kc, lwh)
    if s is None:
        s = pnp(np.mean(C, 0), Kc, lwh)
    return None if s is None else (s[0], s[1], avg)


def cmd_pseudo(name, reads_pkl, outname, groups):
    """groups: dict(T=[ckpts of teachers], IR=[...], LiD=[...])"""
    info = pickle.load(open('%s/oc_%s.pkl' % (W, name), 'rb')); lwh = np.asarray(info['size'], np.float64); info = info['info']
    R = pickle.load(open(reads_pkl, 'rb')); rd = R['reads']; cells = R['cells']
    rows = []
    for i, f in enumerate(info):
        Kc = f['K_crop']
        out = {g: ensemble(rd, mems, i, cells[i], Kc, lwh) for g, mems in groups.items()}
        if any(v is None for v in out.values()):
            continue
        d_il = sym(out['IR'][0], out['LiD'][0])
        rows.append(dict(i=i, seq=f['seq'], frame=f['frame'], R=out['T'][0], t=out['T'][1], K=Kc, d_il=d_il, conf=f['conf'], t_pos=f['box'][:3]))
    d_il = np.array([r['d_il'] for r in rows]); gate = d_il <= 10.0
    eul = np.array([Rot.from_matrix(r['R']).as_euler('xyz') for r in rows])
    np.savez('%s/%s.npz' % (W, outname), idx=np.array([r['i'] for r in rows]), seq=np.array([r['seq'] for r in rows]), frame=np.array([r['frame'] for r in rows]),
             K_crop=np.stack([r['K'] for r in rows]), t_pnp=np.stack([r['t'] for r in rows]), t_pos=np.stack([r['t_pos'] for r in rows]),
             lwh=np.tile(lwh, (len(rows), 1)), R=np.stack([r['R'] for r in rows]), eul=eul, d_il=d_il, d_tr=np.zeros(len(rows)), gate=gate,
             w=np.exp(-(d_il ** 2) / (2 * 10.0 ** 2)), conf=np.array([r['conf'] for r in rows]))
    print('pseudo %s: %d / %d crops have a pseudo-orientation; gate (IR vs LiDAR <= 10 deg) passes %d (%.1f%%); d_il median %.1f' % (
        outname, len(rows), len(info), gate.sum(), 100 * gate.mean(), np.median(d_il)))


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'crops':
        cmd_crops(*sys.argv[2:6])
    elif c == 'bank':
        cmd_bank(sys.argv[2], sys.argv[3])
    elif c == 'adabn':
        cmd_adabn(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    elif c == 'read':
        cmd_read(sys.argv[2], sys.argv[3], sys.argv[4].split(','))
    elif c == 'pseudo':
        g = json.loads(sys.argv[5])
        cmd_pseudo(sys.argv[2], sys.argv[3], sys.argv[4], g)


# ------------------------------------------------------------------ height network (Module 2c / 6a)
def cmd_kps(ck, name, out, cap=80):
    """BN re-estimation of the height network on eligible crops (conf >= 0.5, <= cap per sequence, seed 0), then per-crop height ratio
    r = z-extent / mean(x, y extent) of the predicted body-frame structure points at the read cell (heatmap peak within 5x5 of the detection cell)."""
    import torch
    for p in ('E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kzlid', 'E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kzbody', 'E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kpsize'):
        sys.path.insert(0, p)
    import uavdet3d.model.detectors as dets
    from center_det_mm import CenterDetMM
    from center_det_mmk import CenterDetMMK
    dets.__all__['CenterDetMM'] = CenterDetMM; dets.__all__['CenterDetMMK'] = CenterDetMMK
    global CFG
    CFG = sx.S['kps_cfg']
    m = build(ck)
    names = name.split(',')
    X = np.load('%s/oc_%s.npy' % (W, names[0]), mmap_mode='r'); info = pickle.load(open('%s/oc_%s.pkl' % (W, names[0]), 'rb'))['info']
    if cap >= 0:
        el = [i for i, r in enumerate(info) if r['conf'] >= 0.5]
        rng = np.random.default_rng(0); by = {}
        for i in el:
            by.setdefault(info[i]['seq'], []).append(i)
        el = sorted(sum([list(rng.choice(v, min(cap, len(v)), replace=False)) if cap else v for v in by.values()], []))
        bns = [mm for mm in m.modules() if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm)]
        for b in bns:
            b.reset_running_stats(); b.momentum = None; b.train()
        with torch.no_grad():
            for _, x in batches(X, np.array(el), int(os.environ.get('ORI_BS', '16'))):
                tb = {'image': x, 'batch_size': x.shape[0]}
                for mod in m.module_list:
                    tb = mod(tb)
                    if mod is m.dense_head_2d:
                        break
                f = tb['features_2d']; m.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:]))
        m.eval()
        print('kps adabn on %d crops' % len(el))
    for nm_ in names:
      X = np.load('%s/oc_%s.npy' % (W, nm_), mmap_mode='r'); info = pickle.load(open('%s/oc_%s.pkl' % (W, nm_), 'rb'))['info']
      res = {}
      with torch.no_grad():
        for ii, x in batches(X, np.arange(len(info)), int(os.environ.get('ORI_BS', '16'))):
            tb = {'image': x, 'batch_size': x.shape[0]}
            for mod in m.module_list:
                tb = mod(tb)
                if mod is m.dense_head_2d:
                    break
            pc = tb.get('pred_center_dict', m.dense_head_2d.forward_loss_dict['pred_center_dict'])
            hm = torch.sigmoid(pc['hm'].float()).cpu().numpy().reshape(len(ii), -1, 36, 64)[:, 0]
            f = tb['features_2d']; kk = m.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:])).float().cpu().numpy()
            for b, i in enumerate(ii):
                r0, c0 = int(np.clip(info[i]['vc'] // 8, 0, 35)), int(np.clip(info[i]['uc'] // 8, 0, 63))
                rs, cs = np.clip(np.arange(r0 - 2, r0 + 3), 0, 35), np.clip(np.arange(c0 - 2, c0 + 3), 0, 63)
                h = hm[b][np.ix_(rs, cs)]; r, c = np.unravel_index(int(np.argmax(h)), h.shape)
                L = kk[b, 16:, rs[r], cs[c]].reshape(8, 3); e = np.ptp(L, 0)
                res[(info[i]['seq'], info[i]['frame'])] = float(e[2] / max(e[:2].mean(), 1e-6))
      pickle.dump(res, open(out % nm_, 'wb')); v = np.array(list(res.values()))
      print('kps %s on %s: %d crops, ratio median %.3f' % (os.path.basename(ck), nm_, len(res), np.median(v)))


if __name__ == '__main__' and sys.argv[1] == 'kps':
    cmd_kps(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 80)
