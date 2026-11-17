# [review1113 copy of srcx/lib/refiner_x.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s20/lib] SX_SEEDSHIFT on training seeds; env batches; resume states; [H3] no silhouette composites
# -*- coding: utf-8 -*-
"""[outdoor->srcx] Module 4 depth refiner (IR-silhouette supervision + cross-domain composites + real adaptation + pseudo-target fine-tuning),
re-implemented compactly for the outdoor run with the SAME network (softkd_train.DCNetU), losses and recipes as MODULES §Module 4 /
secondstage_v2.py (V2P -> V2Pp -> V2Rp). Helpers reused: softkd_extract.sil_full / proj_box, cropw_extract.window, silw_train.view_img,
dc_train.prep (resolution degradation + photometric), udast_train.appearance (grey / invert / blur / JPEG).

Sub-commands (all outputs under gproto/outdoor/ref/):
  sim   <det_ckpt>                 simulated samples: sim-only detector on outdoor-sim train (every 2nd frame, VAL_ZOOMS 1x/2.5x alternately,
                                   <=5 boxes conf>=0.05), greedy-matched to GT if 3D distance < 0.25*|t_gt| ([outdoor] indoor 1 m at ~4 m);
                                   depth crop + IR silhouette (1-A) + 5 scalars + delta*=log(z_gt/z_det)
  bank  <recs.pkl>                 real background bank from single-frame detections on unlabeled real train (conf>=0.5, width 6-170 px),
                                   same-frame window moved 1.8 box widths in one of 8 directions; 8000 kept (seed 0)
  real  <recs.pkl> <name>          real depth crops of a detector's records (conf>=0.5, width 6-170 px) on the real train split -> ref/real_<name>
  train <stage> <seed> [opts]      stage 1 (V2P: sim 30 epochs), 2 (V2Pp: real adaptation 2000 steps), 3 (V2Rp: pseudo-target 2000 steps)
  infer <ckpt> <recs.pkl> <split> <out.pkl>   per-detection delta_hat, log b, silhouette body brightness (colour class)"""
import os, sys, pickle, time, copy
import numpy as np, cv2, torch
import torch.nn as nn, torch.nn.functional as F
os.environ.setdefault('TORCH_HOME', 'E:/torch_home')
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s20/lib'); import sx  # noqa  [srcx] per source/target constants
SILW = 0.0 if sx.NOIR else 1.0   # [11-12] no-teacher control: IR silhouette losses off without IR
ROOT = 'E:/Open3DUAVDet'; G = ROOT + '/output/camnorm/geo1004'; O = sx.O; RO = O + '/ref'
sys.path.insert(0, ROOT); sys.path.insert(0, G)
import cropw_extract as ce  # noqa  (chdir to tools)
import silw_train as sw  # noqa
import softkd_train as skt  # noqa
import dc_train as dt  # noqa
from udast_train import appearance  # noqa
from uda_train import GRL  # noqa
from scipy.spatial.transform import Rotation as Rot  # noqa
ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(3.0))
P8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
SIM = sx.SIM
SIM_CFG = sx.SIM_CFG
REAL = sx.REAL
os.makedirs(RO, exist_ok=True)


def proj(box, K):
    c = (P8 * box[3:6]) @ Rot.from_euler('xyz', box[6:9]).as_matrix().T + box[:3]
    q = (K @ c.T).T
    if (q[:, 2] <= 1e-3).any():
        return None, None
    uv = q[:, :2] / q[:, 2:3]
    return uv, (uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max())


def crop_at(img, cx, cy, w):
    return sw.view_img(ce.window(img, cx, cy, ce.KW * w), S0, 0.0, 0.0, 1.0, w, False)


def det_crop(img, box, K, wscale=1.0):
    """deployment depth crop of a detected box in the ORIGINAL image (K = raw intrinsics) -> (crop, w_proj, cx, cy) or None"""
    uv, bb = proj(box, K)
    if bb is None or box[2] <= 1e-3:
        return None
    w = bb[1] - bb[0]
    if not (6.0 <= w / wscale <= 170.0):   # 11-13: real raw-K calls pass sx.T wscale (mav 3.75 -> gate in 512-px input px); sim (K_in) and anti/antifs: 1.0
        return None
    cx, cy = 0.5 * (bb[0] + bb[1]), 0.5 * (bb[2] + bb[3])
    return crop_at(img, cx, cy, w), float(w), cx, cy


def scal(w, f, box):
    return [float(np.log(w / f)), float(np.log(box[2]))] + [float(x) for x in np.log(np.maximum(np.abs(box[3:6]), 1e-3))]


# ------------------------------------------------------------------ data
def cmd_sim(det_ck):
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils, pose_eval as pe
    import softkd_extract as SE
    cfg = EasyDict(); cfg_from_yaml_file(SIM_CFG, cfg)
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = 'train'; cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 2
    lg = common_utils.create_logger()
    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=lg, training=False)
    m = build_network(model_cfg=cfg.MODEL, dataset=ds); m.load_params_from_file(filename=det_ck, to_cpu=False, logger=lg); m.cuda().eval()
    recs = pe.run_inference(m, loader); del m; torch.cuda.empty_cache()
    d = pickle.load(open(SIM + '/train/index.pkl', 'rb'))
    key = {(str(mm['seq']), str(mm['frame'])): i for i, mm in enumerate(d['metas']) if mm}
    mm = {k: (np.memmap('%s/train/%s.bin' % (SIM, k), dtype=np.uint8, mode='r'), np.load('%s/train/%s_index.npy' % (SIM, k))) for k in ('rgb_jpg', 'ir_jpg')}
    rgbs, sils, meta = [], [], []
    for r in recs:
        gt, pred, conf = r['gt'], r['pred'], r['conf']
        if not len(gt) or not len(pred):
            continue
        used = np.zeros(len(gt), bool); pairs = []
        for pi in np.argsort(-conf):
            dd = np.linalg.norm(gt[:, :3] - pred[pi, :3], axis=1) / np.linalg.norm(gt[:, :3], axis=1); dd[used] = np.inf
            k = int(np.argmin(dd))
            if dd[k] < 0.25:
                used[k] = True; pairs.append((pi, k))
        if not pairs:
            continue
        i = key[(r['seq_id'], r['frame_id'])]; K = np.asarray(d['metas'][i]['K_in'], np.float64)
        o, ln = mm['rgb_jpg'][1][i]; img = cv2.imdecode(np.asarray(mm['rgb_jpg'][0][int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
        o, ln = mm['ir_jpg'][1][i]; ir = cv2.imdecode(np.asarray(mm['ir_jpg'][0][int(o):int(o) + int(ln)]), cv2.IMREAD_GRAYSCALE)
        for pi, k in pairs:
            p, g = pred[pi], gt[k]
            c = det_crop(img, p, K)
            uvg, bg = proj(g, K)
            if c is None or bg is None:
                continue
            v, w, cx, cy = c
            s = SE.sil_full(ir, uvg, bg)
            sv = crop_at(s, cx, cy, w) if s is not None else np.zeros((128, 128), np.uint8)
            rgbs.append(v); sils.append(sv)
            meta.append(dict(seq=r['seq_id'], frame=r['frame_id'], w=w, s=scal(w, K[0, 0], p), delta=float(np.log(g[2] / p[2])), has_sil=s is not None))
    np.save(RO + '/sim_rgb.npy', np.stack(rgbs)); np.save(RO + '/sim_sil.npy', np.stack(sils)); pickle.dump(meta, open(RO + '/sim_meta.pkl', 'wb'))
    dl = np.array([x['delta'] for x in meta])
    print('sim samples %d (with IR silhouette %d), delta* median %+.3f |.| median %.3f' % (len(meta), sum(x['has_sil'] for x in meta), np.median(dl), np.median(np.abs(dl))))


def real_iter(recs, split='train'):
    idx = pickle.load(open('%s/%s/index.pkl' % (REAL, split), 'rb'))
    meta = {(m['seq'], m['frame']): m for m in idx['metas']}
    for r in recs:
        m = meta[(r['seq'], r['frame'])]
        yield r, m


def cmd_bank(recs_pkl):
    recs = pickle.load(open(recs_pkl, 'rb'))['recs']['train']
    rng = np.random.default_rng(0)
    out, src = [], []
    D = [(a, b) for a in (-1.8, 0, 1.8) for b in (-1.8, 0, 1.8) if (a, b) != (0, 0)]
    for r, m in real_iter(recs):
        if r['box'] is None or r['conf'] < 0.5:
            continue
        K = np.asarray(m['K_raw'], np.float64)
        uv, bb = proj(r['box'], K)
        if bb is None:
            continue
        w = bb[1] - bb[0]
        if not (6 <= w / sx.T.get('wscale', 1.0) <= 170):   # 11-13: see det_crop
            continue
        cx, cy = 0.5 * (bb[0] + bb[1]), 0.5 * (bb[2] + bb[3]); half = 0.5 * S0 * w
        img = None
        for j in rng.permutation(len(D)):
            x, y = cx + D[j][0] * w, cy + D[j][1] * w
            if x - half >= 0 and y - half >= 0 and x + half <= sx.TW and y + half <= sx.TH:
                img = sx.imread_raw(m)
                out.append(crop_at(img, x, y, w)); src.append((r['seq'], r['frame']))
                break
    keep = np.sort(rng.choice(len(out), min(8000, len(out)), replace=False))
    np.save(RO + '/bank.npy', np.stack([out[i] for i in keep]))
    print('bank: %d windows from %d frames, kept %d' % (len(out), len(recs), len(keep)))


def cmd_real(recs_pkl, name, split='train', conf_min=0.5):
    D = pickle.load(open(recs_pkl, 'rb'))
    recs = D['recs'][split]
    rgbs, meta = [], []
    for r, m in real_iter(recs, split):
        if r['box'] is None or r['conf'] < conf_min:
            continue
        K = np.asarray(m['K_raw'], np.float64)
        img = sx.imread_raw(m)
        c = det_crop(img, r['box'], K, sx.T.get('wscale', 1.0))
        if c is None:
            continue
        rgbs.append(c[0]); meta.append(dict(seq=r['seq'], frame=r['frame'], w=c[1], s=scal(c[1], K[0, 0], r['box']), z_det=float(r['box'][2]), conf=r['conf']))
    np.save('%s/real_%s_rgb.npy' % (RO, name), np.stack(rgbs)); pickle.dump(meta, open('%s/real_%s_meta.pkl' % (RO, name), 'wb'))
    print('real crops %s/%s: %d of %d frames' % (name, split, len(meta), len(recs)))


# ------------------------------------------------------------------ training
class Net(skt.DCNetU):
    def fwd(self, x, s):
        s0 = self.stem(x); c1 = self.l1(self.pool(s0)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        y = self.lat[4](c4)
        for f, lat in ((c3, self.lat[3]), (c2, self.lat[2]), (c1, self.lat[1]), (s0, self.lat[0])):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        lg = self.sil(F.interpolate(self.smooth(y), size=x.shape[-2:], mode='bilinear', align_corners=False)).squeeze(1)
        feat = c4.mean(dim=(2, 3))
        z = torch.cat([feat, self.sc((s - self.s_mean) / self.s_std), self.sil_feats(lg).to(c4.dtype)], 1)
        o = self.head(z)
        return o[:, 0], o[:, 1].clamp(-6.0, 2.0), lg, feat


def sim_batch(rgb, sil, S, dl, idx, bank, rng, strong=True):
    """simulated batch with composites (prob 0.5, alpha = IR silhouette/255, needs max > 128) + appearance aug + flip"""
    x = rgb[idx].copy(); m = sil[idx].astype(np.float32) / 255.0
    for n in range(len(idx)):
        if not sx.NOIR and rng.random() < 0.5 and sil[idx[n]].max() > 128 and False:   # [heads H3] silhouette composites OFF (draw kept)
            bg = bank[rng.integers(len(bank))]
            if rng.random() < 0.5:
                bg = bg[:, ::-1]
            a = m[n][..., None]
            x[n] = np.clip(a * x[n] + (1 - a) * bg, 0, 255).astype(np.uint8)
    if strong:
        x = appearance(x, rng)
    fl = rng.random(len(idx)) < 0.5
    x[fl] = x[fl][:, :, ::-1]; mm = m.copy(); mm[fl] = mm[fl][:, :, ::-1]
    X = dt.prep(x, rng, True)
    return X, torch.from_numpy(S[idx]), torch.from_numpy(dl[idx]), torch.from_numpy((mm > 0.5).astype(np.float32))


def real_views(x, rng):
    fl = rng.random(len(x)) < 0.5
    w = x.copy(); w[fl] = w[fl][:, :, ::-1]
    s = w.copy()
    s = appearance(s, rng)
    return dt.prep(w, rng, False), dt.prep(s, rng, True), fl


def load_sim():
    rgb = np.load(RO + '/sim_rgb.npy'); sil = np.load(RO + '/sim_sil.npy'); meta = pickle.load(open(RO + '/sim_meta.pkl', 'rb'))
    S = np.array([x['s'] for x in meta], np.float32); dl = np.array([x['delta'] for x in meta], np.float32)
    return rgb, sil, S, dl


def ckp(stage, j, tag=''):
    return '%s/ref_st%d_s%d%s.pt' % (RO, stage, j, tag)


def cmd_train(stage, j, real_name='d0', dp_file=None, tag=''):
    js = j + int(os.environ.get('SX_SEEDSHIFT', '0')); torch.manual_seed(js); rng = np.random.default_rng(1000 + js)
    print('[sup3] refiner stage %d nominal seed %d -> effective seed %d' % (stage, j, js), flush=True)
    dev = torch.device('cuda')
    rgb, sil, S, dl = load_sim(); bank = np.load(RO + '/bank.npy')
    net = Net().to(dev)
    log = open('%s/train_st%d_s%d%s.log' % (RO, stage, j, tag), 'a')
    RS = '%s/resume_st%d_s%d%s.pt' % (RO, stage, j, tag)
    if stage == 1:
        net.s_mean.copy_(torch.from_numpy(S.mean(0))); net.s_std.copy_(torch.from_numpy(S.std(0) + 1e-6))
        BS, EP = int(os.environ.get('REF_ST1_BS', '128')), int(os.environ.get('REF_EP', '30'))
        nb = len(S) // BS; tot = nb * EP
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=tot)
        net.train(); t0 = time.time(); ep0 = 0
        if os.path.exists(RS):
            z = torch.load(RS, map_location=dev, weights_only=False)
            if z.get('BS') == BS:
                net.load_state_dict(z['net']); opt.load_state_dict(z['opt']); sch.load_state_dict(z['sch']); rng.bit_generator.state = z['rng']
                torch.set_rng_state(z['trng']); ep0 = z['ep']; print('RESUME stage 1 from epoch %d (%s)' % (ep0, RS), file=log, flush=True); print('RESUME stage 1 from epoch %d' % ep0, flush=True)
            else:
                print('resume state has batch %s != %d (OOM halving): stage 1 restarts from scratch' % (z.get('BS'), BS), file=log, flush=True)
        for ep in range(ep0, EP):
            perm = rng.permutation(len(S))
            for b in range(nb):
                X, s, d, m = sim_batch(rgb, sil, S, dl, perm[b * BS:(b + 1) * BS], bank, rng)
                dh, lb, lg, _ = net.fwd(X.to(dev), s.to(dev))
                loss = skt.nll(dh, lb, d.to(dev)) + SILW * skt.bce_dice(lg.float(), m.to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            print('ep %d loss %.4f %.0fs' % (ep, loss.item(), time.time() - t0), file=log, flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), rng=rng.bit_generator.state, trng=torch.get_rng_state(), ep=ep + 1, BS=BS), RS)
        torch.save(net.state_dict(), ckp(1, j, tag)); os.path.exists(RS) and os.remove(RS); return
    # stages 2 / 3
    xr = np.load('%s/real_%s_rgb.npy' % (RO, real_name))
    if stage == 2:
        net.load_state_dict(torch.load(ckp(1, j, tag), map_location=dev))
    else:
        net.load_state_dict(torch.load(ckp(2, j, tag), map_location=dev))
        dp = np.load(dp_file).astype(np.float32); ok = np.where(np.isfinite(dp))[0]
        print('stage 3: %d / %d real crops with pseudo-target' % (len(ok), len(dp)), file=log, flush=True)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    disc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)).to(dev) if stage == 2 else None
    params = list(net.parameters()) + (list(disc.parameters()) if disc is not None else [])
    STEPS = int(os.environ.get('REF_STEPS', '2000'))
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    net.train(); t0 = time.time(); st0 = 0
    pool = np.arange(len(xr)) if stage == 2 else ok
    if os.path.exists(RS):
        z = torch.load(RS, map_location=dev, weights_only=False)
        net.load_state_dict(z['net']); ema.load_state_dict(z['ema']); opt.load_state_dict(z['opt']); sch.load_state_dict(z['sch'])
        if disc is not None:
            disc.load_state_dict(z['disc'])
        rng.bit_generator.state = z['rng']; torch.set_rng_state(z['trng']); st0 = z['st']
        print('RESUME stage %d from step %d (%s)' % (stage, st0, RS), file=log, flush=True); print('RESUME stage %d from step %d' % (stage, st0), flush=True)
    for st in range(st0, STEPS):
        X, s, d, m = sim_batch(rgb, sil, S, dl, rng.choice(len(S), int(os.environ.get('REF_TRAIN_BS', '64')), replace=False), bank, rng)
        dh, lb, lg, fs = net.fwd(X.to(dev), s.to(dev))
        loss = skt.nll(dh, lb, d.to(dev)) + SILW * skt.bce_dice(lg.float(), m.to(dev))
        ri = rng.choice(pool, int(os.environ.get('REF_TRAIN_BS', '64')), replace=len(pool) < int(os.environ.get('REF_TRAIN_BS', '64')))
        sr = torch.from_numpy(np.array([real_meta_s[i] for i in ri], np.float32)).to(dev)
        xw, xs, fl = real_views(xr[ri], rng)
        with torch.no_grad():
            dt_, lbt, lgt, _ = ema.fwd(xw.to(dev), sr)
            pt = torch.sigmoid(lgt.float())
        ds_, lbs, lgs, fr = net.fwd(xs.to(dev), sr)
        if stage == 2:
            loss = loss + ((ds_.float() - dt_.float()).abs() * torch.exp(-lbt.float())).mean() + SILW * F.binary_cross_entropy_with_logits(lgs.float(), pt)
            p = st / STEPS; lam = 0.1 * (2 / (1 + np.exp(-10 * p)) - 1)
            f = torch.cat([fs, fr]); y = torch.cat([torch.zeros(len(fs)), torch.ones(len(fr))]).to(dev)
            loss = loss + F.binary_cross_entropy_with_logits(disc(GRL.apply(f.float(), lam)).squeeze(1), y)
        else:
            tgt = torch.from_numpy(dp[ri]).to(dev)
            loss = loss + skt.nll(ds_, lbs, tgt) + SILW * F.binary_cross_entropy_with_logits(lgs.float(), pt)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            for pe_, pn in zip(ema.parameters(), net.parameters()):
                pe_.mul_(0.999).add_(pn.detach(), alpha=0.001)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        if st % 200 == 0:
            print('step %d loss %.4f %.0fs' % (st, loss.item(), time.time() - t0), file=log, flush=True)
        if (st + 1) % 250 == 0 and st + 1 < STEPS:
            torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), disc=(disc.state_dict() if disc is not None else None),
                            rng=rng.bit_generator.state, trng=torch.get_rng_state(), st=st + 1), RS)
    torch.save(ema.state_dict(), ckp(stage, j, tag)); os.path.exists(RS) and os.remove(RS)


real_meta_s = None


def cmd_infer(ck, recs_pkl, split, out, conf_min=0.0):
    dev = torch.device('cuda')
    net = Net().to(dev); net.load_state_dict(torch.load(ck, map_location=dev)); net.eval()
    recs = pickle.load(open(recs_pkl, 'rb'))['recs'][split]
    res = {}
    buf = []

    def flush():
        if not buf:
            return
        X = dt.prep(np.stack([b[1] for b in buf]), None, False).to(dev)
        s = torch.from_numpy(np.array([b[2] for b in buf], np.float32)).to(dev)
        with torch.no_grad():
            dh, lb, lg, _ = net.fwd(X, s)
            p = torch.sigmoid(lg.float()).cpu().numpy()
        for (k, v, _), a, b, pp in zip(buf, dh.float().cpu().numpy(), lb.float().cpu().numpy(), p):
            gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32)
            msk = pp > 0.5
            res[k] = dict(d=float(a), lb=float(b), lum=float(gray[msk].mean()) if msk.sum() >= 10 else None)
        buf.clear()
    for r, m in real_iter(recs, split):
        if r['box'] is None or r['conf'] < conf_min:
            continue
        K = np.asarray(m['K_raw'], np.float64)
        img = sx.imread_raw(m)
        c = det_crop(img, r['box'], K, sx.T.get('wscale', 1.0))
        if c is None:
            continue
        buf.append(((r['seq'], r['frame']), c[0], scal(c[1], K[0, 0], r['box'])))
        if len(buf) == int(os.environ.get('REF_BS', '128')):
            flush()
    flush()
    pickle.dump(res, open(out, 'wb'))
    d = np.array([v['d'] for v in res.values()])
    print('infer %s %s: %d crops / %d frames, delta_hat median %+.3f, |.|<=log1.4 %.1f%%' % (os.path.basename(ck), split, len(res), len(recs), np.median(d), 100 * np.mean(np.abs(d) <= np.log(1.4))))


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'sim':
        cmd_sim(sys.argv[2])
    elif c == 'bank':
        cmd_bank(sys.argv[2])
    elif c == 'real':
        cmd_real(sys.argv[2], sys.argv[3])
    elif c == 'train':
        stage, j = int(sys.argv[2]), int(sys.argv[3])
        rn = os.environ.get('REF_REAL', 'd0'); tag = os.environ.get('REF_TAG', '')
        if stage > 1:
            real_meta_s = [x['s'] for x in pickle.load(open('%s/real_%s_meta.pkl' % (RO, rn), 'rb'))]
        cmd_train(stage, j, rn, os.environ.get('REF_DP'), tag)
    elif c == 'infer':
        cmd_infer(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
