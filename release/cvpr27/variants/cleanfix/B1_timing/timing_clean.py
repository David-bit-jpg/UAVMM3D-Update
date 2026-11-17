# -*- coding: utf-8 -*-
"""[cleanfix B1] per-frame inference timing + peak GPU memory of the CLEAN deployed PST chain (srcx/in2mav) on MAV6D test.
Replicates review1113/M12/timing_m12.py (M12_REPORT.md 'Timing and memory'): batch 1, WU warm-up frames then N timed frames, torch.cuda.synchronize
around every stage, process memory fraction 0.3 (memcap.py), detector-input cache read excluded and reported separately ('load'), raw-frame
read reported as its own I/O column, peak = torch.cuda.max_memory_allocated with all models loaded, weights resident = memory_allocated after
loading. Modes as M12:
  full    3 self-trained detectors (srcx st_in2mav DET1_s0..2, eval_camnorm ensemble hook) + flip TTA | 2 depth refiners (ref_st3_s5/s6) |
          2 orientation students (ou/S_ab0/1, BN-adapted) on the zoom crop AND its mirror | height network (sx kps, KPSs0) | PnP 4 member + 1 fused
  single  1 detector (DET1_s0), no flip | 1 refiner (s5) | 1 student (S_ab0), no mirror | height network | PnP 1 + 1
  det     detector only (DET1_s0, no flip)
Only difference to M12: every stage uses the CLEAN chain's code paths (srcx/lib copies): det_infer_x.build/run post-processing (REAL_CFG,
known size = the run's s_bar), refiner_x.det_crop + Net.fwd (fp32, as refiner_x infer) on the raw undistorted frame (sx.imread_raw, K_raw),
ori_x zoom window (95 / w2D clipped to [1, ZMAX]) + cut + real-norm + head_out (+ torch.flip mirror, as ori_x read) and the cmd_kps head path,
ori_x.pnp (SQPnP, EPnP fallback) on the decoded corners. The raw frame is read ONCE and shared by refiner and orientation crops (deployed
chain also reads it in both steps; reported once in io_raw). If the deployed width gate (6-170 px) would skip the refiner, the crop is made with
the width clipped (upper bound: the refiner always runs). No label is read (the dataset GT is never accessed).
usage: python timing_clean.py <full|single|det> [N=300] [WU=30]    env SMOKE=1 -> CPU (plumbing test; writes timing_<mode>_smoke.json)"""
import os, sys, time, json, pickle
import numpy as np
MODE = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 300; WU = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SMOKE = os.environ.get('SMOKE') == '1'
B = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B1_timing'
sys.path.insert(0, B + '/lib')
os.environ.setdefault('TORCH_HOME', 'E:/torch_home')
import torch  # noqa
dev = torch.device('cpu' if SMOKE else 'cuda')
if dev.type == 'cuda':
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEMFRAC', '0.3')))
import sx  # noqa  (copy; SX_SRC=in SX_TGT=mav)
assert (sx.SRC, sx.TGT, sx.TEACH) == ('in', 'mav', 'full'), (sx.SRC, sx.TGT, sx.TEACH)
import det_infer_x as DI  # noqa  (chdir tools)
import refiner_x as RX  # noqa
import ori_x as OO  # noqa
from easydict import EasyDict  # noqa
from uavdet3d.config import cfg_from_yaml_file, cfg_from_list  # noqa
from uavdet3d.datasets import build_dataloader  # noqa
from uavdet3d.model import build_network, load_data_to_gpu  # noqa
from uavdet3d.utils import common_utils  # noqa
import eval_camnorm as EC  # noqa
RUN = sx.O_RUN                                   # srcx/in2mav (read only)
MD = sx.M + '/srcx/st_in2mav'
DETS = [MD + '/DET1_s%d/ckpt/checkpoint_epoch_3.pth' % k for k in range(3)]
REFS = [RUN + '/ref/ref_st3_s%d.pt' % j for j in (5, 6)]
STUS = [RUN + '/ou/S_ab%d.pth' % k for k in (0, 1)]
sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
sbar = np.asarray(pickle.load(open(RUN + '/recs/det1.pkl', 'rb'))['size'], np.float64)

# ---------------- detector(s): = det_infer_x.build (ensemble hook, flip hook in full) with batch 1, known size s_bar
lg = common_utils.create_logger()
cfg = EasyDict(); cfg_from_yaml_file(DI.CFG, cfg)
cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], cfg)
cfg.DATA_CONFIG.DATA_SPLIT['test'] = 'test'; cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 1
ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=lg, training=False)
if dev.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
nd = 3 if MODE == 'full' else 1
ms = []
for ck in DETS[:nd]:
    m = build_network(model_cfg=cfg.MODEL, dataset=ds); a, b = m.load_params_from_file(filename=ck, to_cpu=dev.type == 'cpu', logger=lg); assert a == b
    ms.append(m.to(dev).eval())
det = ms[0]
if nd > 1:
    EC.install_ensemble_hook(det, ms[1:])
if MODE == 'full':
    EC.install_flip_hook(det)
det.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'; det.model_cfg.POST_PROCESSING.KNOWN_SIZE = [float(v) for v in sbar]


def build_kz(ck, cfgp):
    """= ori_x.build with a device argument"""
    c = EasyDict(); cfg_from_yaml_file(cfgp, c)
    d2, _, _ = build_dataloader(dataset_cfg=c.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=lg, training=False)
    m = build_network(model_cfg=c.MODEL, dataset=d2)
    a, b = m.load_params_from_file(filename=ck, to_cpu=dev.type == 'cpu', logger=lg); assert a == b, (ck, a, b)
    return m.to(dev).eval()


if MODE != 'det':
    nets = []
    for p in REFS[:(2 if MODE == 'full' else 1)]:
        n = RX.Net().to(dev); n.load_state_dict(torch.load(p, map_location=dev)); nets.append(n.eval())
    stu = [build_kz(ck, sx.S['kp_cfg']) for ck in STUS[:(2 if MODE == 'full' else 1)]]
    for p in ('E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kzlid', 'E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kzbody', 'E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kpsize'):
        sys.path.insert(0, p)
    import uavdet3d.model.detectors as dets  # noqa
    from center_det_mm import CenterDetMM  # noqa
    from center_det_mmk import CenterDetMMK  # noqa
    dets.__all__['CenterDetMM'] = CenterDetMM; dets.__all__['CenterDetMMK'] = CenterDetMMK
    kps = build_kz(sx.S['kps'], sx.S['kps_cfg'])   # deployed: BN re-estimated in memory by cmd_kps (weights' timing identical)
    META = OO.metas('test')
    RN0 = torch.from_numpy(OO.RN[0]).to(dev)[None, :, None, None]; RN1 = torch.from_numpy(OO.RN[1]).to(dev)[None, :, None, None]
    npnp = 5 if MODE == 'full' else 2
wres = torch.cuda.memory_allocated() / 2 ** 20 if dev.type == 'cuda' else float('nan')
print('[timing] mode %s: models loaded, allocated %.1f MB' % (MODE, wres), flush=True)
if dev.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()


def decode(pc, b, r0, c0, Kc):
    """ori_x.ensemble member step: argmax cell in the 5x5 around the detection cell -> corners -> PnP input"""
    hm = torch.sigmoid(pc['hm'].float())[b, 0].cpu().numpy(); cr = pc['center_res'].float()[b].cpu().numpy(); kp = pc['kp2d'].float()[b].cpu().numpy()
    rs, cs = np.clip(np.arange(r0 - 2, r0 + 3), 0, 35), np.clip(np.arange(c0 - 2, c0 + 3), 0, 63)
    h = hm[np.ix_(rs, cs)]; r, c = np.unravel_index(int(np.argmax(h)), h.shape); gr, gc = rs[r], cs[c]
    ctr = (np.array([gc, gr], np.float64) + cr[:2, gr, gc]) * 8.0
    return ctr[None] + kp[:16, gr, gc].reshape(8, 2) * 64.0


T = {k: [] for k in ('load', 'det', 'io_raw', 'depth', 'orient', 'height', 'pnp', 'total_gpu_chain')}
k = 0; it = iter(loader); nskip_gate = 0
with torch.no_grad():
    while k < N + WU:
        t0 = time.perf_counter()
        batch = next(it)
        meta = ds.metas[int(ds.valid_idx[k])]
        if dev.type == 'cuda':
            load_data_to_gpu(batch)
        else:   # SMOKE: same conversion on CPU
            import uavdet3d.model as _um
            for kk_, vv_ in batch.items():
                if isinstance(vv_, np.ndarray) and kk_ not in _um.__dict__.get('_SKIP', ['batch_size', 'scene_id', 'seq_id', 'frame_id', 'brightness', 'intrinsic', 'extrinsic', 'distortion', 'raw_im_size', 'new_im_size', 'obj_size', 'stride', 'im_num', 'obj_num', 'gt_name', 'relative_im_path', 'setting_id', 'gt_diff']):
                    batch[kk_] = torch.from_numpy(vv_).float()
        sync(); t1 = time.perf_counter()
        res = det(batch); sync(); t2 = time.perf_counter()
        row = dict(load=t1 - t0, det=t2 - t1)
        if MODE != 'det':
            p = np.asarray(res['pred_boxes9d'][0], np.float64).reshape(-1, 10); c = np.asarray(res['confidence'][0], np.float64).reshape(-1)
            Kin = np.asarray(meta['K_in'], np.float64)
            g = p[int(np.argmax(c)), :9] if len(c) else np.array([0, 0, 3.5, *sbar, 0, 0, 0])
            w2d = DI.w2d_of(g, Kin)
            m = META[(meta['seq'], meta['frame'])]; Kr = np.asarray(m['K_raw'], np.float64)
            img = sx.imread_raw(m); t3 = time.perf_counter()
            # depth refiner(s): refiner_x.det_crop + Net.fwd (as cmd_infer)
            cc = RX.det_crop(img, g, Kr, sx.T.get('wscale', 1.0))
            if cc is None:
                nskip_gate += 1
                uv, bb = RX.proj(g, Kr)
                bb = bb or (900., 1020., 500., 580.); ws = sx.T.get('wscale', 1.0)
                w = float(np.clip(bb[1] - bb[0], 6 * ws, 170 * ws)); cc = (RX.crop_at(img, 0.5 * (bb[0] + bb[1]), 0.5 * (bb[2] + bb[3]), w), w)
            X = RX.dt.prep(cc[0][None], None, False).to(dev)
            s = torch.from_numpy(np.array([RX.scal(cc[1], Kr[0, 0], g)], np.float32)).to(dev)
            for n in nets:
                dh, lb, lgt, _ = n.fwd(X, s)
                torch.sigmoid(lgt.float()).cpu(); dh.float().cpu()
            sync(); t4 = time.perf_counter()
            # orientation: ori_x zoom window from the raw frame -> students (+ mirror in full)
            q = Kin @ g[:3]; u, v = q[0] / q[2], q[1] / q[2]
            z = float(np.clip(95.0 / max(w2d, 1e-3), 1.0, OO.ZMAX))
            x0, y0, w, h = OO.window(u, v, z)
            crop, sc = OO.cut(img, x0, y0, w); Kc = OO.crop_K(Kr, x0, y0, sc); qc = Kc @ g[:3]
            r0, c0 = int(np.clip((qc[1] / qc[2]) // 8, 0, 35)), int(np.clip((qc[0] / qc[2]) // 8, 0, 63))
            xi = torch.from_numpy(np.ascontiguousarray(crop.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)).to(dev)
            xi = ((xi - RN0) / RN1)[:, None]
            outs = []
            for mm in stu:
                outs.append(OO.head_out(mm, xi))
                if MODE == 'full':
                    outs.append(OO.head_out(mm, torch.flip(xi, dims=[-1])))
            C2 = [decode(pc, 0, r0, c0, Kc) for pc in outs]       # (mirror members are decoded without the un-mirroring bookkeeping: timing only)
            sync(); t5 = time.perf_counter()
            # height network: cmd_kps head path
            tb = {'image': xi, 'batch_size': 1}
            for mod in kps.module_list:
                tb = mod(tb)
                if mod is kps.dense_head_2d:
                    break
            f = tb['features_2d']; kps.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:])).float().cpu()
            sync(); t6 = time.perf_counter()
            for j in range(npnp):     # member solves + fused solve (ori_x.pnp)
                OO.pnp(C2[j] if j < len(C2) else np.mean(C2, 0), Kc, sbar)
            t7 = time.perf_counter()
            row.update(io_raw=t3 - t2, depth=t4 - t3, orient=t5 - t4, height=t6 - t5, pnp=t7 - t6)
            row['total_gpu_chain'] = row['det'] + row['depth'] + row['orient'] + row['height'] + row['pnp']
        else:
            row['total_gpu_chain'] = row['det']
        if k >= WU:
            for kk, vv in row.items():
                T[kk].append(vv * 1000.0)
        k += 1
R = {kk: dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90))) for kk, v in T.items() if v}
R.update(n_frames=N, warmup=WU, mode=MODE, weights_resident_MB=wres, refiner_width_gate_clipped=nskip_gate,
         peak_mem_MB_inference=torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == 'cuda' else None,
         gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu (SMOKE)', nets=dict(det=DETS[:nd], ref=REFS[:1 if MODE == 'single' else 2] if MODE != 'det' else [], stu=STUS[:1 if MODE == 'single' else 2] if MODE != 'det' else [], kps=sx.S['kps'] if MODE != 'det' else None),
         nvidia_smi_used_MB_at_end=os.popen('nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader').read().strip())
os.makedirs(B + '/out', exist_ok=True)
json.dump(R, open('%s/out/timing_%s%s.json' % (B, MODE, '_smoke' if SMOKE else ''), 'w'), indent=1)
print(json.dumps(R, indent=1), flush=True)
