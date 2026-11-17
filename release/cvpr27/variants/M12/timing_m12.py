# -*- coding: utf-8 -*-
"""[M12] per-frame inference timing + peak GPU memory of the deployed PST chain on MAV6D test (batch 1, after warm-up).
Modes (argv[1]):
  full    3 detectors (ensemble hook) + flip TTA | 2 depth refiners (XFZnl075 s5, s6) | 2 orientation students (S6I s0, s1, BN-adapted)
          each on the crop AND its horizontal mirror | height network (KPS s0, BN-adapted) | PnP: 4 member solves + 1 fused solve
  single  1 detector (NL075_s0), no flip | 1 depth refiner (s5) | 1 orientation student (S6I s0), no mirror | height network | PnP 1 + 1
  det     detector only (NL075_s0, no flip)
Every stage uses the deployed code paths (eval_camnorm hooks, final_unc crop/forward, adabn_kz recut/forward, kps head) on real test frames.
Timed per frame with torch.cuda.synchronize(); data loading of the detector input (cache read) is excluded and reported separately; the raw-image
read + undistortion for the zoom crop is reported as its own column (I/O). No labels are read (dataset ground truth is never accessed).
usage (any cwd): python timing_m12.py <full|single|det> [n_frames=300] [warmup=30]"""
import os, sys, time, json, pickle
import numpy as np
MODE = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 300; WU = int(sys.argv[3]) if len(sys.argv) > 3 else 30
ROOT = 'E:/Open3DUAVDet'; TOOLS = ROOT + '/tools'; G = ROOT + '/output/camnorm/geo1004'; NL = G + '/gproto/nolabel'; A = NL + '/nl075'
SCR = ROOT + '/output/camnorm/night1003/scratch'
OUT = G + '/gproto/review1113/M12/M1/timing'
os.environ.setdefault('TORCH_HOME', 'E:/torch_home')
for p in (G + '/lidar/kpsize', G + '/lidar/kzbody', SCR + '/kzst1', SCR + '/repl', G + '/finalads', G, TOOLS, ROOT):
    sys.path.insert(0, p)
import torch  # noqa
torch.cuda.set_per_process_memory_fraction(0.3)
dev = torch.device('cuda')
os.chdir(TOOLS)
from easydict import EasyDict  # noqa
from uavdet3d.config import cfg_from_yaml_file, cfg_from_list  # noqa
from uavdet3d.datasets import build_dataloader  # noqa
from uavdet3d.model import build_network, load_data_to_gpu  # noqa
from uavdet3d.utils import common_utils  # noqa
import eval_camnorm as EC  # noqa

MD = ROOT + '/output/models/uavdet_3d/camnorm/st2nl075wa_mav6d_r34_geo_full_tr'
DETS = [MD + '/NL075_s%d/ckpt/checkpoint_epoch_3.pth' % k for k in range(3)]
cfgp = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'
lg = common_utils.create_logger()
cfg = EasyDict(); cfg_from_yaml_file(cfgp, cfg)
cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], cfg)
cfg.DATA_CONFIG.DATA_SPLIT['test'] = 'test'; cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 1
ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=lg, training=False)
torch.cuda.reset_peak_memory_stats()
nd = 3 if MODE == 'full' else 1
ms = []
for ck in DETS[:nd]:
    m = build_network(model_cfg=cfg.MODEL, dataset=ds); a, b = m.load_params_from_file(filename=ck, to_cpu=False, logger=lg); assert a == b
    ms.append(m.cuda().eval())
det = ms[0]
if nd > 1:
    EC.install_ensemble_hook(det, ms[1:])
if MODE == 'full':
    EC.install_flip_hook(det)
# constant size for the geometric depth (value irrelevant for timing; deployed uses the val self-calibrated s_bar)
det.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'; det.model_cfg.POST_PROCESSING.KNOWN_SIZE = [0.35, 0.35, 0.2]

if MODE != 'det':
    import xpaste_final as XF  # noqa  (final_unc_nl crop / net code)
    nets = []
    for j in ((5, 6) if MODE == 'full' else (5,)):
        n = XF.v2.make('XFZnl075').cuda(); n.load_state_dict(torch.load('%s/softkd/XFZnl075_s%d.pt' % (G, j), map_location='cuda')); nets.append(n.eval())
    import adabn_kz as AB  # noqa
    import kzst_common as C  # noqa
    import kps_eval as KE  # noqa
    KE.register()
    stu, _, _, in_wh = AB.build_models(torch, dev, [A + '/adabn_all/S6I_s%d.pth' % k for k in ((0, 1) if MODE == 'full' else (0,))])
    kps = KE.E.build(A + '/adabn/kps_s0_adabn.pth', torch, dev)
    import repl_common as RC  # noqa
    P = RC.tag_paths('nl075_test_kzt7q')
    Dn, info = C.load_npz_info(P['npz'])
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1); ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    src = AB.NpzSource(Dn)
    kidx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(Dn['seq'], Dn['frame']))}
    import cv2  # noqa
    P8 = (np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * [0.35, 0.35, 0.2]).astype(np.float64)
    npnp = 5 if MODE == 'full' else 2
    rgb = np.load('E:/mmcache/mav6d_cn_obj/test/rgb.npy', mmap_mode='r')
print('[timing] mode %s: models loaded, allocated %.1f MB' % (MODE, torch.cuda.memory_allocated() / 2 ** 20), flush=True)
torch.cuda.reset_peak_memory_stats()

T = {k: [] for k in ('load', 'det', 'depth', 'io_raw', 'orient', 'height', 'pnp', 'total_gpu_chain')}
sync = torch.cuda.synchronize
k = 0
it = iter(loader)
with torch.no_grad():
    while k < N + WU:
        t0 = time.perf_counter()
        batch = next(it)
        meta = ds.metas[int(ds.valid_idx[k])]
        load_data_to_gpu(batch); sync(); t1 = time.perf_counter()
        res = det(batch); sync(); t2 = time.perf_counter()
        row = dict(load=t1 - t0, det=t2 - t1)
        if MODE != 'det':
            p = np.asarray(res['pred_boxes9d'][0], np.float64).reshape(-1, 10); c = np.asarray(res['confidence'][0], np.float64).reshape(-1)
            g = p[int(np.argmax(c)), :9] if len(c) else np.array([0, 0, 3.5, .35, .35, .2, 0, 0, 0])
            # depth refiner: final_unc_nl.run statements for the top box (crop of the 512x288 cache image)
            img = np.ascontiguousarray(rgb[int(ds.valid_idx[k])]); K = np.asarray(meta['K_in'], np.float64)
            cc = (XF.pe.PROTO8 * g[3:6]) @ XF.R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            q = (K @ cc.T).T; uv = q[:, :2] / q[:, 2:3]
            u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max(); w = float(np.clip(u1 - u0, 6, 170))
            v = XF.sw.view_img(XF.ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), XF.ce.KW * w), XF.S0, 0.0, 0.0, 1.0, w, False)
            x = XF.v2.skt.dt.prep(v[None], None, False).cuda()
            s = torch.from_numpy(np.array([[np.log(w / K[0, 0]), np.log(max(g[2], 1e-3))] + list(np.log(np.maximum(g[3:6], 1e-3)))], np.float32)).cuda()
            for n in nets:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    dh, lb, lgt = n(x, s)
                torch.sigmoid(lgt.float()).cpu()
            sync(); t3 = time.perf_counter()
            # orientation: zoom crop from the raw frame (deployed window of this frame) -> students (+ mirror) -> height net
            i = kidx[(str(meta['seq']), str(meta['frame']))]
            crop = AB.recut(Dn, i, src, in_wh); t4 = time.perf_counter()
            xi = AB.to_input(crop, nm, ns)[None]
            outs = AB.forward(stu, xi, torch, dev)
            if MODE == 'full':
                outs += AB.forward(stu, np.ascontiguousarray(xi[..., ::-1]), torch, dev)
            [o[0].cpu() for o in outs]
            sync(); t5 = time.perf_counter()
            tb = {'image': torch.from_numpy(np.ascontiguousarray(xi[:, None])).to(dev), 'batch_size': 1}
            for mod in kps.module_list:
                tb = mod(tb)
            f = tb['features_2d']; kps.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:])).float().cpu()
            sync(); t6 = time.perf_counter()
            Kc = np.array([[1000., 0, 256], [0, 1000., 144], [0, 0, 1]])
            for _ in range(npnp):     # member solves + fused solve (SQPnP on the 8 corners, as KFO.pnp)
                c2 = (Kc @ (P8 + [0, 0, 3.5]).T).T; c2 = c2[:, :2] / c2[:, 2:3] + np.random.randn(8, 2)
                cv2.solvePnP(P8, c2, Kc, None, flags=cv2.SOLVEPNP_SQPNP)
            t7 = time.perf_counter()
            row.update(depth=t3 - t2, io_raw=t4 - t3, orient=t5 - t4, height=t6 - t5, pnp=t7 - t6)
            row['total_gpu_chain'] = row['det'] + row['depth'] + row['orient'] + row['height'] + row['pnp']
        else:
            row['total_gpu_chain'] = row['det']
        if k >= WU:
            for kk, vv in row.items():
                T[kk].append(vv * 1000.0)
        k += 1
peak = torch.cuda.max_memory_allocated() / 2 ** 20
R = {kk: dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90))) for kk, v in T.items() if v}
R['n_frames'] = N; R['warmup'] = WU; R['peak_mem_MB_inference'] = peak
R['gpu'] = torch.cuda.get_device_name(0); R['mode'] = MODE
os.makedirs(OUT, exist_ok=True)
json.dump(R, open('%s/timing_%s.json' % (OUT, MODE), 'w'), indent=1)
print(json.dumps(R, indent=1), flush=True)
