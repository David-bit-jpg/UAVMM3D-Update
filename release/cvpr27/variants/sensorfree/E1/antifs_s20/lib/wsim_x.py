# [review1113 copy of srcx/lib/wsim_x.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s20/lib] sys.path -> redirected sx; code unchanged
# -*- coding: utf-8 -*-
"""[outdoor->srcx] fusion weight w by the inverse-variance rule on SIMULATED (outdoor) validation (copy of the rule in nolabel/w_sim.py):
  w = s_IR^2 / (s_IR^2 + s_kp^2), s = 1.4826*MAD of the (median-removed) log-distance errors on sim val (= outdoor sim test split, every 2nd frame).
Detections: sim-only detector, boxes with conf >= 0.5, matched to a sim GT within 25% of its distance ([outdoor] indoor 1 m), projected width 6-170 px.
 (a) IR-taught refiner = stage 2 (stage 3 needs w): e_IR = delta_hat - log(z_gt/z_det) on the deployment depth crop.
 (b) corner distance: orientation student(s) + mirror view on the orientation crop (zoom 95/w2D clipped to [1, 12.5] = same max 5x raw upscaling
     on the 1280-wide sim image), symmetric ensemble -> PnP depth with the detected box size; e_kp = log(z_kp / z_gt).
Only simulation labels are read. usage: python wsim_out.py <det_ckpt> <refiner_st2.pt> <student ck,...> <out.json>"""
import os, sys, json, pickle
import numpy as np, cv2, torch
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s20/lib'); import sx  # noqa  [srcx] per source/target constants
O = sx.O
import refiner_x as RF  # noqa (chdir to tools)
import ori_x as OO  # noqa
from scipy.spatial.transform import Rotation as Rot  # noqa
os.environ['ORI_MIRROR'] = '1'
det_ck, ref_ck, stu, out = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(','), sys.argv[4]   # [srcx] refiner stage-2 members averaged (original w_sim.py: V2Pp s5, s6)
CACHE = O + '/wsim_cache.pkl'
if not os.path.exists(CACHE):
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils, pose_eval as pe
    cfg = EasyDict(); cfg_from_yaml_file(RF.SIM_CFG, cfg)
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = 'test'; cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 2
    lg = common_utils.create_logger()
    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=lg, training=False)
    m = build_network(model_cfg=cfg.MODEL, dataset=ds); m.load_params_from_file(filename=det_ck, to_cpu=False, logger=lg); m.cuda().eval()
    recs = pe.run_inference(m, loader); del m; torch.cuda.empty_cache()
    pickle.dump(recs, open(CACHE, 'wb'))
recs = pickle.load(open(CACHE, 'rb'))
d = pickle.load(open(RF.SIM + '/test/index.pkl', 'rb'))
key = {(str(mm['seq']), str(mm['frame'])): i for i, mm in enumerate(d['metas']) if mm}
mm = (np.memmap(RF.SIM + '/test/rgb_jpg.bin', dtype=np.uint8, mode='r'), np.load(RF.SIM + '/test/rgb_jpg_index.npy'))
crops, sc, rows, oc, info = [], [], [], [], []
for r in recs:
    for pi in np.where(r['conf'] >= 0.5)[0]:
        p = r['pred'][pi]; g = r['gt']
        if not len(g):
            continue
        dd = np.linalg.norm(g[:, :3] - p[:3], axis=1) / np.linalg.norm(g[:, :3], axis=1); k = int(np.argmin(dd))
        if dd[k] >= 0.25:
            continue
        i = key[(r['seq_id'], r['frame_id'])]; K = np.asarray(d['metas'][i]['K_in'], np.float64)
        o, ln = mm[1][i]; img = cv2.imdecode(np.asarray(mm[0][int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
        c = RF.det_crop(img, p, K)
        if c is None:
            continue
        crops.append(c[0]); sc.append(RF.scal(c[1], K[0, 0], p)); rows.append((p, g[k]))
        # orientation crop on the 1280x720 sim frame (w2D in 512-wide full-frame px = w_proj * 512/1280)
        SW, SH = sx.S['sim_wh']
        z = float(np.clip(95.0 / (c[1] * 512 / SW), 1.0, sx.S['sim_zmax'])); w = SW / z; h = w * 288 / 512
        x0 = float(np.clip(c[2] - w / 2, 0, SW - w)); y0 = float(np.clip(c[3] - h / 2, 0, SH - h))
        sc_ = 512.0 / w
        Mx = np.array([[sc_, 0, (0.5 - x0) * sc_ - 0.5], [0, sc_, (0.5 - y0) * sc_ - 0.5]], np.float64)
        oc.append(cv2.warpAffine(img, Mx, (512, 288), flags=cv2.INTER_AREA if sc_ < 1 else cv2.INTER_LINEAR))
        Kc = OO.crop_K(K, x0, y0, sc_); q = Kc @ p[:3]
        info.append(dict(seq=r['seq_id'], frame=r['frame_id'], conf=float(r['conf'][pi]), box=p, K_crop=Kc, uc=q[0] / q[2], vc=q[1] / q[2], zoom=z))
print('sim val boxes', len(rows))
# (a) refiner stage 2
dev = torch.device('cuda'); nets = []
for ck in ref_ck:
    net = RF.Net().to(dev); net.load_state_dict(torch.load(ck, map_location=dev)); nets.append(net.eval())
dh = []
with torch.no_grad():
    for b in range(0, len(crops), 128):
        x = RF.dt.prep(np.stack(crops[b:b + 128]), None, False).to(dev); s = torch.from_numpy(np.array(sc[b:b + 128], np.float32)).to(dev)
        dh += list(np.mean([net.fwd(x, s)[0].float().cpu().numpy() for net in nets], 0))
e_ir = np.array([a - np.log(g[2] / p[2]) for a, (p, g) in zip(dh, rows)])
# (b) students on orientation crops
np.save(OO.W + '/oc_simval.npy', np.stack(oc)); pickle.dump(dict(info=info, size=np.zeros(3)), open(OO.W + '/oc_simval.pkl', 'wb'))
OO.RN = (np.array(sx.S['sim_norm'][0], np.float32), np.array(sx.S['sim_norm'][1], np.float32))   # sim-domain input norm (source images)
OO.cmd_read('simval', O + '/wsim_read.pkl', stu)
R = pickle.load(open(O + '/wsim_read.pkl', 'rb'))
e_kp = np.full(len(rows), np.nan)
for i, (f, (p, g)) in enumerate(zip(info, rows)):
    s = OO.ensemble(R['reads'], R['order'], i, R['cells'][i], f['K_crop'], np.abs(p[3:6]))
    if s is not None and s[1][2] > 0:
        e_kp[i] = np.log(s[1][2] / g[2])
ok = np.isfinite(e_kp)
rob = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))
s_ir, s_kp = rob(e_ir[ok]), rob(e_kp[ok])
w = s_ir ** 2 / (s_ir ** 2 + s_kp ** 2)
res = dict(w=w, s_ir=s_ir, s_kp=s_kp, n=int(ok.sum()), n_boxes=len(rows), bias_ir=float(np.median(e_ir[ok])), bias_kp=float(np.median(e_kp[ok])),
           corr=float(np.corrcoef(e_ir[ok] - np.median(e_ir[ok]), e_kp[ok] - np.median(e_kp[ok]))[0, 1]))
json.dump(res, open(out, 'w'), indent=1); print(res)
