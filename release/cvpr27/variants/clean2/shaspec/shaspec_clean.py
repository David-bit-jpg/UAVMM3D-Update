# -*- coding: utf-8 -*-
"""[clean2 item 4] ShaSpec orientation re-evaluated on the clean chain's orientation crops (det1 windows). Inference only (GPU, memory fraction 0.15,
batch 4); no training. Model build / AdaBN / corner reading = cmpeval/cmp_eval.py (kz kind) unchanged except:
  - crops = D:/S2MUAV3D/srcx_work/in2mav/oc_{val1,te1}.npy (clean det1 windows; same 512x288 BGR uint8 format and input norm as cmpeval crops);
  - reading centre uv = (uc, vc) = det1 box centre projected into the crop (cmp_eval: deployed dev-detector centre); K = K_crop; PnP size = s_bar of det1
    (= the box the clean students use for corner PnP);
  - AdaBN source = 2000 uniformly spaced clean unlabeled train crops oc_tr1 (cmp_eval used the development kzst6 crops_train_u8), batch 4 (was 16).
Sanity: raw (no AdaBN) seed 0 on the original cmpeval test crops must reproduce res_SHASPEC_s0.json test_raw (24.74 mean).
Writes per-frame rotations pf_<seed>_<split>_<raw|adabn>.npz (keys seq, frame, eul). Labels: only for the sanity number here; evaluation in eval_shaspec.py."""
import os, sys, json, time, pickle
os.environ.setdefault('GPU_FRAC', '0.15')
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval')
import cmp_eval as CE  # noqa  (imports torch, sets memory fraction, chdir tools)
import numpy as np, torch, cv2
from scipy.spatial.transform import Rotation as Rot
WK = 'D:/S2MUAV3D/srcx_work/in2mav'
CFGP = 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_shaspec.yaml'
CK = 'E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_shaspec/SHASPEC_s%d/ckpt/checkpoint_epoch_8.pth'
BS = 4


@torch.no_grad()
def adabn_clean(m, n=2000):
    crops = np.load(WK + '/oc_tr1.npy', mmap_mode='r'); idx = np.linspace(0, len(crops) - 1, n).astype(int)
    bb = m.backbone_2d if hasattr(m, 'backbone_2d') else m.backbone
    for mod in bb.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); mod.momentum = None
    m.eval(); bb.train()
    for c0 in range(0, n, BS):
        x = torch.from_numpy(np.stack([CE.to_in(crops[i]) for i in idx[c0:c0 + BS]])).cuda()
        bb({'image': x[:, None], 'batch_size': len(x)})
    m.eval()


def solve(c, K, lwh):
    for fl in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
        try:
            ok, rv, tv = cv2.solvePnP((CE.PROTO * lwh).reshape(-1, 1, 3), c.reshape(-1, 1, 2), K, np.zeros(5), flags=fl)
        except cv2.error:
            ok = False
        if ok:
            return cv2.Rodrigues(rv)[0]
    return None


@torch.no_grad()
def read(m, crops, uv_all, K_all, lwh_all):
    N = len(crops); E = np.full((N, 3), np.nan)
    for c0 in range(0, N, BS):
        ii = list(range(c0, min(c0 + BS, N)))
        x = torch.from_numpy(np.stack([CE.to_in(crops[i]) for i in ii])).cuda()
        bd = {'image': x[:, None], 'batch_size': len(ii)}
        for mod in m.module_list:
            bd = mod(bd)
        pc = bd['pred_center_dict']; hm = pc['hm'].float().cpu().numpy()[:, 0]; kp = pc['kp2d'].float().cpu().numpy()
        for j, i in enumerate(ii):
            uv = uv_all[i]; gi, gj = int(uv[1] // 8), int(uv[0] // 8)
            r0, r1, c0_, c1 = max(gi - 2, 0), min(gi + 3, hm.shape[1]), max(gj - 2, 0), min(gj + 3, hm.shape[2])
            if r0 >= r1 or c0_ >= c1:
                continue
            win = hm[j, r0:r1, c0_:c1]; a = np.unravel_index(np.argmax(win), win.shape)
            c = uv[None] + kp[j, :, r0 + a[0], c0_ + a[1]].reshape(8, 2) * 64.0
            R = solve(c, K_all[i], lwh_all[i])
            if R is not None:
                E[i] = Rot.from_matrix(R).as_euler('xyz')
    return E


def sanity(m):
    """raw seed-0 reading on the ORIGINAL cmpeval test crops with the cmp_eval rule (uv from M['t'], deployed lwh) -> mean sym angle"""
    crops = np.load(CE.O + '/test_crops_u8.npy', mmap_mode='r'); M = np.load(CE.O + '/test_meta.npz')
    q = np.einsum('nij,nj->ni', M['K'], M['t']); uv = q[:, :2] / q[:, 2:]
    E = read(m, crops, uv, M['K'], M['lwh'])
    a = [CE.sym(Rot.from_euler('xyz', e).as_matrix(), Rot.from_euler('xyz', g[6:9]).as_matrix()) if np.isfinite(e).all() else 90.0 for e, g, n in zip(E, M['gt'], M['n_gt']) if n > 0]
    return float(np.mean(a)), float(np.median(a))


if __name__ == '__main__':
    seeds = [int(s) for s in (sys.argv[1:] or ['0', '1'])]; log = {}
    for s in seeds:
        t0 = time.time(); m = CE.build(CFGP, CK % s)
        if s == 0:
            log['sanity_s0_test_raw_devcrops'] = sanity(m); print('sanity', log['sanity_s0_test_raw_devcrops'], flush=True)
        for tag in ('raw', 'adabn'):
            if tag == 'adabn':
                adabn_clean(m)
            for nm, sp in (('val1', 'val'), ('te1', 'test')):
                crops = np.load('%s/oc_%s.npy' % (WK, nm), mmap_mode='r'); I = pickle.load(open('%s/oc_%s.pkl' % (WK, nm), 'rb'))
                info = I['info']; lwh = np.tile(np.asarray(I['size'], np.float64), (len(info), 1))
                uv = np.array([[f['uc'], f['vc']] for f in info]); K = np.stack([f['K_crop'] for f in info])
                E = read(m, crops, uv, K, lwh)
                np.savez('%s/pf_s%d_%s_%s.npz' % (OUT, s, sp, tag), seq=np.array([f['seq'] for f in info]), frame=np.array([f['frame'] for f in info]), eul=E)
                print('seed %d %s %s: %d crops, %d solved, %.0fs' % (s, sp, tag, len(info), int(np.isfinite(E).all(1).sum()), time.time() - t0), flush=True)
        log['secs_s%d' % s] = time.time() - t0
        del m; torch.cuda.empty_cache()
    json.dump(log, open(OUT + '/shaspec_infer_%s.json' % '_'.join(map(str, seeds)), 'w'), indent=1)
