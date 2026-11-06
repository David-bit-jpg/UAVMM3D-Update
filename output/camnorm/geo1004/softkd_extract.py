# -*- coding: utf-8 -*-
"""软剪影蒸馏（PREREG_softkd.md）第 1 步：与 dc_extract.py 完全相同的样本与取窗，另存同一窗口的红外图裁窗（给红外教师网络）。
原说明：「第二段直接修正深度」（PREREG_dc.md）第 1 步：在仿真训练集上跑检测模型 P0_s{j}，与真值配对，取第二段的训练样本。
检测：irloc 配方的 P0_s{j}（与已有评测同一个类 CenterDetGeoIR），仿真 train 每 2 帧取 1（与 silw 同批帧），测试模式（不增广），pose_eval greedy、3D 距离 < 1 m 配对。
每个配对样本：
    窗口：预测 3D 框在原分辨率（1280×720）上的投影外接框中心、宽 w；与部署完全相同的取窗（大窗口 3w、256 像素 → 子窗口 √3·w → 128×128）
    彩色裁窗（uint8，BGR）；红外剪影裁窗（真值框 8 角点凸包内、比外扩环带亮的红外像素，同一取窗变换，0–255）
    标量：log(w / 焦距)、log(预测深度)、log(预测长宽高)（与分辨率无关，真实上也一样算）
    答案：δ = log(真值深度 / 预测深度)
只用 w 在 [6, 170] 像素内的检测（与部署同一范围）。写 dc/train_s{j}_{rgb,sil}.npy + _meta.pkl。"""
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
import cropw_extract as ce  # noqa: E402  （导入时会切到 tools 目录）
import silw_train as sw  # noqa: E402
from easydict import EasyDict  # noqa: E402
import uavdet3d.model.detectors as dets  # noqa: E402
from uavdet3d.config import cfg_from_yaml_file  # noqa: E402
from uavdet3d.datasets import build_dataloader  # noqa: E402
from uavdet3d.model import build_network  # noqa: E402
from uavdet3d.utils import common_utils, pose_eval as pe  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from center_det_geoir import CenterDetGeoIR  # noqa: E402

dets.__all__['CenterDetGeoIR'] = CenterDetGeoIR
ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
OUT = G + '/softkd'
P0 = ROOT + '/output/models/uavdet_3d/camnorm/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_irloc/P0_s%d/ckpt/checkpoint_epoch_5.pth'
CFG = 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_irloc.yaml'


def proj_box(g, K):
    c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
    q = (K @ c.T).T
    if (q[:, 2] <= 1e-3).any():
        return None, None
    uv = q[:, :2] / q[:, 2:3]
    return uv, (uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max())


def sil_full(ir, uv, bb):
    """全分辨率红外剪影：真值框 8 角点凸包内、比外扩 30% 环带均值高出一半对比度的像素（同 CenterDetGeoIR）；对比度不足返回 None。"""
    H, W = ir.shape
    u0, u1, v0, v1 = bb
    bw, bh = u1 - u0, v1 - v0
    X0, Y0, X1, Y1 = [int(round(x)) for x in (u0, v0, u1, v1)]
    E0, F0, E1, F1 = [int(round(x)) for x in (u0 - 0.3 * bw, v0 - 0.3 * bh, u1 + 0.3 * bw, v1 + 0.3 * bh)]
    X0, Y0, X1, Y1 = max(X0, 0), max(Y0, 0), min(X1, W), min(Y1, H)
    E0, F0, E1, F1 = max(E0, 0), max(F0, 0), min(E1, W), min(F1, H)
    hull = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(uv.astype(np.int32)), 1)
    x = ir.astype(np.float32) / 255.0
    ring = x[F0:F1, E0:E1].copy()
    ring[Y0 - F0:Y1 - F0, X0 - E0:X1 - E0] = np.nan
    rv = ring[~np.isnan(ring)]
    hv = x[hull > 0]
    if rv.size < 8 or hv.size < 4:
        return None
    rm, q90 = rv.mean(), np.quantile(hv, 0.9)
    if q90 - rm < 0.03:
        return None
    return (((x > rm + 0.5 * (q90 - rm)) & (hull > 0)) * 255).astype(np.uint8)


def run(j, limit=0):
    t0 = time.time()
    logger = common_utils.create_logger(OUT + '/extract_log.txt')
    cfg = EasyDict()
    cfg_from_yaml_file(CFG, cfg)
    dc = cfg.DATA_CONFIG
    dc.DATA_SPLIT['test'] = 'train'
    dc.SAMPLED_INTERVAL['test'] = 2
    ds, loader, _ = build_dataloader(dataset_cfg=dc, batch_size=8, dist=False, workers=0, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    nl, nt = model.load_params_from_file(filename=P0 % j, to_cpu=False, logger=logger)
    model.cuda().eval()
    if limit:
        ds.valid_idx = ds.valid_idx[:limit]
    recs = pe.run_inference(model, loader)
    print('s%d 推理 %d 帧 %.0fs' % (j, len(recs), time.time() - t0), flush=True)
    del model
    torch.cuda.empty_cache()
    base = '@@MMCACHE@@/full_v2/train'
    d = pickle.load(open(base + '/index.pkl', 'rb'))
    key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(d['metas'])}
    mm = {k: (np.memmap('%s/%s.bin' % (base, k), dtype=np.uint8, mode='r'), np.load('%s/%s_index.npy' % (base, k))) for k in ('rgb_jpg', 'ir_jpg')}
    rgbs, sils, irs, meta = [], [], [], []
    for r in recs:
        gt, pred, conf = r['gt'], r['pred'], r['conf']
        if len(gt) == 0 or len(pred) == 0:
            continue
        used = np.zeros(len(gt), bool)
        pairs = []
        for pi in np.argsort(-conf):
            dd = np.linalg.norm(gt[:, :3] - pred[pi, :3], axis=1)
            dd[used] = np.inf
            k = int(np.argmin(dd))
            if dd[k] < 1.0:
                used[k] = True
                pairs.append((pi, k))
        if not pairs:
            continue
        i = key[(r['seq_id'], r['frame_id'])]
        m = d['metas'][i]
        K = np.asarray(m['K_in'], np.float64)
        o, ln = mm['rgb_jpg'][1][i]
        img = cv2.imdecode(np.asarray(mm['rgb_jpg'][0][int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
        o, ln = mm['ir_jpg'][1][i]
        ir = cv2.imdecode(np.asarray(mm['ir_jpg'][0][int(o):int(o) + int(ln)]), cv2.IMREAD_GRAYSCALE)
        for pi, k in pairs:
            p, g = pred[pi], gt[k]
            uvp, bp = proj_box(p, K)
            uvg, bg = proj_box(g, K)
            if bp is None or bg is None or p[2] <= 1e-3:
                continue
            w = bp[1] - bp[0]
            if not (6.0 <= w <= 170.0):
                continue
            cx, cy = 0.5 * (bp[0] + bp[1]), 0.5 * (bp[2] + bp[3])
            patch = ce.window(img, cx, cy, ce.KW * w)
            v = sw.view_img(patch, S0, 0.0, 0.0, 1.0, w, False)
            s = sil_full(ir, uvg, bg)
            sv = sw.view_img(ce.window(s, cx, cy, ce.KW * w), S0, 0.0, 0.0, 1.0, w, False) if s is not None else np.zeros((128, 128), np.uint8)
            rgbs.append(v)
            irs.append(sw.view_img(ce.window(ir, cx, cy, ce.KW * w), S0, 0.0, 0.0, 1.0, w, False))
            sils.append(sv)
            meta.append(dict(seq=r['seq_id'], frame=r['frame_id'], w=float(w), lw=float(np.log(w / K[0, 0])), lz=float(np.log(p[2])),
                             ldims=[float(x) for x in np.log(np.maximum(p[3:6], 1e-3))], delta=float(np.log(g[2] / p[2])),
                             has_sil=bool(s is not None)))
    tag = 'train_s%d' % j + ('_smoke' if limit else '')
    np.save('%s/%s_rgb.npy' % (OUT, tag), np.stack(rgbs))
    np.save('%s/%s_sil.npy' % (OUT, tag), np.stack(sils))
    np.save('%s/%s_ir.npy' % (OUT, tag), np.stack(irs))
    pickle.dump(meta, open('%s/%s_meta.pkl' % (OUT, tag), 'wb'))
    dl = np.array([m['delta'] for m in meta])
    print('s%d 完成：%d 个样本（有红外剪影 %d），深度修正量中位 %+.3f、绝对值中位 %.3f，%.0fs' % (
        j, len(meta), sum(m['has_sil'] for m in meta), np.median(dl), np.median(np.abs(dl)), time.time() - t0), flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lim = int(os.environ.get('DC_LIMIT', '0'))
    for j in [int(x) for x in sys.argv[1:]] or [0, 1, 2]:
        run(j, lim)
