# -*- coding: utf-8 -*-
"""KZ 可视化（2026-09-22）：关键点网络 B 在放大裁窗上看到了什么、预测了什么。

对真实 test 抽若干帧：位置网络 A（发布版 ramp_s0 单模型，够定中心与 2D 宽）给出峰值与 w2d → 按 --kp2d-zoom 裁原图 →
B（可多种子平均）在裁窗上预测 8 个角点；图上画：红点 = B 预测角点（解 PnP 用的就是这 8 个点），绿线 = 真值 3D 框投到裁窗，
右下角 = 真值机头方向 / PnP 机头方向的俯视罗盘。每帧标裁窗倍数、角点误差、航向折180 误差。

    python annot_audit/kz_view.py --kp2d-from <权重...> [--kp2d-zoom 95] [--n 16] [--out ...]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)
import kp_zoom   # noqa: E402
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import camera_geometry as cg   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402

REAL = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'
PROTO8 = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kp2d-from', nargs='+', required=True)
    ap.add_argument('--kp2d-cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml')
    ap.add_argument('--kp2d-zoom', type=float, default=95.0)
    ap.add_argument('--raw-root', default='@@MAV6D@@')
    ap.add_argument('--ckpt', default=os.path.join(TOOLS, '..', 'output', 'release_20260920', 'ramp_s0.pth'))
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--interval', type=int, default=300)
    ap.add_argument('--seq-filter', default='', help='只看序列名含该子串的帧，如 mavic2')
    ap.add_argument('--from-dump', default=None, help='裁窗的中心/宽改从这个预测 dump 取（= 部署管线里集成+背景后的位置网络输出），不再跑单模型 A')
    ap.add_argument('--out', default=os.path.join(TOOLS, '..', 'output', 'camnorm', 'kz', 'kz_view.jpg'))
    a = ap.parse_args()
    cfg = EasyDict()
    cfg_from_yaml_file(REAL, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = a.interval
    logger = common_utils.create_logger()
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=logger, training=False)
    A = build_network(cfg.MODEL, ds)
    A.load_params_from_file(a.ckpt, to_cpu=False)
    A.cuda().eval()
    kcfg = EasyDict()
    cfg_from_yaml_file(a.kp2d_cfg, kcfg)
    kds, _, _ = build_dataloader(kcfg.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=logger, training=False)
    Bs = []
    for c in a.kp2d_from:
        m = build_network(kcfg.MODEL, kds)
        m.load_params_from_file(c, to_cpu=False)
        Bs.append(m.cuda().eval())
    src = kp_zoom.RawSource(ds, a.raw_root)
    dump = None
    if a.from_dump:
        import pickle
        Dd = pickle.load(open(a.from_dump, 'rb'))
        lwh_c = np.asarray(Dd['info']['calib_lwh'], np.float64)
        dump = {(f['seq'], f['frame']): f for f in Dd['frames']}
    nm = np.asarray(cfg.DATA_CONFIG.NORM_MEAN, np.float32).reshape(3, 1, 1)
    ns = np.asarray(cfg.DATA_CONFIG.NORM_STD, np.float32).reshape(3, 1, 1)
    stride = int(cfg.DATA_CONFIG.STRIDE)
    in_w, in_h = 512, 288
    tiles = []
    skipped = [0]
    with torch.no_grad():
        for batch in loader:
            if len(tiles) >= a.n:
                break
            if a.seq_filter and a.seq_filter not in str(batch['seq_id'][0]):
                continue
            load_data_to_gpu(batch)
            if dump is not None:
                fd = dump.get((str(batch['seq_id'][0]), str(batch['frame_id'][0])))
                if fd is None or fd['matched'] is None:
                    skipped[0] += 1
                    continue
                pb = fd['pred'][fd['matched']]
                Kd = np.asarray(fd['K_in'], np.float64)
                q = Kd @ pb[:3]
                u, v = float(q[0] / q[2]), float(q[1] / q[2])
                span = float(np.ptp(((PROTO8 * lwh_c) @ R.from_euler('xyz', pb[6:9]).as_matrix().T)[:, 0]))
                w2d = float(Kd[0, 0] * span / pb[2])
            else:
                bd = batch
                for mod in A.module_list:
                    bd = mod(bd)
                pc = bd['pred_center_dict']
                hm = torch.sigmoid(pc['hm'].float())[0, 0]
                idx = int(torch.argmax(hm.reshape(-1)))
                r, c = idx // hm.shape[1], idx % hm.shape[1]
                u = (c + float(pc['center_res'][0, 0, r, c])) * stride
                v = (r + float(pc['center_res'][0, 1, r, c])) * stride
                w2d = float(torch.exp(pc['size2d'][0, 0, r, c].float().clamp(-5, 9)))
            z = float(np.clip(a.kp2d_zoom / max(w2d, 1e-3), 1.0, 4.0))
            # 这里的 A 是单模型、没装时序背景（部署管线里有），偶尔会峰在顶灯上；那种帧裁出来的是灯，看不出 B 的行为，跳过
            g = batch['gt_box9d'][0].cpu().numpy().reshape(-1, 9)
            g = g[np.abs(g).sum(1) > 0][0]
            K_in = np.asarray(batch['intrinsic'][0].cpu() if hasattr(batch['intrinsic'][0], 'cpu') else batch['intrinsic'][0]).reshape(-1, 3, 3)[0]
            qg = K_in @ g[:3]
            if np.hypot(qg[0] / qg[2] - u, qg[1] / qg[2] - v) > 40:
                skipped[0] += 1
                continue
            img, K_raw = src.get(str(batch['seq_id'][0]), str(batch['frame_id'][0]))
            raw_h, raw_w = img.shape[:2]
            x0, y0, w, h = kp_zoom.crop_window(u, v, z, raw_w, raw_h, in_w, in_h)
            s = in_w / w
            M = np.array([[s, 0, (0.5 - x0) * s - 0.5], [0, s, (0.5 - y0) * s - 0.5]], np.float64)
            crop = cv2.warpAffine(img, M, (in_w, in_h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            x = (crop.astype(np.float32).transpose(2, 0, 1) / 255.0 - nm) / ns
            t = torch.from_numpy(x[None, None]).cuda()
            maps, hms = [], []
            for B in Bs:
                tb = {'image': t, 'batch_size': 1}
                for mod in B.module_list:
                    tb = mod(tb)
                maps.append(tb['pred_center_dict']['kp2d'].float()[0])
                hms.append(torch.sigmoid(tb['pred_center_dict']['hm'].float()[0, 0]))
            kp_map, hm_b = torch.stack(maps).mean(0), torch.stack(hms).mean(0)
            uc, vc = (u * raw_w / in_w - x0) * s, (v * raw_h / in_h - y0) * s
            cc, rc = int(uc / stride), int(vc / stride)
            r0, r1, c0, c1 = max(0, rc - 2), min(hm_b.shape[0], rc + 3), max(0, cc - 2), min(hm_b.shape[1], cc + 3)
            sub = hm_b[r0:r1, c0:c1]
            j = int(torch.argmax(sub.reshape(-1)))
            rb, cb = r0 + j // sub.shape[1], c0 + j % sub.shape[1]
            off = kp_map[:, rb, cb].cpu().numpy().reshape(8, 2) * kp_zoom.KP_SCALE
            ctr_b = np.array([(cb + 0.5) * stride, (rb + 0.5) * stride])
            pts = off + ctr_b
            # 真值投到裁窗
            Kc = cg.scale_K(cg.translate_K(K_raw, -x0, -y0), s, s)
            Rg = R.from_euler('xyz', g[6:9]).as_matrix()
            C = (PROTO8 * g[3:6]) @ Rg.T + g[:3]
            uv = C @ Kc.T
            uv = uv[:, :2] / uv[:, 2:]
            kp_err = float(np.linalg.norm(pts - uv, axis=1).mean())
            # PnP（与解码器同：对象点用自标定尺寸近似 → 这里直接用真值尺寸，只看旋转）
            obj = (PROTO8 * g[3:6]).reshape(-1, 1, 3)
            ok, rvec, tvec = cv2.solvePnP(obj, pts.reshape(-1, 1, 2).astype(np.float64), Kc, np.zeros((1, 5)), flags=cv2.SOLVEPNP_SQPNP)
            he = np.nan
            if ok:
                Rp = cv2.Rodrigues(rvec)[0]
                E = Rg.T @ Rp
                he = abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
                he = min(he, 180 - he)
            vis = crop.copy()
            p = np.round(uv).astype(int)
            for i, jx in EDGES:
                cv2.line(vis, tuple(p[i]), tuple(p[jx]), (60, 220, 60), 1, cv2.LINE_AA)
            for q in np.round(pts).astype(int):
                cv2.circle(vis, tuple(q), 4, (0, 0, 255), -1, cv2.LINE_AA)
            for k_ in range(4):        # 机头面（x=+0.5：编号 4..7）的预测角点圈一下
                cv2.circle(vis, tuple(np.round(pts[4 + k_]).astype(int)), 7, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.rectangle(vis, (0, 0), (in_w, 20), (0, 0, 0), -1)
            cv2.putText(vis, '%s/%s  zoom x%.2f  kp err %.1fpx  heading fold180 %s' % (
                str(batch['seq_id'][0]).split('/', 1)[1], str(batch['frame_id'][0]).split('.')[0][-6:], z, kp_err,
                ('%.0f deg' % he) if np.isfinite(he) else 'PnP fail'), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(vis)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('图：', os.path.abspath(a.out), '（红点 = B 预测角点，橙圈 = 机头面 4 个角点，绿线 = 真值框）；单模型 A 峰值离真值 >40 px 跳过 %d 帧' % skipped[0])


if __name__ == '__main__':
    main()
