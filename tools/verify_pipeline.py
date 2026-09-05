# -*- coding: utf-8 -*-
"""逐步验证仿真（mm 缓存）与真实（MAV6D）两条数据链路：坐标变换、对齐、编解码。

每一项都给数字（PASS/FAIL）并落一张图到 output/verify/{sim,real}/，肉眼能对上才算过。

仿真链路（需要 build_mm_cache.py 的缓存）：
    S1  角点 -> 9 参数 -> 角点 往返（旋转从三条正交边解出，误差应 ~1e-6 m）
    S2  编码器 -> 解码器 往返（位置/旋转误差应 ~1e-6）
    S3  热图峰值落在 GT 中心格子上；图：缩放后的 RGB + GT 框 + 热图叠加
    S4  LiDAR 深度图在 GT 框内的中位深度 vs 标注深度（只看噪声 0 的序列）；tag 像素落在 GT 框内的比例
    S5  IR 对齐残差：最近合格目标在 IR 上的投影(含平移) vs RGB 投影，应 ~0 px；其余目标给出视差残差
真实链路（MAV6D）：
    R1  编码器 -> 解码器 往返（带畸变）
    R2  热图峰值落在 GT 中心格子上；图：缩放后图像 + GT 框 + 热图叠加
    R3  畸变对投影的影响量（k1=-0.23 有多大）
    R4  跨域 MAX_DIS 重标定：center_dis 输出应恰好缩放 src/dst 倍
    R5  外参：GT 投影落在画面内的比例

用法：
    python tools/verify_pipeline.py --sim-cache E:/mmcache/mm20 --n 12
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mm_cache import corners_to_9params, cam_to_world, world_to_pixel   # noqa: E402
from uavdet3d.config import cfg_from_yaml_file                                  # noqa: E402
from uavdet3d.datasets import build_dataloader                                  # noqa: E402
from uavdet3d.utils import common_utils, frame_convention                       # noqa: E402
from uavdet3d.utils.object_encoder_mav6d import center_point_encoder, center_point_decoder  # noqa: E402

GREEN, RED, CYAN, WHITE = (60, 220, 60), (40, 40, 255), (255, 220, 0), (255, 255, 255)
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print('  [%s] %-6s %s' % ('PASS' if ok else 'FAIL', name, detail))


def proto_corners(p9, seq):
    c = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
    return (c * p9[3:6]) @ R.from_euler(seq, p9[6:9]).as_matrix().T + p9[:3]


def draw_corners(img, pts, K, color, th=1, dist=None):
    if np.any(pts[:, 2] <= 1e-3):
        return
    if dist is not None:
        uv, _ = cv2.projectPoints(pts.astype(np.float64), np.zeros(3), np.zeros(3), K, np.asarray(dist).reshape(1, -1))
        uv = uv.reshape(-1, 2)
    else:
        uv = (K @ pts.T).T
        uv = uv[:, :2] / uv[:, 2:3]
    uv = np.round(uv).astype(int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, th, cv2.LINE_AA)


def heat_overlay(img_bgr, hm2d):
    """hm2d: (h, w) in [0,1] at stride 8 -> upsample and blend (JET) on the image."""
    h, w = img_bgr.shape[:2]
    up = cv2.resize(hm2d, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((np.clip(up, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(up, 0, 1)[..., None] * 0.7
    return (img_bgr * (1 - alpha) + color * alpha).astype(np.uint8)


def roundtrip_enc_dec(K, dist, raw_wh, new_wh, stride, classes, seq, boxes_cls, use_dist):
    hm, res, dis, dim, rt = center_point_encoder(boxes_cls, [K], [np.eye(4)], [dist], new_wh[0], new_wh[1],
                                                 raw_wh[0], raw_wh[1], stride, 1, classes, 2,
                                                 use_distortion=use_dist, rot_repr='euler6', euler_seq=seq)
    if hm.max() <= 0:
        return None
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    pred, conf = center_point_decoder(t(hm), t(res), t(dis), t(dim), t(rt), [K], [np.eye(4)], [dist],
                                      new_wh[0], new_wh[1], raw_wh[0], raw_wh[1], stride, 1,
                                      max_num=len(boxes_cls), use_distortion=use_dist, rot_repr='euler6', euler_seq=seq)
    return np.asarray(pred)


# ============================== 仿真 ============================== #
def verify_sim(args):
    out = os.path.join(args.out, 'sim')
    os.makedirs(out, exist_ok=True)
    print('\n===== 仿真链路 (%s) =====' % args.sim_cache)
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/mmcache/teacher.yaml', cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.sim_cache
    cfg.DATA_CONFIG.MODALITIES = ['rgb', 'ir', 'depth', 'tag']
    logger = common_utils.create_logger()
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0,
                                logger=logger, training=False)
    seq_name = frame_convention.get_default_euler_seq()
    classes = list(cfg.DATA_CONFIG.CLASS_NAMES)
    stride = int(cfg.DATA_CONFIG.STRIDE)
    picks = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).round().astype(int)
    root = args.sim_root

    # ---- S1: 角点 -> 9 参数 -> 角点 ----
    err1 = []
    for k in picks:
        m = ds.metas[int(ds.valid_idx[k])]
        p = os.path.join(root, m['seq'], 'boxes_rgb', os.path.splitext(m['frame'])[0] + '.pkl')
        if not os.path.exists(p):
            continue
        for row in pickle.load(open(p, 'rb')):
            c = np.array(row[1:] if isinstance(row[0], str) else row, dtype=np.float64).reshape(8, 3)
            p9 = corners_to_9params(c, seq_name)
            rb = proto_corners(p9, seq_name)
            # 按【集合】比：真实框先列顶面 4 点、原型先列底面 4 点（映射 4,5,6,7,0,1,2,3），
            # 逐槽位比会把纯顺序差异误报成几何误差。解码器永远按原型重建，顺序无关。
            D = np.linalg.norm(c[:, None, :] - rb[None, :, :], axis=2)
            err1.append(D.min(axis=1).max())
    err1 = np.array(err1)
    check('S1', len(err1) > 0 and err1.max() < 1e-3,
          '角点->9参数->角点(集合匹配) 最大误差 %.2e m（%d 个框；角点顺序与原型差一个顶/底面交换，属预期）'
          % (err1.max() if len(err1) else -1, len(err1)))

    # ---- S2: 编码 -> 解码 ----
    perr, aerr = [], []
    for k in picks:
        m = ds.metas[int(ds.valid_idx[k])]
        b = m['boxes9d'].astype(np.float64)
        cls = np.array([classes.index(n) for n in m['names']], dtype=np.float64)[:, None]
        pred = roundtrip_enc_dec(m['K_raw'].astype(np.float64), np.zeros(5), m['raw_wh'], (ds.W, ds.H), stride,
                                 classes, seq_name, np.c_[b, cls], use_dist=True)
        if pred is None:
            continue
        for g in b:
            d = np.linalg.norm(pred[:, :3] - g[:3], axis=1)
            j = int(np.argmin(d))
            perr.append(d[j])
            aerr.append(np.degrees((R.from_euler(seq_name, pred[j, 6:9]).inv() * R.from_euler(seq_name, g[6:9])).magnitude()))
    perr, aerr = np.array(perr), np.array(aerr)
    # 同一格子里有两个目标时解码只保留一个，允许少量匹配不上；看中位与 90% 分位
    check('S2', len(perr) > 0 and np.percentile(perr, 90) < 1e-3 and np.percentile(aerr, 90) < 1e-3,
          '编码->解码 位置误差 中位 %.2e m / 90%% %.2e m；旋转 中位 %.2e° / 90%% %.2e°（%d 个框）'
          % (np.median(perr), np.percentile(perr, 90), np.median(aerr), np.percentile(aerr, 90), len(perr)))

    # ---- S3 / S4 / S5：逐帧 ----
    peak_off, dep_err, tag_in, ir_res_near, ir_res_other = [], [], [], [], []
    sx, sy = ds.W / 1280.0, ds.H / 720.0
    seq_cache = {}
    for n, k in enumerate(picks):
        d = ds[int(k)]
        m = ds.metas[int(ds.valid_idx[k])]
        K = m['K_raw'].astype(np.float64)
        Ks = K.copy(); Ks[0] *= sx; Ks[1] *= sy
        img = d['image'][0]
        rgb = np.ascontiguousarray((img[:3].transpose(1, 2, 0) * 255).astype(np.uint8))
        ir = cv2.cvtColor((img[3] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        dep_m = img[4] * ds.depth_max
        tag = img[5] > 0
        hm = d['hm'][0]                              # (C, h, w)
        hmax = hm.max(axis=0)

        # S3 热图峰值 vs GT 中心
        boxes = m['boxes9d']
        for b in boxes:
            uv = (K @ b[:3])[:2] / b[2]
            gu, gv = uv[0] * sx / stride, uv[1] * sy / stride
            r0, c0 = int(gv), int(gu)
            if 0 <= r0 < hmax.shape[0] and 0 <= c0 < hmax.shape[1]:
                # 该格子应当是 1（高斯中心）
                peak_off.append(1.0 - hmax[r0, c0])
        # S4 深度 / tag
        gt_mask = np.zeros(hmax.shape[0] * stride and (ds.H, ds.W), np.uint8)
        for b in boxes:
            pts = proto_corners(b.astype(np.float64), seq_name)
            uv = (Ks @ pts.T).T; uv = uv[:, :2] / uv[:, 2:3]
            cv2.fillConvexPoly(gt_mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        gt_mask_d = cv2.dilate(gt_mask, np.ones((5, 5), np.uint8))
        if m['lidar_noise'] == 0.0:
            for b in boxes:
                pts = proto_corners(b.astype(np.float64), seq_name)
                uv = (Ks @ pts.T).T; uv = uv[:, :2] / uv[:, 2:3]
                mk = np.zeros((ds.H, ds.W), np.uint8)
                cv2.fillConvexPoly(mk, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
                # 只看框内【打在无人机上】(tag) 的像素：框多边形比无人机剪影大得多，
                # 里面大部分 LiDAR 点其实是几十米外的背景，取整框中位数会被背景淹没。
                vals = dep_m[(mk > 0) & tag & (dep_m > 0)]
                if len(vals) >= 3:
                    dep_err.append(abs(np.median(vals) - b[2]))
            if tag.sum() > 0:
                tag_in.append(float((tag & (gt_mask_d > 0)).sum() / tag.sum()))
        # S5 IR 对齐残差（几何：IR 投影 + 记录的平移 vs RGB 投影）
        if m['seq'] not in seq_cache:
            info = pickle.load(open(os.path.join(root, m['seq'], 'im_info.pkl'), 'rb'))
            seq_cache[m['seq']] = {mm: (np.array(info[mm]['intrinsic'], float), np.array(info[mm]['extrinsic'], float)) for mm in ('rgb', 'ir')}
        (K_rgb, E_rgb), (K_ir, E_ir) = seq_cache[m['seq']]['rgb'], seq_cache[m['seq']]['ir']
        shift = m['ir_shift'].astype(np.float64)
        zs = [b[2] for b, q in zip(boxes, m['qualified']) if q]
        znear = min(zs) if zs else None
        for b, q in zip(boxes, m['qualified']):
            cw = cam_to_world(b[None, :3].astype(np.float64), E_rgb)
            uv_r, _ = world_to_pixel(cw, E_rgb, K_rgb)
            uv_i, _ = world_to_pixel(cw, E_ir, K_ir)
            res = np.linalg.norm(uv_i[0] + shift - uv_r[0])
            (ir_res_near if (q and znear is not None and abs(b[2] - znear) < 1e-6) else ir_res_other).append(res)

        if n < args.n_img:
            vis_rgb = heat_overlay(rgb.copy(), hmax)
            for b in boxes:
                draw_corners(vis_rgb, proto_corners(b.astype(np.float64), seq_name), Ks, GREEN, 1)
                draw_corners(ir, proto_corners(b.astype(np.float64), seq_name), Ks, GREEN, 1)
            depv = cv2.applyColorMap((255 - np.clip(dep_m / ds.depth_max, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            depv[dep_m <= 0] = (25, 25, 25); depv[tag] = (255, 0, 255)
            for b in boxes:
                draw_corners(depv, proto_corners(b.astype(np.float64), seq_name), Ks, GREEN, 1)
            for p, t in ((vis_rgb, 'S3 RGB + GT box + heatmap  %s' % m['seq'].split('/')[3]),
                         (ir, 'S5 IR shifted %+.0f,%+.0f px; GT boxes from RGB' % (shift[0], shift[1])),
                         (depv, 'S4 LiDAR depth + tag(magenta)  noise %.1f m' % m['lidar_noise'])):
                cv2.putText(p, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(p, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            sep = np.full((ds.H, 3, 3), 90, np.uint8)
            top = np.concatenate([vis_rgb, sep, ir, sep, depv], 1)
            # 第二排：围绕最近合格目标的 4x 放大裁剪（三个模态同一窗口），肉眼核对是否对齐
            qb = [b for b, q in zip(boxes, m['qualified']) if q]
            if qb:
                b = min(qb, key=lambda x: x[2])
                cu = (Ks @ b[:3].astype(np.float64))[:2] / b[2]
                r = int(max(24, 1.2 * max(b[3:6]) * Ks[0, 0] / b[2]))
                x0, y0 = int(max(0, cu[0] - r)), int(max(0, cu[1] - r))
                x1, y1 = int(min(ds.W, cu[0] + r)), int(min(ds.H, cu[1] + r))
                crops = []
                for src in (vis_rgb, ir, depv):
                    cr = src[y0:y1, x0:x1]
                    if cr.size:
                        cr = cv2.resize(cr, (4 * (x1 - x0), 4 * (y1 - y0)), interpolation=cv2.INTER_NEAREST)
                        crops.append(cr)
                if crops:
                    hh = max(c.shape[0] for c in crops)
                    row = np.full((hh, top.shape[1], 3), 30, np.uint8)
                    x = 0
                    for c in crops:
                        row[:c.shape[0], x:x + c.shape[1]] = c
                        x += c.shape[1] + 12
                    cv2.putText(row, 'zoom x4 around nearest qualified target (%.1f m): RGB | IR | LiDAR' % b[2],
                                (6, hh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
                    top = np.concatenate([top, row], 0)
            cv2.imwrite(os.path.join(out, 'sim_%02d.jpg' % n), top, [cv2.IMWRITE_JPEG_QUALITY, 90])

    peak_off = np.array(peak_off)
    check('S3', len(peak_off) > 0 and np.percentile(peak_off, 95) < 1e-3,
          '热图在 GT 中心格子的值 1-hm 中位 %.2e / 95%% %.2e（%d 个框，应为 0）' % (np.median(peak_off), np.percentile(peak_off, 95), len(peak_off)))
    dep_err = np.array(dep_err)
    if len(dep_err):
        check('S4a', np.median(dep_err) < 1.0,
              '噪声 0 序列：框内 LiDAR 深度中位 vs 标注深度 |Δ| 中位 %.2f m / 90%% %.2f m（%d 个框）'
              % (np.median(dep_err), np.percentile(dep_err, 90), len(dep_err)))
    else:
        check('S4a', False, '采样里没有噪声为 0 且框内有点的帧，换 --n 更大再试')
    if tag_in:
        check('S4b', np.median(tag_in) > 0.8, 'tag 像素落在 GT 框(膨胀 2px)内的比例 中位 %.2f（%d 帧）' % (np.median(tag_in), len(tag_in)))
    ir_res_near = np.array(ir_res_near); ir_res_other = np.array(ir_res_other)
    check('S5', len(ir_res_near) > 0 and ir_res_near.max() < 1.5,
          'IR 平移后最近合格目标残差 最大 %.2f px（%d 个）；其余目标视差残差 中位 %.1f px / 最大 %.1f px（%d 个，属预期）'
          % (ir_res_near.max() if len(ir_res_near) else -1, len(ir_res_near),
             np.median(ir_res_other) if len(ir_res_other) else 0, ir_res_other.max() if len(ir_res_other) else 0, len(ir_res_other)))


# ============================== 真实 ============================== #
def verify_real(args):
    out = os.path.join(args.out, 'real')
    os.makedirs(out, exist_ok=True)
    print('\n===== 真实链路 (MAV6D) =====')
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/mav6d/centerdet.yaml', cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.mav6d
    logger = common_utils.create_logger()
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0,
                                logger=logger, training=True)
    seq_name = frame_convention.get_default_euler_seq()
    classes = list(cfg.DATA_CONFIG.CLASS_NAMES)
    stride = int(cfg.DATA_CONFIG.STRIDE)
    K = np.array(ds.intrinsic, dtype=np.float64)
    dist = np.array(ds.distortion_matrix, dtype=np.float64)
    raw_wh = (ds.raw_im_width, ds.raw_im_hight)
    new_wh = (ds.new_im_width, ds.new_im_hight)
    picks = np.linspace(0, len(ds) - 1, min(args.n * 20, len(ds))).round().astype(int)

    # ---- R1 编码->解码（带畸变）----
    perr, aerr, inside, dist_delta = [], [], [], []
    for k in picks:
        d = ds[int(k)]
        b = d['gt_box9d'].astype(np.float64)
        cls = np.array([classes.index(n) for n in d['gt_name']], dtype=np.float64)[:, None]
        pred = roundtrip_enc_dec(K, dist, raw_wh, new_wh, stride, classes, seq_name, np.c_[b, cls], use_dist=True)
        if pred is not None:
            for g in b:
                j = int(np.argmin(np.linalg.norm(pred[:, :3] - g[:3], axis=1)))
                perr.append(np.linalg.norm(pred[j, :3] - g[:3]))
                aerr.append(np.degrees((R.from_euler(seq_name, pred[j, 6:9]).inv() * R.from_euler(seq_name, g[6:9])).magnitude()))
        # R5 投影在画面内；R3 畸变影响
        for g in b:
            uv_d, _ = cv2.projectPoints(g[None, :3], np.zeros(3), np.zeros(3), K, dist.reshape(1, -1))
            uv_d = uv_d.reshape(2)
            uv_p = (K @ g[:3])[:2] / g[2]
            inside.append(0 <= uv_d[0] < raw_wh[0] and 0 <= uv_d[1] < raw_wh[1])
            dist_delta.append(np.linalg.norm(uv_d - uv_p))
    perr, aerr = np.array(perr), np.array(aerr)
    check('R1', len(perr) and perr.max() < 1e-3 and aerr.max() < 1e-3,
          '编码->解码(带畸变) 位置误差最大 %.2e m，旋转最大 %.2e°（%d 帧）' % (perr.max(), aerr.max(), len(perr)))
    check('R3', True, '畸变 k1=%.3f 对 GT 中心投影的影响：中位 %.1f px / 最大 %.1f px（不带畸变就会差这么多）'
          % (dist[0], np.median(dist_delta), np.max(dist_delta)))
    check('R5', np.mean(inside) > 0.99, 'GT 中心投影落在画面内 %.1f%%（%d 帧）' % (100 * np.mean(inside), len(inside)))

    # ---- R2 热图峰值 + 图 ----
    peak_off = []
    sx, sy = new_wh[0] / raw_wh[0], new_wh[1] / raw_wh[1]
    Ks = K.copy(); Ks[0] *= sx; Ks[1] *= sy
    for n, k in enumerate(picks[::20][:args.n_img]):
        d = ds[int(k)]
        img = np.ascontiguousarray((d['image'][0].transpose(1, 2, 0) * 255).astype(np.uint8))
        hm = d['hm'][0].max(axis=0)
        for g in d['gt_box9d'].astype(np.float64):
            uv_d, _ = cv2.projectPoints(g[None, :3], np.zeros(3), np.zeros(3), K, dist.reshape(1, -1))
            u, v = uv_d.reshape(2)
            r0, c0 = int(v * sy / stride), int(u * sx / stride)
            if 0 <= r0 < hm.shape[0] and 0 <= c0 < hm.shape[1]:
                peak_off.append(1.0 - hm[r0, c0])
        vis = heat_overlay(img.copy(), hm)
        for g in d['gt_box9d'].astype(np.float64):
            # 缩放后的图像上按【带畸变】投影画框：先在原分辨率投影，再按比例缩
            pts = proto_corners(g, seq_name)
            uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist.reshape(1, -1))
            uv = (uv.reshape(-1, 2) * [sx, sy]).round().astype(int)
            for a, b2 in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                cv2.line(vis, tuple(uv[a]), tuple(uv[b2]), GREEN, 1, cv2.LINE_AA)
            # 对照：不带畸变的投影（青色），看差多少
            draw_corners(vis, pts, Ks, CYAN, 1)
        cv2.putText(vis, 'R2 MAV6D resized %dx%d + heatmap; green=GT(with distortion) cyan=pinhole' % new_wh,
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, 'R2 MAV6D resized %dx%d + heatmap; green=GT(with distortion) cyan=pinhole' % new_wh,
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        big = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out, 'real_%02d.jpg' % n), big, [cv2.IMWRITE_JPEG_QUALITY, 90])
    peak_off = np.array(peak_off)
    check('R2', len(peak_off) and np.percentile(peak_off, 95) < 1e-3,
          '热图在 GT 中心格子的值 1-hm 中位 %.2e / 95%% %.2e（%d 帧）' % (np.median(peak_off), np.percentile(peak_off, 95), len(peak_off)))

    # ---- R4 MAX_DIS 重标定 ----
    if args.src_ckpt and os.path.exists(args.src_ckpt):
        from uavdet3d.model import build_network, load_data_to_gpu
        src_max_dis = args.src_max_dis
        x = torch.from_numpy(ds[int(picks[0])]['image'][None]).float().cuda()
        outs = []
        for rescale in (None, src_max_dis / float(cfg.DATA_CONFIG.MAX_DIS)):
            model = build_network(model_cfg=cfg.MODEL, dataset=ds)
            model.load_params_from_file(filename=args.src_ckpt, to_cpu=True, skip_patterns=['hm'], dis_rescale=rescale)
            model.cuda().eval()
            with torch.no_grad():
                bd = {'image': x}
                for mod in model.module_list:
                    bd = mod(bd)
                outs.append(bd['pred_center_dict']['center_dis'].cpu().numpy())
        want = src_max_dis / float(cfg.DATA_CONFIG.MAX_DIS)
        # 输出接近 0 的位置比值没有意义，改成绝对+相对容差直接比整张图
        max_dev = float(np.abs(outs[1] - outs[0] * want).max())
        ratio = outs[1] / np.where(np.abs(outs[0]) < 1e-3, np.nan, outs[0])
        ratio = ratio[np.isfinite(ratio)]
        # 权重在 float32 里缩放后再做卷积，与「先算再乘」不逐位相等，1e-3 量级的差是浮点累加误差
        check('R4', np.allclose(outs[1], outs[0] * want, rtol=2e-3, atol=2e-3),
              'center_dis 重标定：|out_rescaled - out*%.4f| 最大 %.2e，比值中位 %.4f（源 MAX_DIS %g -> 目标 %g）'
              % (want, max_dev, np.median(ratio) if len(ratio) else float('nan'), src_max_dis, cfg.DATA_CONFIG.MAX_DIS))
    else:
        check('R4', False, '未给 --src-ckpt，跳过')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-cache', default='E:/mmcache/mm20')
    ap.add_argument('--sim-root', default='E:/data_collect')
    ap.add_argument('--mav6d', default='E:/MAV6D')
    ap.add_argument('--src-ckpt', default=None, help='任意源域 ckpt，用来验 MAX_DIS 重标定')
    ap.add_argument('--src-max-dis', type=float, default=15.0)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--n-img', type=int, default=4)
    ap.add_argument('--out', default='../output/verify')
    ap.add_argument('--only', choices=['sim', 'real', 'both'], default='both')
    args = ap.parse_args()
    if args.only in ('sim', 'both'):
        verify_sim(args)
    if args.only in ('real', 'both'):
        verify_real(args)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print('\n===== 汇总: %d 项, %d 项未通过 =====' % (len(RESULTS), n_fail))
    for name, ok, detail in RESULTS:
        print('  %s %s' % ('PASS' if ok else 'FAIL', name))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
