# -*- coding: utf-8 -*-
"""定位实验之二（2026-09-18）：模型对「目标表观大小」的响应，两域同一协议。

已知：域内固定裁窗时，预测的机身倍数 t 对真值的斜率 0.96~0.99、corr 0.99（模型确实在量目标）；
真实域跨帧斜率 1.14、corr 0.74、整体偏 1.27 倍。本实验把同一帧按 s 倍围绕目标裁放（只放大，两域同协议），
真值 t 精确移动 −log s，看预测跟不跟：
    帧内斜率 ≈ 1 -> 真实图上也在跟着目标大小走，1.27 是固定偏移（标定/外观问题）
    帧内斜率 ≈ 0 -> 真实图上完全不跟，跨帧那 0.74 的相关性来自别的东西

    python annot_audit/zoom_response.py [--n 120] [--scales 1.0,1.3,1.6]
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(TOOLS)
import eval_2d as E   # noqa: E402
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import camera_geometry as cg   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402

M = '../output/models/uavdet_3d/camnorm/'
C = 'cfgs/models/uavdet_3d/camnorm/'
# 臂: (评测配置=真实域, 仿真域配置, 权重)
ARMS = {'Q2': (C + 'mav6d_r34_ratio_pp.yaml', C + 'sim_pp_r34_ratio.yaml', M + 'sim_pp_r34_ratio/Q2/ckpt/best.pth')}


def build_model(cfgf, ckpt, sets=None):
    cfg = EasyDict()
    cfg_from_yaml_file(cfgf, cfg)
    cfg_from_list((sets or []) + ['MODEL.POST_PROCESSING.MAX_OBJ', '5'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 1
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(ckpt, to_cpu=False)
    assert n == t, (n, t)
    model.cuda().eval()
    return model, ds, cfg


def frames_mav(n):
    root = 'E:/mmcache/mav6d_cn/test'
    idx = pickle.load(open(os.path.join(root, 'index.pkl'), 'rb'))
    rgb = np.load(os.path.join(root, 'rgb.npy'), mmap_mode='r')
    step = max(1, len(idx['valid_idx']) // n)
    for i in idx['valid_idx'][::step][:n]:
        m = idx['metas'][int(i)]
        yield np.ascontiguousarray(rgb[int(i)]), np.asarray(m['K_in']).reshape(3, 3), np.asarray(m['boxes9d']).reshape(-1, 9)


def frames_sim(n):
    root = 'E:/mmcache/full_v2/test'
    idx = pickle.load(open(os.path.join(root, 'index.pkl'), 'rb'))
    blob = np.memmap(os.path.join(root, 'rgb_jpg.bin'), dtype=np.uint8, mode='r')
    ji = np.load(os.path.join(root, 'rgb_jpg_index.npy'), mmap_mode='r')
    step = max(1, len(idx['valid_idx']) // n)
    for i in idx['valid_idx'][::step][:n]:
        m = idx['metas'][int(i)]
        o, ln = ji[int(i)]
        img = cv2.imdecode(np.asarray(blob[int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
        # 先按整幅缩放到网络输入（等效 z=1），与 MAV6D 缓存同口径，再走同一套放大协议
        K = cg.scale_K(np.asarray(m['K_in']).reshape(3, 3), 512.0 / img.shape[1], 288.0 / img.shape[0])
        img = cv2.resize(img, (512, 288), interpolation=cv2.INTER_AREA)
        yield img, K, np.asarray(m['boxes9d']).reshape(-1, 9)


def crop_zoom(img, K, center_uv, s, oW, oH):
    """围绕 center_uv 裁 (oW/s, oH/s) 的窗口再放大回 (oW, oH)；返回图与新 K。"""
    w, h = oW / s, oH / s
    x0 = float(np.clip(center_uv[0] - w / 2, 0, img.shape[1] - w))
    y0 = float(np.clip(center_uv[1] - h / 2, 0, img.shape[0] - h))
    xi, yi, wi, hi = int(round(x0)), int(round(y0)), int(round(w)), int(round(h))
    sub = img[yi:yi + hi, xi:xi + wi]
    if sub.shape[0] < 8 or sub.shape[1] < 8:
        return None, None
    out = cv2.resize(sub, (oW, oH), interpolation=cv2.INTER_AREA if wi > oW else cv2.INTER_LINEAR)
    K2 = cg.scale_K(cg.translate_K(K, -xi, -yi), oW / float(wi), oH / float(hi))
    return out, K2


def infer_one(model, cfg, img, K, oW, oH):
    nm = cfg.DATA_CONFIG.get('NORM_MEAN', None)
    ns = cfg.DATA_CONFIG.get('NORM_STD', None)
    x = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    if nm:
        x = (x - np.asarray(nm, np.float32)[:, None, None]) / np.asarray(ns, np.float32)[:, None, None]
    batch = {'image': x[None, None], 'intrinsic': np.asarray(K, np.float64)[None, None],
             'extrinsic': np.eye(4, dtype=np.float32)[None, None], 'distortion': np.zeros((1, 1, 5), np.float32),
             'raw_im_size': np.array([[oW, oH]]), 'new_im_size': np.array([[oW, oH]]),
             'stride': np.array([cfg.DATA_CONFIG.STRIDE]), 'batch_size': 1}
    with torch.no_grad():
        load_data_to_gpu(batch)
        out = model(batch)
    pred = np.asarray(out['pred_boxes9d'][0])
    conf = np.asarray(out['confidence'][0])
    return pred, conf


def run(domain, gen, model, cfg, scales, oW=512, oH=288):
    f_ref = float(cfg.DATA_CONFIG.get('DEPTH_F_REF', 512))
    slopes, offs, n_ok = [], [], 0
    for img, K, boxes in gen:
        boxes = boxes[(np.abs(boxes).sum(1) > 0) & (boxes[:, 2] > 0)]
        if not len(boxes):
            continue
        g = boxes[0]
        uv = E.proj(K, g[:3])
        xs, ys = [], []
        for s in scales:
            im2, K2 = crop_zoom(img, K, uv, s, oW, oH)
            if im2 is None:
                continue
            pred, conf = infer_one(model, cfg, im2, K2, oW, oH)
            if not len(pred):
                continue
            gb = E.box2d(E.corners_of(g), K2, oW, oH)
            if gb is None:
                continue
            side = max(gb[2] - gb[0], gb[3] - gb[1])
            j = None
            for cand in np.argsort(-conf):
                p = pred[cand]
                if p[2] > 0 and np.linalg.norm(E.proj(K2, p[:3]) - E.proj(K2, g[:3])) <= 0.5 * side:
                    j = cand
                    break
            if j is None:
                continue
            p = pred[j]
            f_in = float(np.sqrt(K2[0, 0] * K2[1, 1]))
            Lg = float(np.linalg.norm(g[3:6]))
            Lp = max(float(np.linalg.norm(p[3:6])), 1e-3)
            xs.append(np.log(g[2] * f_ref / f_in / Lg))
            ys.append(np.log(p[2] * f_ref / f_in / Lp))
        if len(xs) >= 3 and np.ptp(xs) > 0.2:
            k, b = np.polyfit(xs, ys, 1)
            slopes.append(k)
            offs.append(np.mean(np.array(ys) - np.array(xs)))
            n_ok += 1
    if not slopes:
        print('  %s: 有效帧太少' % domain)
        return
    sl, of = np.array(slopes), np.array(offs)
    print('  %-10s 有效帧 %3d | 帧内斜率 中位 %.2f (p25 %.2f p75 %.2f) | 帧内平均偏移 exp(中位) %.2f 倍' % (
        domain, n_ok, np.median(sl), *np.percentile(sl, [25, 75]), np.exp(np.median(of))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='Q2')
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--scales', default='1.0,1.25,1.5,1.8')
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(',')]
    real_cfg, sim_cfg, ckpt = ARMS[a.arm]
    print('%s：帧内「围绕目标放大 %s 倍」时，预测的机身倍数跟不跟真值走（斜率 1 = 完全跟上）' % (a.arm, a.scales))
    model, _, cfg = build_model(real_cfg, ckpt)
    run('MAV6D 真实', frames_mav(a.n), model, cfg, scales)
    model2, _, cfg2 = build_model(sim_cfg, ckpt)
    run('仿真 test', frames_sim(a.n), model2, cfg2, scales)


if __name__ == '__main__':
    main()
