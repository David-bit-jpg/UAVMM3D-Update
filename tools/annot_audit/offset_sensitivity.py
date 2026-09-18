# -*- coding: utf-8 -*-
"""定位实验之三（2026-09-18）：那 1.27 倍的常数偏移对什么敏感。

已定位：真实图上模型对表观大小的响应是对的（帧内放大斜率 0.98，仿真 1.02），只是零点偏 1.27 倍
（等于把目标的表观大小低估 21%）。本实验对真实图做各种可逆的图像变换，看偏移动不动：
    某个光度变换能把偏移拉回 1.0  -> 零点由该外观因素决定（对症：仿真里随机化它）
    全都动不了                    -> 零点由形状/轮廓差异决定（对症：改资产，如桨叶）

    python annot_audit/offset_sensitivity.py [--n 200]
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
from zoom_response import build_model, frames_mav, frames_sim, infer_one   # noqa: E402


def tf_identity(img):
    return img


def tf_contrast(f):
    def g(img):
        m = img.reshape(-1, 3).mean(0)
        return np.clip((img.astype(np.float32) - m) * f + m, 0, 255).astype(np.uint8)
    return g


def tf_gamma(f):
    def g(img):
        x = (img.astype(np.float32) / 255.0) ** f
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return g


def tf_unsharp(amount):
    def g(img):
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        return np.clip(img.astype(np.float32) * (1 + amount) - blur.astype(np.float32) * amount, 0, 255).astype(np.uint8)
    return g


def tf_blur(sigma):
    def g(img):
        return cv2.GaussianBlur(img, (0, 0), sigma)
    return g


def tf_dilate(k):
    """把暗色目标「变胖」：灰度腐蚀（暗区扩张）k 像素，测「轮廓粗细」这条轴。"""
    def g(img):
        return cv2.erode(img, np.ones((k, k), np.uint8))
    return g


TFS = [('原图', tf_identity), ('对比度 x1.4', tf_contrast(1.4)), ('对比度 x0.7', tf_contrast(0.7)),
       ('gamma 0.7（提亮）', tf_gamma(0.7)), ('gamma 1.4（压暗）', tf_gamma(1.4)),
       ('锐化 +0.8', tf_unsharp(0.8)), ('高斯模糊 σ1.0', tf_blur(1.0)),
       ('暗区扩张 3px（轮廓变粗）', tf_dilate(3))]


def measure(gen, model, cfg, tf, oW=512, oH=288):
    f_ref = float(cfg.DATA_CONFIG.get('DEPTH_F_REF', 512))
    offs = []
    for img, K, boxes in gen:
        boxes = boxes[(np.abs(boxes).sum(1) > 0) & (boxes[:, 2] > 0)]
        if not len(boxes):
            continue
        g = boxes[0]
        im2 = tf(img)
        pred, conf = infer_one(model, cfg, im2, K, oW, oH)
        if not len(pred):
            continue
        gb = E.box2d(E.corners_of(g), K, oW, oH)
        if gb is None:
            continue
        side = max(gb[2] - gb[0], gb[3] - gb[1])
        j = None
        for cand in np.argsort(-conf):
            p = pred[cand]
            if p[2] > 0 and np.linalg.norm(E.proj(K, p[:3]) - E.proj(K, g[:3])) <= 0.5 * side:
                j = cand
                break
        if j is None:
            continue
        p = pred[j]
        f_in = float(np.sqrt(K[0, 0] * K[1, 1]))
        Lg, Lp = float(np.linalg.norm(g[3:6])), max(float(np.linalg.norm(p[3:6])), 1e-3)
        offs.append(np.log(p[2] * f_ref / f_in / Lp) - np.log(g[2] * f_ref / f_in / Lg))
    return np.array(offs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='Q2')
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    from zoom_response import ARMS
    real_cfg, sim_cfg, ckpt = ARMS[a.arm]
    print('%s：真实图做各种变换后，「机身倍数」的偏移（1.00 = 无偏，1.27 = 现状）' % a.arm)
    model, _, cfg = build_model(real_cfg, ckpt)
    for name, tf in TFS:
        o = measure(frames_mav(a.n), model, cfg, tf)
        if len(o) < 20:
            print('  %-24s 命中太少 %d' % (name, len(o)))
            continue
        print('  %-24s n=%3d  偏移 %.2f 倍（p25 %.2f p75 %.2f）' % (
            name, len(o), np.exp(np.median(o)), np.exp(np.percentile(o, 25)), np.exp(np.percentile(o, 75))), flush=True)
    print('对照：仿真 test 同样跑一遍（应当都在 1.0 附近）')
    model2, _, cfg2 = build_model(sim_cfg, ckpt)
    for name, tf in (TFS[0], TFS[1], TFS[6], TFS[7]):
        o = measure(frames_sim(a.n), model2, cfg2, tf)
        if len(o) >= 20:
            print('  %-24s n=%3d  偏移 %.2f 倍' % (name, len(o), np.exp(np.median(o))), flush=True)


if __name__ == '__main__':
    main()
