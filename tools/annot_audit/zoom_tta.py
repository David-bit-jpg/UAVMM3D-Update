# -*- coding: utf-8 -*-
"""推理期多尺度平均（2026-09-18）：同一帧在几个裁窗倍数下各推一次，深度在对数域取中位，2D 中心用原尺度。

为什么可行：帧内「围绕目标放大」的响应斜率实测 0.98（仿真 1.02），即不同裁窗下预测的公制深度本该一致，
差异就是噪声。对数域平均只压方差、不动系统偏移，所以它能改善 <0.2m 这类比例指标，但不会修好尺度偏差。

    python annot_audit/zoom_tta.py --ckpt <权重> [--scales 1.0,1.2,1.4] [--interval 4]
"""
import argparse
import os
import sys

import numpy as np
from easydict import EasyDict

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(TOOLS)
import eval_2d as E   # noqa: E402
from zoom_response import build_model, crop_zoom, frames_mav, infer_one   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml')
    ap.add_argument('--scales', default='1.0,1.2,1.4')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--size-source', default='known')
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(',')]
    model, _, cfg = build_model(a.cfg, a.ckpt, ['MODEL.POST_PROCESSING.SIZE_SOURCE', a.size_source,
                                                'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'])
    oW, oH = 512, 288
    base, tta, zr_b, zr_t = [], [], [], []
    for img, K, boxes in frames_mav(a.n):
        boxes = boxes[(np.abs(boxes).sum(1) > 0) & (boxes[:, 2] > 0)]
        if not len(boxes):
            continue
        g = boxes[0]
        pred, conf = infer_one(model, cfg, img, K, oW, oH)
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
        p0 = pred[j]
        uv = E.proj(K, p0[:3])                       # 用【预测的】中心裁窗，评测时不碰真值
        zs = [float(p0[2])]
        for s in scales[1:]:
            im2, K2 = crop_zoom(img, K, uv, s, oW, oH)
            if im2 is None:
                continue
            pr2, cf2 = infer_one(model, cfg, im2, K2, oW, oH)
            if not len(pr2):
                continue
            k = int(np.argmax(cf2))
            if pr2[k][2] > 0 and np.linalg.norm(E.proj(K2, pr2[k][:3]) - uv * 0 - E.proj(K2, p0[:3])) < 2 * side * s:
                zs.append(float(pr2[k][2]))
        z_tta = float(np.exp(np.median(np.log(zs))))
        base.append(np.linalg.norm(p0[:3] - g[:3]))
        tta.append(np.linalg.norm(p0[:3] * (z_tta / p0[2]) - g[:3]))
        zr_b.append(p0[2] / g[2])
        zr_t.append(z_tta / g[2])
    b, t = np.array(base), np.array(tta)
    print('权重 %s | 尺度 %s | n=%d' % (os.path.basename(a.ckpt), a.scales, len(b)))
    for lab, arr, zr in (('单尺度', b, np.array(zr_b)), ('多尺度平均', t, np.array(zr_t))):
        print('  %-8s 位置中位 %.3f m | <0.2m %.1f%% | <0.5m %.1f%% | 深度比 中位 %.2f (p10 %.2f p90 %.2f, p90/p10 %.2f)' % (
            lab, np.median(arr), 100 * (arr < 0.2).mean(), 100 * (arr < 0.5).mean(),
            np.median(zr), np.percentile(zr, 10), np.percentile(zr, 90),
            np.percentile(zr, 90) / np.percentile(zr, 10)))


if __name__ == '__main__':
    main()
