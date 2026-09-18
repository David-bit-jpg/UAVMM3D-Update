# -*- coding: utf-8 -*-
"""定位实验之四（2026-09-18）：在仿真图上「抹掉桨叶环带」，看能否复现真实域那 1.28 倍的深度偏移。

背景：录制器把网格整体外框（含摊开的静止桨叶）缩放到【电机对角距】规格值，所以仿真小机的机身只有标称的
0.41~0.64（avata2 0.41 / mavic-mini 0.48 / unk3 0.60 / phantom4 0.64；大机型 1.02 未被压）。
真实相机下桨叶转起来几乎看不见，可见部分就是机身 = 标称尺寸。
所以两域「标注框内部的结构分布」不同：仿真的框被机身+静止桨叶填满，真实的框只有机身。

做法（不需要重采）：用真值 2D 框，把「机身圈（core 比例）以外、框以内」的环带用周围背景填掉，
近似「桨叶看不见」，再看模型的深度比怎么变：
    偏移从 ~1.0 明显升到 ~1.2~1.3  -> 复现了真实域的偏移，根因确认是这条结构差异
    几乎不变                       -> 不是这条，别去改资产

    python annot_audit/prop_ablation.py [--cores 0.64,0.5] [--n 200]
"""
import argparse
import os
import sys

import cv2
import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(TOOLS)
import eval_2d as E   # noqa: E402
from zoom_response import build_model, frames_sim, infer_one   # noqa: E402

C = 'cfgs/models/uavdet_3d/camnorm/'
M = '../output/models/uavdet_3d/camnorm/'
ARMS = {'Q2': (C + 'sim_pp_r34_ratio.yaml', M + 'sim_pp_r34_ratio/Q2/ckpt/best.pth'),
        'A': (C + 'sim_full_r34_ratio_t.yaml', M + 'sim_full_r34_ratio_t/A/ckpt/best.pth'),
        'C': (C + 'sim_full_r34_ratio_tr.yaml', M + 'sim_full_r34_ratio_tr/C/ckpt/best.pth')}


def erase_region(img, box, inner, outer):
    """把 [inner*框, outer*框] 之间的区域用背景中位色填掉。inner=0 即整块填掉；outer 可 >1（抹框外背景）。"""
    x0, y0, x1, y1 = box
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = x1 - x0, y1 - y0
    out = img.copy()
    # 背景估计：框外 1.6 倍环带的中位颜色（逐通道）
    bx0, by0 = int(max(0, cx - 0.8 * w)), int(max(0, cy - 0.8 * h))
    bx1, by1 = int(min(img.shape[1], cx + 0.8 * w)), int(min(img.shape[0], cy + 0.8 * h))
    patch = img[by0:by1, bx0:bx1].reshape(-1, 3)
    if len(patch) < 10:
        return None
    bg = np.median(patch, axis=0)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(mask, (int(cx - outer * w / 2), int(cy - outer * h / 2)),
                  (int(cx + outer * w / 2), int(cy + outer * h / 2)), 1, -1)
    if inner > 0:
        cv2.rectangle(mask, (int(cx - inner * w / 2), int(cy - inner * h / 2)),
                      (int(cx + inner * w / 2), int(cy + inner * h / 2)), 0, -1)
    m3 = mask.astype(bool)
    if m3.sum() < 4:
        return None
    out[m3] = bg
    # 边界羽化，避免硬边成为新线索
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    edge = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - cv2.erode(mask, np.ones((3, 3), np.uint8))
    out[edge.astype(bool)] = blur[edge.astype(bool)]
    return out


def flatten_target(img, box, mode):
    """只改标注框内部的「材质/纹理」，保持形状和轮廓：median = 中值滤波去细纹理；gray = 去色；flat = 贴平均色但保留边缘亮度结构。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    out = img.copy()
    sub = out[y0:y1, x0:x1]
    if mode == 'median':
        out[y0:y1, x0:x1] = cv2.medianBlur(sub, 5)
    elif mode == 'gray':
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        out[y0:y1, x0:x1] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    elif mode == 'lowpass':
        out[y0:y1, x0:x1] = cv2.GaussianBlur(sub, (0, 0), 1.5)
    return out


def measure(arm, spec, n):
    cfgf, ckpt = ARMS[arm]
    model, _, cfg = build_model(cfgf, ckpt)
    f_ref = float(cfg.DATA_CONFIG.get('DEPTH_F_REF', 512))
    offs = []
    for img, K, boxes in frames_sim(n):
        boxes = boxes[(np.abs(boxes).sum(1) > 0) & (boxes[:, 2] > 0)]
        if not len(boxes):
            continue
        g = boxes[0]
        gb = E.box2d(E.corners_of(g), K, 512, 288)
        if gb is None:
            continue
        if spec is None:
            im2 = img
        elif isinstance(spec, str):
            im2 = flatten_target(img, gb, spec)
        else:
            im2 = erase_region(img, gb, spec[0], spec[1])
        if im2 is None:
            continue
        pred, conf = infer_one(model, cfg, im2, K, 512, 288)
        if not len(pred):
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
    ap.add_argument('--cores', default='none')
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    print('%s：仿真 test 上抹掉「机身圈以外、框以内」的环带后的深度偏移（真实域现状 1.27~1.28）' % a.arm)
    SPECS = [(None, '原图（什么都不抹）'),
             ((0.64, 1.0), '抹 框内 0.64~1.0 环带（原消融：目标外圈 + 紧贴的背景）'),
             ((1.0, 1.5), '抹 框外 1.0~1.5 环带（只动背景，目标完整）'),
             ((1.5, 3.0), '抹 框外 1.5~3.0（更远的背景）'),
             ((0.0, 1.0), '抹 整个框内（目标全没了，只剩背景）'),
             ((1.0, 6.0), '抹 框外全部（只剩目标，没有背景）'),
             ('median', '目标内部中值滤波（去细纹理，保形状）'),
             ('lowpass', '目标内部低通模糊（保形状）'),
             ('gray', '目标内部去色（只去颜色）')]
    for spec, tag in SPECS:
        o = measure(a.arm, spec, a.n)
        if len(o) < 20:
            print('  %-28s 命中太少 %d' % (tag, len(o)))
            continue
        print('  %-28s n=%3d  偏移 %.2f 倍（p25 %.2f p75 %.2f）' % (
            tag, len(o), np.exp(np.median(o)), np.exp(np.percentile(o, 25)), np.exp(np.percentile(o, 75))), flush=True)


if __name__ == '__main__':
    main()
