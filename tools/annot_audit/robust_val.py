# -*- coding: utf-8 -*-
"""只用仿真的稳健性验证：在仿真验证集上对目标施加外观扰动，按扰动下的表现选轮（2026-09-18）。

为什么：D 臂 24 轮实测 —— 域内验证单调变好（score 0.066→0.196、角度 84.9→34.6），
真实域深度却从 0.395 退到 0.743。训得越久越贴合仿真那套「外观→大小」映射，而按域内分数选轮
恰好选中最贴合仿真的权重。不能用真实标签选轮，所以改用「目标被扰动后还准不准」这个纯仿真判据：
    扰动 = 标注框内 [core,1] 环带向局部背景衰减 + 目标区域运动模糊（与训练增广同族但更强）

    python annot_audit/robust_val.py --stem sim_full_r34_geo_tr --tag D24 [--interval 12]
逐个 checkpoint 打印：干净 / 扰动 下的位置中位与深度中位，最后给出按扰动分数选出的轮次。
"""
import argparse
import glob
import os
import re
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
from uavdet3d.utils import common_utils   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def np_(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def perturb(image, boxes, K, rng):
    """对每个目标：框内 [core,1] 环带向局部背景衰减 + 目标区域运动模糊。image: (C,H,W) 已归一化。"""
    from scipy.spatial.transform import Rotation as R
    C, H, W = image.shape
    nc = min(3, C)
    if not len(boxes):
        return image
    Rm = R.from_euler('xyz', boxes[:, 6:9]).as_matrix()
    for bi in range(len(boxes)):
        if boxes[bi, 2] <= 1e-6:
            continue
        cs = (PROTO8 * boxes[bi, 3:6]) @ Rm[bi].T + boxes[bi, :3]
        pr = (K @ cs.T).T
        if (pr[:, 2] <= 1e-6).any():
            continue
        uv = pr[:, :2] / pr[:, 2:3]
        x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
        bw, bh = x1 - x0, y1 - y0
        if bw < 6 or bh < 6:
            continue
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        core = float(rng.uniform(0.55, 0.85))
        ox0, ox1 = int(max(0, cx - 0.8 * bw)), int(min(W, cx + 0.8 * bw))
        oy0, oy1 = int(max(0, cy - 0.8 * bh)), int(min(H, cy + 0.8 * bh))
        if ox1 - ox0 < 4 or oy1 - oy0 < 4:
            continue
        bg = np.median(image[:nc, oy0:oy1, ox0:ox1].reshape(nc, -1), axis=1)
        m = np.zeros((H, W), np.uint8)
        cv2.rectangle(m, (int(x0), int(y0)), (int(np.ceil(x1)), int(np.ceil(y1))), 1, -1)
        cv2.rectangle(m, (int(cx - core * bw / 2), int(cy - core * bh / 2)),
                      (int(cx + core * bw / 2), int(cy + core * bh / 2)), 0, -1)
        mb = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
        image[:nc] = image[:nc] * (1 - mb) + bg[:, None, None] * mb
        ker = np.zeros((5, 5), np.float32)
        ang = rng.uniform(0, np.pi)
        for t_ in np.linspace(-2, 2, 10):
            ker[int(round(2 + t_ * np.sin(ang))), int(round(2 + t_ * np.cos(ang)))] = 1
        ker /= ker.sum()
        px0, px1 = int(max(0, x0 - 2)), int(min(W, x1 + 2))
        py0, py1 = int(max(0, y0 - 2)), int(min(H, y1 + 2))
        if px1 - px0 > 4 and py1 - py0 > 4:
            for c in range(nc):
                image[c, py0:py1, px0:px1] = cv2.filter2D(image[c, py0:py1, px0:px1], -1, ker)
    return image


def evaluate(model, loader, ds, do_perturb, seed=0):
    rng = np.random.default_rng(seed)
    pos, zerr = [], []
    with torch.no_grad():
        for batch in loader:
            gts = [np_(x).reshape(-1, 9) for x in batch['gt_box9d']]
            Ks = [np_(x).reshape(-1, 3, 3)[0] for x in batch['intrinsic']]
            if do_perturb:
                img = np.asarray(batch['image']).copy()
                for b in range(len(gts)):
                    g = gts[b][np.abs(gts[b]).sum(1) > 0]
                    img[b, 0] = perturb(img[b, 0], g, Ks[b], rng)
                batch['image'] = img
            load_data_to_gpu(batch)
            out = model(batch)
            for b in range(out['batch_size']):
                gt = gts[b][np.abs(gts[b]).sum(1) > 0]
                pred, conf = np_(out['pred_boxes9d'][b]), np_(out['confidence'][b])
                if not len(gt) or not len(pred):
                    continue
                K = Ks[b]
                for g in gt:
                    gb = E.box2d(E.corners_of(g), K, ds.new_im_width, ds.new_im_hight)
                    if gb is None:
                        continue
                    side = max(gb[2] - gb[0], gb[3] - gb[1])
                    hit = None
                    for j in np.argsort(-conf):
                        p = pred[j]
                        if p[2] > 0 and np.linalg.norm(E.proj(K, p[:3]) - E.proj(K, g[:3])) <= 0.5 * side:
                            hit = p
                            break
                    if hit is None:
                        continue
                    pos.append(float(np.linalg.norm(hit[:3] - g[:3])))
                    zerr.append(float(abs(hit[2] - g[2])))
    if not pos:
        return float('nan'), float('nan'), 0
    return float(np.median(pos)), float(np.median(zerr)), len(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='sim_full_r34_geo_tr')
    ap.add_argument('--tag', default='D24')
    ap.add_argument('--interval', type=int, default=12)
    a = ap.parse_args()
    cfgf = 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % a.stem
    cfg = EasyDict()
    cfg_from_yaml_file(cfgf, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.MAX_OBJ', '5'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = a.interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    ckdir = os.path.join('..', 'output', 'models', 'uavdet_3d', 'camnorm', a.stem, a.tag, 'ckpt')
    cks = sorted(glob.glob(os.path.join(ckdir, 'checkpoint_epoch_*.pth')),
                 key=lambda p: int(re.search(r'(\d+)\.pth$', p).group(1)))
    if not cks:
        print('没有找到 checkpoint：%s' % ckdir)
        return
    print('%s/%s：仿真验证集 %d 帧，干净 vs 目标被扰动' % (a.stem, a.tag, len(ds)))
    rows = []
    model = build_network(cfg.MODEL, ds)
    for ck in cks:
        ep = int(re.search(r'(\d+)\.pth$', ck).group(1))
        n, t = model.load_params_from_file(ck, to_cpu=False)
        assert n == t, (ck, n, t)
        model.cuda().eval()
        p0, z0, n0 = evaluate(model, loader, ds, False)
        p1, z1, n1 = evaluate(model, loader, ds, True)
        rows.append((ep, p0, z0, p1, z1))
        print('  第 %2d 轮 | 干净 位置 %.3f 深度 %.3f (n=%d) | 扰动 位置 %.3f 深度 %.3f (n=%d) | 扰动/干净 深度 %.2f' % (
            ep, p0, z0, n0, p1, z1, n1, z1 / max(z0, 1e-6)), flush=True)
    best = min(rows, key=lambda r: r[3])
    print('按【扰动下位置中位】选轮 -> 第 %d 轮（扰动 %.3f m，干净 %.3f m）' % (best[0], best[3], best[1]))


if __name__ == '__main__':
    main()
