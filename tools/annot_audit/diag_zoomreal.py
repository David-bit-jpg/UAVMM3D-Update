# -*- coding: utf-8 -*-
"""真实图上做「目标变大/变小」对照：把网络输入围绕目标裁放 z 倍（真实深度不变）。

模型若真的在读目标像素大小 -> 预测深度不随 z 变（深度比恒定）；
模型若只在吐训练先验 -> 预测的虚拟深度 Zv 恒定，即预测深度 ∝ z。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402
from easydict import EasyDict   # noqa: E402
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402


def zoomer(real_pp, z):
    def f(dd):
        img, K = dd['image'], dd['intrinsic'][0].astype(np.float64)
        boxes = dd['gt_box9d'].reshape(-1, 9)
        if z != 1.0 and len(boxes):
            H, W = img.shape[-2], img.shape[-1]
            uv = K @ boxes[0, :3].astype(np.float64)
            cu, cvv = uv[0] / uv[2], uv[1] / uv[2]
            w, h = W / z, H / z
            x0 = float(np.clip(cu - w / 2, 0, max(W - w, 0)))
            y0 = float(np.clip(cvv - h / 2, 0, max(H - h, 0)))
            M = np.array([[z, 0, -z * x0], [0, z, -z * y0]], np.float32)
            src = img[0].transpose(1, 2, 0)
            dst = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            dd['image'] = dst.transpose(2, 0, 1)[None].astype(img.dtype)
            S = np.array([[z, 0, -z * x0], [0, z, -z * y0], [0, 0, 1.0]])
            dd['intrinsic'] = np.array([S @ K], dtype=np.float64)
        return real_pp(dd)
    return f


def run(cfg_file, ckpt, z, interval):
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    ds.data_pre_processor = zoomer(ds.data_pre_processor, z)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(ckpt, to_cpu=False)
    assert n == t
    model.cuda().eval()
    return pose_eval.run_inference(model, loader)


def go(tag, recs):
    rows = []
    for r in recs:
        K = r['K']
        f = float(np.sqrt(K[0, 0] * K[1, 1]))
        for g in r['gt']:
            if len(r['pred']) == 0:
                continue
            uvg = (K @ g[:3])[:2] / g[2]
            d = [np.linalg.norm((K @ p[:3])[:2] / p[2] - uvg) for p in r['pred']]
            j = int(np.argmin(d))
            s_px, _ = U.proj_extent(g, K)
            rows.append((d[j], s_px, f, g[2], r['pred'][j][2]))
    a = np.array(rows)
    ok = a[:, 0] < 0.5 * a[:, 1]
    if ok.sum() < 5:
        print('%-26s 找到太少 %d' % (tag, ok.sum())); return
    print('%-26s n=%3d 找到%3d(%2.0f%%) | f %5.0f | 目标 s_px %5.1f | Z 真 %.2f 预测 %.2f | 深度比 %.2f | 预测 Zv %.2f' % (
        tag, len(a), ok.sum(), 100 * ok.mean(), np.median(a[ok, 2]), np.median(a[ok, 1]),
        np.median(a[ok, 3]), np.median(a[ok, 4]), np.median(a[ok, 4] / a[ok, 3]),
        np.median(a[ok, 4] * 512 / a[ok, 2])))


def main():
    mav = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
    for tag, ck in (('P1', '/sim_pp_realsize/P1/ckpt/best.pth'), ('S1', '/sim_indoor8_mz/S1/ckpt/best.pth')):
        print('\n%s · MAV6D 真实 test：围绕目标裁放 z 倍（真实深度不变，目标像素 x z）' % tag)
        for z in (0.6, 1.0, 1.6, 2.2):
            go('  z=%.1f' % z, run(mav, U.M + ck, z, 20))


if __name__ == '__main__':
    main()
