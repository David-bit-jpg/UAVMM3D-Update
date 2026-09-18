# -*- coding: utf-8 -*-
"""关键点头在两域的定位误差（2026-09-18）：绝对误差 / 去掉整体平移后的相对误差 / 隐含尺度。

为什么要分开看：
  * 绝对误差里含「整体平移」，它只影响位置的横向分量，不影响深度和朝向；
  * 深度与朝向由【相对形状】决定，所以要看去掉每个目标平均平移后的残差；
  * 隐含尺度 = 预测角点到中心的平均距离 / 真值的同一量 —— 它应当等于深度偏移的倒数，
    用来判断关键点这条路是否也带着「尺度被读小」的老毛病（size2d 头在真实域是宽 0.85 / 高 0.53）。
PnP 敏感度参考（annot_audit/check_kp2d.py 实测）：相对误差 0.5/1/2/4 px -> 位置 0.031/0.067/0.129/0.206 m。

    python annot_audit/kp_error.py --ckpt <权重> [--epochs 4,8,12,16]
"""
import argparse
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(TOOLS)
from scipy.spatial.transform import Rotation as R   # noqa: E402

import eval_2d as E   # noqa: E402
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import KP_SCALE   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
DOMAINS = (('MAV6D 真实', 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml', 4),
           ('仿真 test', 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp.yaml', 20))


def np_(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def measure(ckpt, cfgf, interval):
    cfg = EasyDict()
    cfg_from_yaml_file(cfgf, cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(ckpt, to_cpu=False)
    assert n == t, (ckpt, n, t)
    model.cuda().eval()
    seq = str(cfg.DATA_CONFIG.get('EULER_SEQ', 'xyz'))
    abs_e, rel_e, scale, box_px = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            gts = [np_(x).reshape(-1, 9) for x in batch['gt_box9d']]
            Ks = [np_(x).reshape(-1, 3, 3)[0] for x in batch['intrinsic']]
            stride = float(np_(batch['stride'])[0])
            load_data_to_gpu(batch)
            model(batch)
            kp = model.dense_head_2d.forward_loss_dict['pred_center_dict']['kp2d']
            B = len(gts)
            kp = kp.reshape(B, kp.shape[0] // B, 16, kp.shape[-2], kp.shape[-1])
            for b in range(B):
                gt = gts[b][np.abs(gts[b]).sum(1) > 0]
                if not len(gt):
                    continue
                K = Ks[b]
                for g in gt[:1]:
                    cs = (PROTO8 * g[3:6]) @ R.from_euler(seq, g[6:9]).as_matrix().T + g[:3]
                    pr = (K @ cs.T).T
                    if (pr[:, 2] <= 1e-6).any():
                        continue
                    uv_gt = pr[:, :2] / pr[:, 2:3]
                    ctr = E.proj(K, g[:3])
                    r, c = int(ctr[1] / stride), int(ctr[0] / stride)
                    if not (0 <= r < kp.shape[-2] and 0 <= c < kp.shape[-1]):
                        continue
                    off = np_(kp[b, 0, :, r, c]).astype(np.float64).reshape(8, 2) * KP_SCALE
                    uv_p = off + ctr[None, :]          # 用真值中心，隔离检测误差
                    d = np.linalg.norm(uv_p - uv_gt, axis=1)
                    shift = (uv_p - uv_gt).mean(0)
                    drel = np.linalg.norm((uv_p - shift) - uv_gt, axis=1)
                    abs_e.append(d.mean())
                    rel_e.append(drel.mean())
                    scale.append(np.linalg.norm(uv_p - uv_p.mean(0), axis=1).mean()
                                 / max(np.linalg.norm(uv_gt - uv_gt.mean(0), axis=1).mean(), 1e-6))
                    box_px.append(max(np.ptp(uv_gt[:, 0]), np.ptp(uv_gt[:, 1])))
    return np.array(abs_e), np.array(rel_e), np.array(scale), np.array(box_px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    a = ap.parse_args()
    print('权重 %s' % os.path.basename(a.ckpt))
    for name, cfgf, interval in DOMAINS:
        ae, re_, sc, bp = measure(a.ckpt, cfgf, interval)
        if len(ae) < 10:
            print('  %-10s 样本太少 %d' % (name, len(ae)))
            continue
        print('  %-10s n=%4d | 角点绝对误差 中位 %.2f px | 去掉平移后 %.2f px | 隐含尺度 %.2f (p25 %.2f p75 %.2f) | 真值框长边中位 %.0f px' % (
            name, len(ae), np.median(ae), np.median(re_), np.median(sc),
            np.percentile(sc, 25), np.percentile(sc, 75), np.median(bp)))


if __name__ == '__main__':
    main()
