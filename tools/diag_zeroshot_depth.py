# -*- coding: utf-8 -*-
"""零样本深度诊断：预测深度 / 真值深度、预测尺寸、2D 中心像素误差、表观大小，仿真验证集 vs MAV6D 验证集对照。

    python diag_zeroshot_depth.py --ckpt ../output/models/uavdet_3d/camnorm/sim_indoor8/S0/ckpt/best.pth
"""
import argparse
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402
from uavdet3d.utils import camera_geometry as cg   # noqa: E402


def analyse(name, cfg_file, split, interval, ckpt, match):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = split
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    model.load_params_from_file(ckpt, to_cpu=False)
    model.cuda().eval()
    recs = pose_eval.run_inference(model, loader)
    ratio, uv_err, pdim, gdim, app_px, f_in, zv_pred, zv_gt = [], [], [], [], [], [], [], []
    for r in recs:
        if len(r['gt']) == 0 or len(r['pred']) == 0:
            continue
        K = r['K']
        f = cg.focal(K)
        for g in r['gt']:
            d = np.linalg.norm(r['pred'][:, :3] / np.maximum(r['pred'][:, 2:3], 1e-6) - g[:3] / g[2], axis=1)
            j = int(np.argmin(d)) if match == 'nearest_ray' else int(np.argmax(r['conf']))
            p = r['pred'][j]
            ug = K @ g[:3]
            up = K @ p[:3]
            uv_err.append(np.linalg.norm(ug[:2] / ug[2] - up[:2] / up[2]))
            ratio.append(p[2] / g[2])
            pdim.append(p[3:6])
            gdim.append(g[3:6])
            app_px.append(f * float(np.max(g[3:6])) / g[2])
            f_in.append(f)
            zv_pred.append(p[2] * 512.0 / f)
            zv_gt.append(g[2] * 512.0 / f)
    ratio, uv_err, app_px = map(np.array, (ratio, uv_err, app_px))
    pdim, gdim = np.array(pdim), np.array(gdim)
    print('== %s（%s，%d 个目标）' % (name, split, len(ratio)))
    print('   输入焦距 f_in 中位 %.1f px | 真值表观长边 中位 %.1f px (p10 %.1f p90 %.1f)'
          % (np.median(f_in), np.median(app_px), np.percentile(app_px, 10), np.percentile(app_px, 90)))
    print('   预测深度/真值深度 中位 %.3f (p10 %.3f p90 %.3f) | 真值虚拟深度 中位 %.2f 预测虚拟深度 中位 %.2f'
          % (np.median(ratio), np.percentile(ratio, 10), np.percentile(ratio, 90), np.median(zv_gt), np.median(zv_pred)))
    print('   2D 中心误差 中位 %.1f px (p90 %.1f) | 预测尺寸 l/w/h 中位 %s  真值 %s'
          % (np.median(uv_err), np.percentile(uv_err, 90), np.round(np.median(pdim, 0), 3), np.round(np.median(gdim, 0), 3)))
    size_ratio = np.max(pdim, 1) / np.max(gdim, 1)
    print('   预测尺寸/真值尺寸（长边）中位 %.3f | 深度比 / 尺寸比 中位 %.3f（≈1 说明深度误差来自把机子认大/认小）'
          % (np.median(size_ratio), np.median(ratio / size_ratio)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()
    analyse('仿真验证集', 'cfgs/models/uavdet_3d/camnorm/sim_indoor8.yaml', 'test', 10, args.ckpt, 'nearest_ray')
    analyse('MAV6D 验证集', 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml', 'val', 5, args.ckpt, 'top1')


if __name__ == '__main__':
    main()
