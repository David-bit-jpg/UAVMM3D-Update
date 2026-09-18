# -*- coding: utf-8 -*-
"""深度到底有没有从图里读出来：网络输出 vs 常数先验 vs 真值上界（2026-09-18）。

为什么要有这个对照：log_ratio + 已知尺寸时 深度 = exp(t) x 尺寸，只要 t 接近训练均值，深度就落在一个固定值上，
而 MAV6D 的真实深度中位恰好在那附近 —— 「给尺寸就大幅提升」可能只是先验凑巧。实测 Q2：
网络 0.942 m，常数对照 0.754 m（更好），真值上界 0.044 m。所以任何新臂都必须先打赢常数对照。

    python annot_audit/compare_depth_prior.py [--arms A,B]
只看 2D 命中的帧（峰值落在机身上），三种深度来源共用同一份 2D 中心。
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
import eval_2d as E   # noqa: E402
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

M = '../output/models/uavdet_3d/camnorm/'
C = 'cfgs/models/uavdet_3d/camnorm/'
KNOWN = np.linalg.norm([0.34, 0.34, 0.23])          # MAV6D 官方框对角线
F_REF = 512.0
# 臂名: (评测配置, 权重, 该臂训练集的 t 均值来源配置)
ARMS = {
    'Q2': (C + 'mav6d_r34_ratio_pp.yaml', M + 'sim_pp_r34_ratio/Q2/ckpt/best.pth', 'log_ratio'),
    'A': (C + 'mav6d_r34_ratio_full_t.yaml', M + 'sim_full_r34_ratio_t/A/ckpt/best.pth', 'log_ratio'),
    'B': (C + 'mav6d_r34_geo_full_t.yaml', M + 'sim_full_r34_geo_t/B/ckpt/best.pth', 'value'),
    'C': (C + 'mav6d_r34_ratio_full_t.yaml', M + 'sim_full_r34_ratio_tr/C/ckpt/best.pth', 'log_ratio'),
}


def run(tag, interval):
    cfgf, ckpt, mode = ARMS[tag]
    if not os.path.exists(ckpt):
        print('%s: 没有权重 %s，跳过' % (tag, ckpt))
        return
    cfg = EasyDict()
    cfg_from_yaml_file(cfgf, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'known', 'MODEL.POST_PROCESSING.MAX_OBJ', '1'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    dc = cfg.DATA_CONFIG
    ds, loader, _ = build_dataloader(dc, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(ckpt, to_cpu=False)
    assert n == t, (n, t)
    model.cuda().eval()
    recs = pose_eval.run_inference(model, loader)

    # 常数先验：log_ratio 臂用 exp(DEPTH_MEAN)*L；value 臂直接用 DEPTH_MEAN（都是训练集自身的均值）
    dmu = float(dc.DEPTH_MEAN)
    zv_prior = np.exp(dmu) * KNOWN if mode == 'log_ratio' else dmu
    out = {k: [] for k in ('net', 'const', 'oracle')}
    zr, zp, zg = [], [], []
    hits = tot = 0
    for r in recs:
        tot += 1
        if not len(r['conf']):
            continue
        K = r['K']
        g = r['gt'][0]
        p = r['pred'][int(np.argmax(r['conf']))]
        gb = E.box2d(E.corners_of(g), K, ds.new_im_width, ds.new_im_hight)
        if gb is None or p[2] <= 0:
            continue
        if np.linalg.norm(E.proj(K, p[:3]) - E.proj(K, g[:3])) > 0.5 * max(gb[2] - gb[0], gb[3] - gb[1]):
            continue
        hits += 1
        f_in = float(np.sqrt(K[0, 0] * K[1, 1]))
        z_const = zv_prior * f_in / F_REF
        out['net'].append(np.linalg.norm(p[:3] - g[:3]))
        out['const'].append(np.linalg.norm(p[:3] * (z_const / p[2]) - g[:3]))
        out['oracle'].append(np.linalg.norm(p[:3] * (g[2] / p[2]) - g[:3]))
        zr.append(p[2] / g[2]); zp.append(p[2]); zg.append(g[2])
    net, const, oracle = (np.array(out[k]) for k in ('net', 'const', 'oracle'))
    zr, zp, zg = np.array(zr), np.array(zp), np.array(zg)
    print('%s  2D 命中 %d/%d 帧  (深度先验常数 = %.2f m 处)' % (tag, hits, tot, zv_prior * 492 / F_REF))
    for lab, arr in (('网络输出', net), ('常数先验对照', const), ('真值上界', oracle)):
        print('    %-12s 位置中位 %.3f m | <0.2 m %4.1f%% | <0.5 m %4.1f%%' % (
            lab, np.median(arr), 100 * (arr < 0.2).mean(), 100 * (arr < 0.5).mean()))
    print('    深度比 中位 %.2f (p10 %.2f p90 %.2f) | log(预测深度) 与 log(真值深度) 相关性 %.2f' % (
        np.median(zr), np.percentile(zr, 10), np.percentile(zr, 90),
        np.corrcoef(np.log(zp), np.log(zg))[0, 1] if len(zp) > 2 else float('nan')))
    print('    判据：网络输出 %s 常数先验（%.3f vs %.3f）' % (
        '打赢了' if np.median(net) < np.median(const) else '**没打赢**', np.median(net), np.median(const)))
    return dict(tag=tag, net=float(np.median(net)), const=float(np.median(const)), oracle=float(np.median(oracle)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='A,B')
    ap.add_argument('--interval', type=int, default=2)
    a = ap.parse_args()
    for tag in a.arms.split(','):
        if tag in ARMS:
            run(tag, a.interval)
            print()


if __name__ == '__main__':
    main()
