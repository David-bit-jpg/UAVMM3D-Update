# -*- coding: utf-8 -*-
"""定位实验（2026-09-18）：模型预测的「机身倍数」t 到底来自目标本身，还是来自「整幅画面放大了多少」这个全局线索。

t = log(k·f_ref / 目标像素跨度)，纯 2D 量，不需要认机型。但训练时 t 的变化很大一部分由裁窗倍数 z 决定
（t = log(2Z/(zL))，对齐采样后 corr(t, log z) ≈ −0.7），而 z 是整幅画面的属性（纹理尺度、模糊程度），
网络不看无人机也能读出来。真实图没有裁窗，这条捷径就失效。

做法：在【仿真 test】上把裁窗倍数固定成单一值（VAL_ZOOMS = [z]），此时样本间 t 的差异只来自
真实深度 Z 和机型尺寸 L。拟合 t_pred ~ t_true：
    斜率 ≈ 1 -> 模型真的在量目标（捷径不是主因）
    斜率 ≈ 0 -> 固定 z 后就预测不出 t，说明它读的是 z（捷径就是主因）

    python annot_audit/fixed_zoom_probe.py --arm Q2 [--zooms 1.0,2.5,4.0]
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
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

M = '../output/models/uavdet_3d/camnorm/'
C = 'cfgs/models/uavdet_3d/camnorm/'
ARMS = {                      # 臂名: (仿真域配置, 权重)
    'Q2': (C + 'sim_pp_r34_ratio.yaml', M + 'sim_pp_r34_ratio/Q2/ckpt/best.pth'),
    'A': (C + 'sim_full_r34_ratio_t.yaml', M + 'sim_full_r34_ratio_t/A/ckpt/best.pth'),
    'B': (C + 'sim_full_r34_geo_t.yaml', M + 'sim_full_r34_geo_t/B/ckpt/best.pth'),
    'R1': (C + 'sim_pp_r34.yaml', M + 'sim_pp_r34/R1/ckpt/best.pth'),
}


def probe(arm, zoom, interval):
    cfgf, ckpt = ARMS[arm]
    cfg = EasyDict()
    cfg_from_yaml_file(cfgf, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.MAX_OBJ', '5'], cfg)
    cfg.DATA_CONFIG.VAL_ZOOMS = [float(zoom)]
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    f_ref = float(cfg.DATA_CONFIG.get('DEPTH_F_REF', 512))
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(ckpt, to_cpu=False)
    assert n == t, (n, t)
    model.cuda().eval()
    recs = pose_eval.run_inference(model, loader)
    xs, ys, Ls = [], [], []
    for r in recs:
        if not len(r['conf']):
            continue
        K = r['K']
        f_in = float(np.sqrt(K[0, 0] * K[1, 1]))
        for g in r['gt']:
            gb = E.box2d(E.corners_of(g), K, ds.new_im_width, ds.new_im_hight)
            if gb is None or g[2] <= 0:
                continue
            side = max(gb[2] - gb[0], gb[3] - gb[1])
            j = None
            for cand in np.argsort(-r['conf']):
                p = r['pred'][cand]
                if p[2] > 0 and np.linalg.norm(E.proj(K, p[:3]) - E.proj(K, g[:3])) <= 0.5 * side:
                    j = cand
                    break
            if j is None:
                continue
            p = r['pred'][j]
            Lg = float(np.linalg.norm(g[3:6]))
            Lp = max(float(np.linalg.norm(p[3:6])), 1e-3)
            xs.append(np.log(g[2] * f_ref / f_in / Lg))        # 真值 t
            ys.append(np.log(p[2] * f_ref / f_in / Lp))        # 预测 t（用模型自己的尺寸，log_ratio 臂即网络直出）
            Ls.append(Lg)
    x, y, L = np.array(xs), np.array(ys), np.array(Ls)
    if len(x) < 20:
        print('  zoom %.1f: 命中太少 (%d)' % (zoom, len(x)))
        return
    k, b = np.polyfit(x, y, 1)
    print('  裁窗固定 %.1fx  n=%4d | 斜率 %.2f 截距 %+.2f | corr %.2f | 真值 t σ %.2f（p10 %.2f p90 %.2f）| 预测 t σ %.2f' % (
        zoom, len(x), k, b, np.corrcoef(x, y)[0, 1], x.std(), np.percentile(x, 10), np.percentile(x, 90), y.std()))
    # 去掉机型尺寸的影响：在同一机型内部看斜率（t 只由真实深度变化驱动）
    for lo, hi, nm in ((0, 0.6, '小机'), (0.6, 1.6, '中机'), (1.6, 9, '大机')):
        m = (L >= lo) & (L < hi)
        if m.sum() >= 30:
            kk = np.polyfit(x[m], y[m], 1)[0]
            print('      %s 内部 n=%4d 斜率 %.2f corr %.2f（真值 t σ %.2f）' % (
                nm, m.sum(), kk, np.corrcoef(x[m], y[m])[0, 1], x[m].std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='Q2')
    ap.add_argument('--zooms', default='1.0,2.5')
    ap.add_argument('--interval', type=int, default=6)
    a = ap.parse_args()
    print('%s（仿真 test，固定裁窗倍数；斜率≈1 = 在量目标，≈0 = 在读裁窗这个全局线索）' % a.arm)
    for z in a.zooms.split(','):
        probe(a.arm, float(z), a.interval)


if __name__ == '__main__':
    main()
