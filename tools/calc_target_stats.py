# -*- coding: utf-8 -*-
"""算回归真值（虚拟深度 / 尺寸）在【训练集自身】上的均值与标准差，用来做自洽标准化。

为什么（2026-09-16）：现在 center_dis 真值 = Zv / MAX_DIS、dim 真值 = lwh / MAX_SIZE，
MAX_DIS/MAX_SIZE 是手填的常数（旧 laam6d 配置里是 150/4，camnorm 改成 24/2）。两个问题：

1. 对 L1 + 线性输出头，MAX_DIS 与损失权重完全简并（loss = w·|p − Zv/M| = (w/M)·|p_物理 − Zv|），
   单独改它等于改权重，不解决任何事。
2. 真正的毛病是：`center_dis` / `dim` / `rot` / `center_res` 四个头的输出层【没有 bias】
   （只有 hm 有）。输出 = W·f 是 ReLU 后特征的一次齐次函数，特征幅值一漂移，预测就整体缩放。
   实测 S1 在 MAV6D 上头内激活是仿真的 3.35 倍、深度输出 1.76 倍，
   而零样本误差正是 `预测 = 2.21×真值 − 0.27`（截距为零的纯乘性误差）—— 这就是无偏置头的签名。
   内部对照：五个头里唯一有 bias 的 hm，恰恰是唯一迁移得好的（2D 定位 84% 成功、中心误差 9.4 px）。

自洽的解法是把常数从特征通路里拿出来：target = (量 − mu) / sigma，解码时 量 = sigma·pred + mu。
mu / sigma 只由【训练集自己】决定，不看任何目标域 —— 换任何部署数据都不用重调。

注意虚拟深度要按训练时真正看到的算：jpeg 缓存在线裁窗 z 会把 Zv 变成 Zv_cache·(z_ref/z)，
所以这里按 VIEW_AUG.zoom 的对数均匀分布蒙特卡洛抽样，而不是直接用缓存里的值。

    python tools/calc_target_stats.py --cfg cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml
"""
import argparse
import os
import pickle
import sys

import numpy as np
from easydict import EasyDict

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--reps', type=int, default=16, help='每个目标按裁窗分布重抽多少次')
    ap.add_argument('--write', default=None, help='把统计量写进这个 yaml')
    a = ap.parse_args()

    cfg = EasyDict()
    cfg_from_yaml_file(a.cfg, cfg)
    dc = cfg.DATA_CONFIG
    f_ref = float(dc.get('DEPTH_F_REF', 512))
    zoom = (dc.get('VIEW_AUG', {}) or {}).get('zoom', [1.0, 1.0])
    z0, z1 = float(zoom[0]), float(zoom[1])
    no_up = bool((dc.get('VIEW_AUG', {}) or {}).get('no_upscale', False))

    idx = pickle.load(open(os.path.join(dc.DATA_PATH, a.split, 'index.pkl'), 'rb'))
    W_src = int(idx['W'])
    W_in = int(dc.IM_RESIZE[0])
    zmax_nomag = W_src / float(W_in)

    Zv_cache, LWH = [], []
    for i in idx['valid_idx']:
        m = idx['metas'][int(i)]
        K = np.asarray(m['K_in']).reshape(3, 3)
        f_in = float(np.sqrt(K[0, 0] * K[1, 1]))
        for b in np.asarray(m['boxes9d']).reshape(-1, 9):
            if np.abs(b).sum() == 0 or b[2] <= 0:
                continue
            Zv_cache.append(b[2] * f_ref / f_in)
            LWH.append(b[3:6])
    Zv_cache = np.asarray(Zv_cache, np.float64)
    LWH = np.asarray(LWH, np.float64).reshape(-1, 3)

    rng = np.random.default_rng(0)
    store = str(idx.get('store', 'npy'))
    if store == 'jpeg':
        # 只有原分辨率 jpeg 缓存才在线裁窗。裁宽 W_src/z 的窗口缩到 W_in：
        #   f_in = f_cache * W_in * z / W_src   ->   Zv = Zv_cache * (W_src / W_in) / z
        z = np.exp(rng.uniform(np.log(z0), np.log(z1), size=(a.reps, len(Zv_cache)))) if z1 > z0             else np.full((a.reps, len(Zv_cache)), z0)
        if no_up:
            z = np.minimum(z, zmax_nomag)
        Zv = (Zv_cache[None, :] * (W_src / float(W_in)) / z).ravel()
    else:
        Zv = Zv_cache          # 缓存分辨率就是网络输入，不裁窗

    dm, ds = float(Zv.mean()), float(Zv.std())
    sm, ss = LWH.mean(0), LWH.std(0)
    print('%s  %s 划分：%d 个目标，裁窗 zoom [%g, %g]%s' % (
        dc.DATA_PATH, a.split, len(Zv_cache), z0, z1, '（不放大，上限 %.2f）' % zmax_nomag if no_up else ''))
    print('  虚拟深度 Zv   均值 %.3f  标准差 %.3f   | p1 %.2f  中位 %.2f  p99 %.2f  最大 %.2f' % (
        dm, ds, np.percentile(Zv, 1), np.median(Zv), np.percentile(Zv, 99), Zv.max()))
    print('  尺寸 l,w,h    均值 %s  标准差 %s' % (np.round(sm, 4), np.round(ss, 4)))
    print()
    print('  # 自洽标准化（tools/calc_target_stats.py 实测，只来自训练集本身）')
    print('  DEPTH_MEAN: %.4f' % dm)
    print('  DEPTH_STD: %.4f' % ds)
    print('  SIZE_MEAN: [%.4f, %.4f, %.4f]' % tuple(sm))
    print('  SIZE_STD: [%.4f, %.4f, %.4f]' % tuple(ss))
    print()
    print('  参考：现行 MAX_DIS=%s 下真值中位 %.3f（只用到 [0,%.2f]）；标准化后真值均值 0 标准差 1' % (
        dc.get('MAX_DIS'), np.median(Zv) / float(dc.get('MAX_DIS', 1)), Zv.max() / float(dc.get('MAX_DIS', 1))))

    if a.write:
        txt = open(a.write, encoding='utf-8').read().rstrip('\n')
        keys = ('DEPTH_MEAN:', 'DEPTH_STD:', 'SIZE_MEAN:', 'SIZE_STD:')
        lines = [ln for ln in txt.split('\n') if not ln.startswith(keys)]
        lines += ['# 自洽标准化（tools/calc_target_stats.py 实测，只来自训练集本身）',
                  'DEPTH_MEAN: %.4f' % dm, 'DEPTH_STD: %.4f' % ds,
                  'SIZE_MEAN: [%.4f, %.4f, %.4f]' % tuple(sm),
                  'SIZE_STD: [%.4f, %.4f, %.4f]' % tuple(ss)]
        open(a.write, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
        print('-> 已写入', a.write)


if __name__ == '__main__':
    main()
