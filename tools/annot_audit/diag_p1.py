# -*- coding: utf-8 -*-
"""P1 零样本为什么 2D 中心就错了 110 px：查置信度、uv 误差分布、目标像素跨度两域是否对齐。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402


def stats(name, recs):
    uv, sc, spx, zg, diag_g = [], [], [], [], []
    for r in recs:
        K = r['K']
        for g in r['gt']:
            s_px, _ = U.proj_extent(g, K)
            spx.append(s_px); zg.append(g[2]); diag_g.append(np.linalg.norm(g[3:6]))
            if len(r['pred']) == 0:
                uv.append(np.nan); sc.append(0.0); continue
            p = r['pred'][0]
            uvp = (K @ p[:3])[:2] / p[2]
            uvg = (K @ g[:3])[:2] / g[2]
            uv.append(float(np.linalg.norm(uvp - uvg)))
            sc.append(float(r['conf'][0]) if len(r['conf']) else np.nan)
    uv, sc, spx, zg, diag_g = map(np.array, (uv, sc, spx, zg, diag_g))
    q = lambda v, p: np.nanpercentile(v, p)
    print('\n' + '=' * 104 + '\n' + name + '  n=%d' % len(uv))
    print('  GT 像素跨度 s_px: 中位 %.1f px  [p10 %.1f, p90 %.1f]   GT 深度中位 %.2f m  GT 尺寸对角线中位 %.3f m' % (
        np.median(spx), q(spx, 10), q(spx, 90), np.median(zg), np.median(diag_g)))
    print('  top1 置信度: 中位 %.4f  [p10 %.4f, p90 %.4f]  最大 %.4f' % (np.nanmedian(sc), q(sc, 10), q(sc, 90), np.nanmax(sc)))
    print('  2D 中心误差 uv: 中位 %.1f px  [p25 %.1f, p75 %.1f]  |  <半个目标的比例 %.3f  <10px %.3f  >50px %.3f' % (
        np.nanmedian(uv), q(uv, 25), q(uv, 75), np.nanmean(uv < 0.5 * spx), np.nanmean(uv < 10), np.nanmean(uv > 50)))
    # 好帧 vs 坏帧的目标大小差别
    good = uv < 0.5 * spx
    if good.sum() > 5 and (~good).sum() > 5:
        print('  找到的帧 s_px 中位 %.1f（n=%d） / 没找到的帧 s_px 中位 %.1f（n=%d）' % (
            np.median(spx[good]), good.sum(), np.median(spx[~good]), (~good).sum()))


def main():
    P1 = U.M + '/sim_pp_realsize/P1/ckpt/best.pth'
    S1 = U.M + '/sim_indoor8_mz/S1/ckpt/best.pth'
    mav = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
    pp = 'cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml'
    stats('P1 · MAV6D 真实 test（零样本）', U.infer(mav, 10, P1))
    stats('S1 · MAV6D 真实 test（零样本，对照）', U.infer(mav, 10, S1))
    for z in ('1.0', '2.5'):
        stats('P1 · 新仿真 test（裁窗 %s，域内）' % z, U.infer(pp, 20, P1, ['DATA_CONFIG.VAL_ZOOMS', '[%s]' % z]))


if __name__ == '__main__':
    main()
