# -*- coding: utf-8 -*-
"""模型的尺寸输出：仿真里按机型走（看图），真实图上塌回训练集先验（不看图）？"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402


def go(tag, recs, by_model=False):
    rows = []
    for r in recs:
        K = r['K']
        for g in r['gt']:
            if len(r['pred']) == 0:
                continue
            uvg = (K @ g[:3])[:2] / g[2]
            d = [np.linalg.norm((K @ p[:3])[:2] / p[2] - uvg) for p in r['pred']]
            j = int(np.argmin(d))
            s_px, _ = U.proj_extent(g, K)
            if d[j] > 0.5 * s_px:
                continue
            rows.append((np.linalg.norm(g[3:6]), np.linalg.norm(r['pred'][j][3:6]),
                         r['pred'][j][2] / g[2], r['seq_id'].split('/')[-1].split('_')[0]))
    if len(rows) < 5:
        print('%-48s 样本太少' % tag); return
    a = np.array([r[:3] for r in rows], float)
    print('%-48s n=%4d | GT 对角线 %.3f  预测 %.3f  比值 %.2f | 深度比 %.2f' % (
        tag, len(a), np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 1] / a[:, 0]), np.median(a[:, 2])))
    if by_model:
        for m in sorted(set(r[3] for r in rows)):
            b = np.array([r[:3] for r in rows if r[3] == m], float)
            if len(b) < 5:
                continue
            print('      %-22s n=%4d GT %.3f -> 预测 %.3f （比值 %.2f） 深度比 %.2f' % (
                m, len(b), np.median(b[:, 0]), np.median(b[:, 1]), np.median(b[:, 1] / b[:, 0]), np.median(b[:, 2])))


def main():
    P1 = U.M + '/sim_pp_realsize/P1/ckpt/best.pth'
    S1 = U.M + '/sim_indoor8_mz/S1/ckpt/best.pth'
    print('P1（训练集 对角线 中位 0.531 / 均值 0.971）')
    go('  P1 · 仿真 test（域内，裁窗 2.5）', U.infer('cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml', 6, P1,
                                                ['DATA_CONFIG.VAL_ZOOMS', '[2.5]']), by_model=True)
    go('  P1 · MAV6D 真实 test（GT 全部 0.533）', U.infer('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', 10, P1))
    print('S1（训练集 对角线 中位 0.829 / 均值 1.243）')
    go('  S1 · MAV6D 真实 test（GT 全部 0.533）', U.infer('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', 10, S1))


if __name__ == '__main__':
    main()
