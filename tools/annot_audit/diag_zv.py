# -*- coding: utf-8 -*-
"""深度偏大 2.2 倍是不是「虚拟深度 Zv 覆盖/回归到训练均值」：同一模型在自己的仿真 test 上按裁窗倍数扫 Zv。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402


def go(name, recs):
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
            p, uv = r['pred'][j], d[j]
            s_px, _ = U.proj_extent(g, K)
            rows.append(dict(uv=uv, s=s_px, f=f, zg=g[2], zp=p[2],
                             zvg=g[2] * 512 / f, zvp=p[2] * 512 / f, zr=p[2] / g[2]))
    a = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    ok = a['uv'] < 0.5 * a['s']
    if ok.sum() < 5:
        print('%-46s 找到太少 (%d/%d)' % (name, ok.sum(), len(rows))); return
    print('%-46s n=%3d 找到 %3d | f_in %5.0f | Zv 真 %5.2f 预测 %5.2f | 深度比 %.3f | s_px 中位 %5.1f' % (
        name, len(rows), ok.sum(), np.median(a['f']), np.median(a['zvg'][ok]), np.median(a['zvp'][ok]),
        np.median(a['zr'][ok]), np.median(a['s'][ok])))


def main():
    pp = 'cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml'
    P1 = U.M + '/sim_pp_realsize/P1/ckpt/best.pth'
    print('P1 在自己的仿真 test 上，按裁窗倍数扫（Zv 越小 = 越像 MAV6D）')
    for z in ('1.0', '1.5', '2.5', '3.5', '4.0'):
        go('  仿真 test 裁窗 x%s' % z, U.infer(pp, 10, P1, ['DATA_CONFIG.VAL_ZOOMS', '[%s]' % z]))
    print('\nMAV6D 真实 test 作对照')
    go('  MAV6D test', U.infer('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', 10, P1))


if __name__ == '__main__':
    main()
