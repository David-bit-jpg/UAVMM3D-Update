# -*- coding: utf-8 -*-
"""只在「2D 真的找到了」的帧上比较 P1 与 S1 的深度比 / 角度误差：区分「检测不出来」和「修正没生效」。"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

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
            # 取 2D 中心最近的那个预测（给模型最好的机会）
            d = [np.linalg.norm((K @ p[:3])[:2] / p[2] - uvg) for p in r['pred']]
            j = int(np.argmin(d))
            p, uv = r['pred'][j], d[j]
            s_px, _ = U.proj_extent(g, K)
            ang = float(np.degrees((R.from_euler('xyz', p[6:9]).inv() * R.from_euler('xyz', g[6:9])).magnitude()))
            rows.append(dict(uv=uv, s=s_px, top1=(j == 0), zr=p[2] / g[2], ang=ang, fold=min(ang, 180 - ang),
                             Li=s_px * p[2] / f, Lt=s_px * g[2] / f, sr=np.linalg.norm(p[3:6]) / np.linalg.norm(g[3:6]),
                             pos=float(np.linalg.norm(p[:3] - g[:3]))))
    a = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    ok = a['uv'] < 0.5 * a['s']
    print('\n' + '=' * 104 + '\n%s  n=%d  其中 2D 找到 %d (%.1f%%)，找到的里 top1 占 %.1f%%' % (
        name, len(rows), ok.sum(), 100 * ok.mean(), 100 * a['top1'][ok].mean()))
    m = lambda k: np.median(a[k][ok])
    print('  【只看找到的帧】深度比 Z_pred/Z_gt 中位 %.3f | 位置误差中位 %.3f m | 角度误差中位 %.1f°（折叠 %.1f°）| 尺寸比 %.2f' % (
        m('zr'), m('pos'), m('ang'), m('fold'), m('sr')))
    print('  模型假设跨度 L_imp %.3f m vs 真实 L_true %.3f m -> 比值 %.2f' % (m('Li'), m('Lt'), np.median(a['Li'][ok] / a['Lt'][ok])))
    print('  【全部帧】位置误差中位 %.3f m  角度中位 %.1f°' % (np.median(a['pos']), np.median(a['ang'])))


def main():
    mav = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
    go('P1（重采 真机尺寸 + 原生机头+x，单场景 PowerPlant）· MAV6D test', U.infer(mav, 10, U.M + '/sim_pp_realsize/P1/ckpt/best.pth'))
    go('S1（原标签，8 场景）· MAV6D test', U.infer(mav, 10, U.M + '/sim_indoor8_mz/S1/ckpt/best.pth'))


if __name__ == '__main__':
    main()
