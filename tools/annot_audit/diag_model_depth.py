# -*- coding: utf-8 -*-
"""看模型：零样本深度 2 倍到底是不是「尺寸先验」造成的。

对每个 2D 中心对上的 (GT, 预测)：
  s_px    GT 8 角点投影长边（输入 512x288 像素）
  L_true  = s_px * Z_gt   / f_in    GT 框在画面上的物理跨度（米）
  L_imp   = s_px * Z_pred / f_in    模型「以为」同样像素大小的东西有多大（米）
若模型深度纯靠尺寸先验：L_imp 在各距离近乎常数（= 它在仿真里学到的该外观无人机的跨度），log Z_pred 对 log s_px 斜率 ≈ -1。
仿真参照：同一模型在仿真 test 上 L_imp ≈ L_true。
    python tools/annot_audit/diag_model_depth.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402

M = U.M


def collect(recs):
    out = []
    for r in recs:
        K = r['K']
        f = float(np.sqrt(K[0, 0] * K[1, 1]))
        for g in r['gt']:
            if len(r['pred']) == 0:
                continue
            j = int(np.argmin(np.linalg.norm(r['pred'][:, :3] / r['pred'][:, 2:3] - g[:3] / g[2], axis=1)))
            p = r['pred'][j]
            s_px, _ = U.proj_extent(g, K)
            uvp = (K @ p[:3])[:2] / p[2]
            uvg = (K @ g[:3])[:2] / g[2]
            if np.linalg.norm(uvp - uvg) > 0.5 * s_px:
                continue
            out.append(dict(seq=r['seq_id'], f=f, s=s_px, zg=g[2], zp=p[2], Lt=s_px * g[2] / f, Li=s_px * p[2] / f,
                            size_g=np.linalg.norm(g[3:6]), size_p=np.linalg.norm(p[3:6])))
    return out


def report(name, rows, groups):
    print('\n' + '=' * 100 + '\n' + name)
    for gname, sel in groups:
        x = [r for r in rows if sel(r)]
        if len(x) < 5:
            continue
        a = {k: np.array([r[k] for r in x]) for k in ('s', 'zg', 'zp', 'Lt', 'Li', 'size_g', 'size_p')}
        slope_p = np.polyfit(np.log(a['s']), np.log(a['zp']), 1)[0]
        slope_g = np.polyfit(np.log(a['s']), np.log(a['zg']), 1)[0]
        q = lambda v: '%.3f [%.3f, %.3f]' % (np.median(v), np.percentile(v, 25), np.percentile(v, 75))
        print('  %-10s n=%4d | GT 深度中位 %.2f m 预测 %.2f m | 画面跨度 L_true %s m | 模型假设跨度 L_imp %s m | L_imp/L_true 中位 %.2f | '
              'log深度~log像素 斜率: 预测 %.2f / GT %.2f | 预测尺寸对角线 %.3f（GT %.3f）' % (
                  gname, len(x), np.median(a['zg']), np.median(a['zp']), q(a['Lt']), q(a['Li']), np.median(a['Li'] / a['Lt']),
                  slope_p, slope_g, np.median(a['size_p']), np.median(a['size_g'])))
        # 按距离分段看 L_imp 是否恒定
        bins = np.percentile(a['zg'], [0, 33, 66, 100])
        segs = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (a['zg'] >= lo) & (a['zg'] <= hi)
            segs.append('Z %.1f~%.1f m: L_imp %.3f, 误差 %.2f m' % (lo, hi, np.median(a['Li'][m]), np.median(np.abs(a['zp'][m] - a['zg'][m]))))
        print('             ' + ' | '.join(segs))


def main():
    s1 = M + '/sim_indoor8_mz/S1/ckpt/best.pth'
    s3 = M + '/sim_indoor8_mz_nosescale/S3/ckpt/best.pth'
    mav = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
    sim = 'cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz_nose.yaml'
    cls = [('全部', lambda r: True), ('phantom4', lambda r: r['seq'].startswith('phantom4')), ('mavic2', lambda r: r['seq'].startswith('mavic2'))]
    report('S1（原标签，仿真 phantom4 盒子 0.545 m、含桨网格外廓 0.73~0.80 m）· MAV6D 真实 test', collect(U.infer(mav, 5, s1)), cls)
    report('S3（尺度修正 phantom4 k=1.65：盒子 0.33 m、含桨外廓 ~0.46 m）· MAV6D 真实 test', collect(U.infer(mav, 5, s3)), cls)
    simcls = [(m, (lambda mm: (lambda r: r['seq'].split('/')[-1].split('_')[0] == mm))(m)) for m in ('DJI-phantom4', 'DJI-mavic-mini', 'm210-rtk')]
    report('S1 · 仿真 test（域内参照，裁窗 1.92 倍，f_in 同 MAV6D）', collect(U.infer(sim, 10, s1, ['DATA_CONFIG.VAL_ZOOMS', '[1.92]'])),
           [('全部', lambda r: True)] + simcls)


if __name__ == '__main__':
    main()
