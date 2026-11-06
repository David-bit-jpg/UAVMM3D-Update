# -*- coding: utf-8 -*-
"""把 dump_real_preds.py 的若干个预测 pkl 汇总成按机型拆开的表（2026-09-22）。

位置：检出率、位置误差中位、<10/<20/<50 cm（分母 = 全部真值帧，未检出算失败）、深度误差中位
朝向（只算检出帧）：航向折180 误差中位（分正面侧面、不分前后）、航向折90 中位（不分正面侧面，0~45°）、航向 2ψ 圆相关、
预测航向集中度（1 = 塌成常数）、倾斜误差中位

    python annot_audit/summarize_dumps.py NAME=path.pkl ...
"""
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def rot(b):
    return R.from_euler('xyz', b[6:9]).as_matrix()


def chordal_mean(Rs):
    u, _, vt = np.linalg.svd(np.asarray(Rs).mean(0))
    M = u @ vt
    if np.linalg.det(M) < 0:
        u[:, -1] *= -1
        M = u @ vt
    return M


def stats(F):
    n = len(F)
    hit = [f for f in F if f['matched'] is not None]
    e = np.full(n, np.inf)
    dz = []
    he, hp, hg, ti, h90, Rgs = [], [], [], [], [], []
    for k, f in enumerate(F):
        if f['matched'] is None:
            continue
        p, g = f['pred'][f['matched']], f['gt']
        e[k] = np.linalg.norm(p[:3] - g[:3])
        dz.append(abs(p[2] - g[2]))
        Rg, Rp = rot(g), rot(p)
        Rgs.append(Rg)
        E = Rg.T @ Rp
        a = abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
        he.append(min(a, 180 - a))
        h90.append(min(a % 90.0, 90.0 - a % 90.0))
        hp.append(np.arctan2(Rp[0, 0], Rp[2, 0]))
        hg.append(np.arctan2(Rg[0, 0], Rg[2, 0]))
        ti.append(np.degrees(np.arccos(np.clip(Rg[:, 2] @ Rp[:, 2], -1, 1))))
    # 常数基线（最优常数，用了真值，是下限参照）：永远输出「水平 + 固定航向 c」的姿态，c 在 0~360° 上扫，
    # 分别取使折180 / 折90 中位最小的 c。MAV6D 朝向集中在相机轴向（0/90/180/270°），所以折 90 后一个对准轴向的常数就很小，
    # 折 90 这一列打不过常数不奇怪，要看折 180 和相关。
    Rgs = np.asarray(Rgs)
    n_ = Rgs[:, :, 0]                                   # 真值机头在相机系
    hg_deg = np.degrees(np.arctan2(n_[:, 0], n_[:, 2]))  # 与 heading_cam 同定义
    best180, best90, bd180 = np.inf, np.inf, None
    for c_ in np.arange(0, 360, 1.0):
        d = np.abs((hg_deg - c_ + 180) % 360 - 180)
        d180 = np.minimum(d, 180 - d)
        if np.median(d180) < best180:
            best180, bd180 = np.median(d180), d180
        best90 = min(best90, np.median(np.minimum(d % 90.0, 90.0 - d % 90.0)))
    b180, b90 = [best180], [best90]
    he = np.asarray(he)
    tail = (100 * (he < 10).mean(), 100 * (he < 20).mean(), 100 * (bd180 < 10).mean(), 100 * (bd180 < 20).mean())
    hp, hg = np.asarray(hp), np.asarray(hg)
    c = np.corrcoef(np.cos(2 * hp), np.cos(2 * hg))[0, 1]
    s = np.corrcoef(np.sin(2 * hp), np.sin(2 * hg))[0, 1]
    f_ = e[np.isfinite(e)]
    return {'n': n, 'det': 100.0 * len(hit) / n, 'pos': np.median(f_), 'lt10': 100 * (e < 0.1).mean(), 'lt20': 100 * (e < 0.2).mean(),
            'lt50': 100 * (e < 0.5).mean(), 'dz': np.median(dz), 'head': np.median(he), 'head90': np.median(h90), 'b180': np.median(b180), 'b90': np.median(b90), 'tail': tail, 'corr': (c + s) / 2,
            'conc': float(np.hypot(np.cos(2 * hp).mean(), np.sin(2 * hp).mean())), 'tilt': np.median(ti)}


def main():
    specs = [a.split('=', 1) for a in sys.argv[1:]]
    print('%-16s %-9s %5s | %6s %8s %6s %6s %6s %8s | %8s %8s %13s %13s | %7s %7s | %7s %7s %6s' % (
        '', '机型', '帧', '检出', '位置中位', '<10cm', '<20cm', '<50cm', '深度中位', '航向折180', '最优常数', '折180<10°(常数)', '折180<20°(常数)', '航向折90', '最优常数', '航向相关', '预测集中', '倾斜'))
    for name, path in specs:
        D = pickle.load(open(path, 'rb'))
        for cls in ('phantom4', 'mavic2', '全部'):
            F = [f for f in D['frames'] if cls == '全部' or f['cls'] == cls]
            s = stats(F)
            t = s['tail']
            print('%-16s %-9s %5d | %5.1f%% %7.3fm %5.1f%% %5.1f%% %5.1f%% %7.3fm | %7.1f° %7.1f° %5.1f%%(%4.1f%%) %5.1f%%(%4.1f%%) | %6.1f° %6.1f° | %+7.3f %7.2f %5.1f°' % (
                name if cls == 'phantom4' else '', cls, s['n'], s['det'], s['pos'], s['lt10'], s['lt20'], s['lt50'], s['dz'],
                s['head'], s['b180'], t[0], t[2], t[1], t[3], s['head90'], s['b90'], s['corr'], s['conc'], s['tilt']))


if __name__ == '__main__':
    main()
