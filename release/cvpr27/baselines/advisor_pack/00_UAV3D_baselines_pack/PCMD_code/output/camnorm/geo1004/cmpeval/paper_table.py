# -*- coding: utf-8 -*-
"""按 AntiUAV3D 论文表 2 的格式出对比表（真实 MAV6D 测试集，只给彩色）。读 pf/<NAME>_test_<adabn|raw>.npz（cmp_eval.py 存的逐帧预测）。
  真正例 TP：预测框与真值框投影到图上的 2D 框 IoU ≥ 0.1（同 anti3d/compare_paper.py）。
  AOE（°）：TP 上四旋翼对称朝向误差均值；ATE（m）：TP 上三维中心误差均值；ASE（cm）：TP 上长宽高绝对误差三边平均。
  3D AP：nuScenes 式中心距 AP。MAV6D 只有 1.6–5.6 m，阈值按距离尺度缩小为 {0.1, 0.2, 0.4, 0.8} m，四个阈值平均。
  UDS：论文没给公式，这里照 nuScenes NDS 定义：UDS = ½·AP + ½·mean(1−min(1, ATE/1 m), 1−min(1, ASE/10 cm), 1−min(1, AOE/57.3°), mDR)（论文说 UDS 含 mAP、ATE/AOE/ASE 归一化和召回 mDR，没给公式），×100。
  距离段：近 <3 m / 中 3–4 m / 远 ≥4 m（测试集大致三等分）。多种子取平均。真值只用于打分。
用法：python paper_table.py [adabn|raw]"""
import glob, os, re, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
C = '@@ROOT@@/output/camnorm/geo1004/cmpeval'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'adabn'
PROTO = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
THR = (0.1, 0.2, 0.4, 0.8)
BINS = (('近 <3 m', 0, 3), ('中 3–4 m', 3, 4), ('远 ≥4 m', 4, 99))
ROWS = [  # (组, 键, 方法名, 出处, 训练模态)
    ('单模态（训练、测试都只用彩色）', 'GUPNETF', 'GUPNet (Lu et al. 2021)', 'ICCV 21', 'RGB'),
    ('单模态（训练、测试都只用彩色）', 'MONODETRF', 'MonoDETR (Zhang et al. 2023)', 'ICCV 23', 'RGB'),
    ('单模态（训练、测试都只用彩色）', 'MONOCDF', 'MonoCD (Yan et al. 2024)', 'CVPR 24', 'RGB'),
    ('单模态（训练、测试都只用彩色）', 'MONODGPF', 'MonoDGP (Pu et al. 2025)', 'CVPR 25', 'RGB'),
    ('单模态（训练、测试都只用彩色）', 'MONOCOPF', 'MonoCoP (Zhang et al. 2026)', 'CVPR 26', 'RGB'),
    ('单模态（训练、测试都只用彩色）', 'FULL1', '我们的管线，不用红外/LiDAR', '–', 'RGB'),
    ('模态缺失（训练 RGB+IR+LiDAR，测试只给 RGB）', 'SHASPEC', 'ShaSpec (Wang et al. 2023)', 'CVPR 23', 'RGB+IR+LiDAR'),
    ('我们（训练 RGB+IR+LiDAR，测试只给 RGB）', 'FULL2', 'Ours（只用红外蒸馏）', '–', 'RGB+IR'),
    ('我们（训练 RGB+IR+LiDAR，测试只给 RGB）', 'FULL4', 'Ours（红外 + LiDAR 蒸馏，完整）', '–', 'RGB+IR+LiDAR'),
]


def box2d(t, lwh, eul, K):
    R = Rot.from_euler('xyz', eul).as_matrix()
    P = (PROTO * lwh) @ R.T + t
    if (P[:, 2] <= 0.01).any():
        return None
    u = P @ K.T
    u = u[:, :2] / u[:, 2:]
    return np.r_[u.min(0), u.max(0)]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def sym(Rp, Rg):
    return min(np.degrees(np.arccos(np.clip((np.trace((Rp @ RZ[q]).T @ Rg) - 1) / 2, -1, 1))) for q in range(4))


def nus_ap(sc, tp, n_gt):
    if not len(sc):
        return 0.0
    o = np.argsort(-sc)
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1)
    rs = np.linspace(0, 1, 101)
    p = np.array([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in rs])[rs > 0.1]
    return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)


def metrics(P, M, sel):
    g = M['gt'][sel]; K = M['K'][sel]
    pt, pl, pe, cf = P['pt'][sel], P['plwh'][sel], P['peul'][sel], P['conf'][sel]
    has = np.isfinite(pt).all(1) & np.isfinite(cf)
    d = np.linalg.norm(pt - g[:, :3], axis=1)
    ap = 100 * np.mean([nus_ap(cf[has], d[has] <= th, len(sel)) for th in THR])
    te, ae, se = [], [], []
    for i in np.where(has)[0]:
        eul = pe[i] if np.isfinite(pe[i]).all() else np.zeros(3)
        if iou(box2d(pt[i], pl[i], eul, K[i]), box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        te.append(d[i]); se.append(100 * np.mean(np.abs(pl[i] - g[i, 3:6])))
        ae.append(sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if np.isfinite(pe[i]).all() else 90.0)
    ATE, ASE, AOE = (float(np.mean(x)) if x else np.nan for x in (te, se, ae))
    mdr = len(te) / max(len(sel), 1)          # 召回：真值被（置信度最高的）预测以 2D IoU ≥ 0.1 检中的比例
    tps = [1 - min(1, ATE / 1.0), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr] if te else [0, 0, 0, 0]
    uds = 100 * (0.5 * ap / 100 + 0.5 * np.mean(tps))
    return dict(AOE=AOE, ATE=ATE, ASE=ASE, AP=ap, UDS=uds, mDR=100 * mdr)


def main():
    M = np.load(C + '/test_meta.npz')
    dist = np.linalg.norm(M['gt'][:, :3], axis=1)
    valid = M['n_gt'] > 0
    R = {}
    for f in glob.glob('%s/pf/*_test_%s.npz' % (C, MODE)):
        m = re.match(r'(.+)_s(\d+)_test_', os.path.basename(f))
        if not m:
            continue
        P = np.load(f)
        r = {'all': metrics(P, M, np.where(valid)[0])}
        for b, lo, hi in BINS:
            r[b] = metrics(P, M, np.where(valid & (dist >= lo) & (dist < hi))[0])
        R.setdefault(m.group(1), []).append(r)

    def avg(key, part, k):
        if key not in R:
            return None
        return float(np.mean([x[part][k] for x in R[key]]))

    cols = [('all', 'AOE'), ('all', 'ATE'), ('all', 'ASE'), ('all', 'AP'), ('all', 'UDS')] + [(b[0], 'AP') for b in BINS] + [(b[0], 'UDS') for b in BINS]
    lower = {'AOE', 'ATE', 'ASE'}
    vals = {k: [avg(k, p, c) for p, c in cols] for _, k, *_ in ROWS}
    best = []
    for j, (p, c) in enumerate(cols):     # 每列最好加粗、第二下划线
        v = [(vals[k][j], k) for _, k, *_ in ROWS if vals[k][j] is not None]
        v.sort(key=lambda x: x[0] if c in lower else -x[0])
        best.append([x[1] for x in v[:2]])
    fmt = {'AOE': '%.2f', 'ATE': '%.3f', 'ASE': '%.2f', 'AP': '%.2f', 'UDS': '%.2f'}
    L = ['# 对比实验（AntiUAV3D 表 2 格式，真实 MAV6D 测试集 4800 帧，测试只给彩色，%s）' % ('AdaBN 后' if MODE == 'adabn' else '直接测'), '',
         '| 方法 | 出处 | 训练模态 | AOE (°) ↓ | ATE (m) ↓ | ASE (cm) ↓ | 3D AP ↑ | UDS ↑ | AP 近 | AP 中 | AP 远 | UDS 近 | UDS 中 | UDS 远 | 种子数 |',
         '|---|---|---|' + '---|' * 12]
    grp = None
    for g, k, name, venue, mod in ROWS:
        if g != grp:
            L.append('| **%s** |' % g + ' |' * 14); grp = g
        cells = []
        for j, (p, c) in enumerate(cols):
            x = vals[k][j]
            s = '训练中' if x is None else fmt[c] % x
            if x is not None and best[j] and k == best[j][0]:
                s = '**%s**' % s
            elif x is not None and len(best[j]) > 1 and k == best[j][1]:
                s = '<u>%s</u>' % s
            cells.append(s)
        L.append('| %s | %s | %s | %s | %s |' % (name, venue, mod, ' | '.join(cells), len(R.get(k, []))))
    L += ['', '注：AP 阈值 {0.1, 0.2, 0.4, 0.8} m（MAV6D 距离只有 1.6–5.6 m，论文的 1/2/4/8 m 不适用）；UDS 论文没给公式，按 nuScenes NDS 写法自定义（见脚本头）；',
          'KZ 骨干 / ShaSpec / 画板的位置和尺寸都来自同一个部署检测器，所以 ATE、ASE 相同，差别在 AOE 和置信度排序；GUPNet / MonoCD 自己出位置和尺寸。']
    out = C + '/PAPER_TABLE_%s.md' % MODE
    open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
