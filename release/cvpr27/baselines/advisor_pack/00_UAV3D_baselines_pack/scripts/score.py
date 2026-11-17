# -*- coding: utf-8 -*-
"""给 runs/*/pred_<real_test|outdoor_test>.pkl 打分，出表 runs/SCORES.md：每个目标域一张表，带源域列（同一方法同一源域多个种子取平均 ± 标准差）。
指标同论文（照 AntiUAV3D 表 2 + nuScenes NDS 写法）：每帧取分数最高的一个框。
  TP：预测框与真值框投影 2D 框 IoU ≥ 0.1；AOE（°）= TP 上四旋翼对称朝向误差（转 90° 的倍数不算错）均值；ATE = TP 上中心误差均值；ASE（cm）= 长宽高绝对误差三边平均；mDR = TP 比例。
  3D AP：nuScenes 式中心距 AP（101 点、召回 >0.1、精度减 0.1），多个阈值平均。
  UDS = ½·AP + ½·mean(1−min(1, ATE/a), 1−min(1, ASE/10 cm), 1−min(1, AOE/57.3°), mDR)。
室内 MAV6D（1.6–5.6 m）：阈值 {0.1, 0.2, 0.4, 0.8} m，a = 1 m，距离段 近<3 / 中3–4 / 远≥4 m。
室外（AntiUAV3D 80 m 以内；看结果之前定死）：阈值 = 真值距离的 {2.5, 5, 10, 20}%，ATE 按真值距离归一（相对误差），a = 25%，距离段 近<25 / 中25–50 / 远≥50 m。
用法：python scripts/score.py"""
import glob, os, pickle, re, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
CFG = {'real_test': dict(rel=False, thr=(0.1, 0.2, 0.4, 0.8), a=1.0, bins=((0, 3), (3, 4), (4, 99))),
       'outdoor_test': dict(rel=True, thr=(0.025, 0.05, 0.10, 0.20), a=0.25, bins=((0, 25), (25, 50), (50, 999)))}


def box2d(t, lwh, eul, K):
    P = (PROTO * lwh) @ Rot.from_euler('xyz', eul).as_matrix().T + t
    if (P[:, 2] <= 0.01).any():
        return None
    u = P @ K.T; u = u[:, :2] / u[:, 2:]
    return np.r_[u.min(0), u.max(0)]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def sym(Rp, Rg):
    return min(np.degrees(np.arccos(np.clip((np.trace((Rp @ RZ[q]).T @ Rg) - 1) / 2, -1, 1))) for q in range(4))


def nus_ap(sc, tp, n_gt):
    if not len(sc):
        return 0.0
    # 11-16: stable sort -> frames with EQUAL scores keep the test-meta order (sequence, then numeric frame number).
    # Without it the AP of predictions with many tied scores (MonoCD outdoor) depended on the sort implementation.
    o = np.argsort(-sc, kind='stable'); tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1)
    rs = np.linspace(0, 1, 101)
    p = np.array([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in rs])[rs > 0.1]
    return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)


def metrics(P, g, K, sel, c):
    pt, pl, pe, cf = P['pt'][sel], P['plwh'][sel], P['peul'][sel], P['conf'][sel]; g = g[sel]; K = K[sel]
    has = np.isfinite(pt).all(1) & np.isfinite(cf)
    d = np.linalg.norm(pt - g[:, :3], axis=1)
    if c['rel']:
        d = d / np.linalg.norm(g[:, :3], axis=1)
    ap = 100 * np.mean([nus_ap(cf[has], d[has] <= th, len(g)) for th in c['thr']])
    te, ae, se = [], [], []
    for i in np.where(has)[0]:
        eul = pe[i] if np.isfinite(pe[i]).all() else np.zeros(3)
        if iou(box2d(pt[i], pl[i], eul, K[i]), box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        te.append(d[i]); se.append(100 * np.mean(np.abs(pl[i] - g[i, 3:6])))
        ae.append(sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if np.isfinite(pe[i]).all() else 90.0)
    ATE, ASE, AOE = (float(np.mean(x)) if x else np.nan for x in (te, se, ae))
    mdr = len(te) / max(len(g), 1)
    tps = [1 - min(1, ATE / c['a']), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr] if te else [0, 0, 0, 0]
    return dict(AOE=AOE, ATE=ATE, ASE=ASE, AP=ap, UDS=100 * (0.5 * ap / 100 + 0.5 * np.mean(tps)), mDR=100 * mdr)


def score(pred_path, split):
    M = np.load(os.path.join(PKG, 'data', 'gt', '%s_meta.npz' % split))
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    N = len(M['seq']); P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(pred_path, 'rb')):
        i = key.get((str(r['seq']), str(r['frame'])))
        b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    c = CFG[split]; g = M['gt']; dist = np.linalg.norm(g[:, :3], axis=1) if c['rel'] else g[:, 2]
    out = {'all': metrics(P, g, M['K'], np.arange(N), c)}
    for lo, hi in c['bins']:
        sel = np.where((dist >= lo) & (dist < hi))[0]
        out['%g-%g' % (lo, hi)] = metrics(P, g, M['K'], sel, c)
    return out


METHODS = ['gupnet', 'monocd', 'monodgp', 'monodetr', 'monocop']
SRC_NAME = {'in': 'in（室内 UE5 仿真）', 'out': 'out（室外 CARLA 仿真）', 'mix': 'mix（in + out + Town10 筛选帧）'}
STEP_NAME = {'sim': '只用仿真', 'st': '+ 无标签自训练', 'kd': '+ PCMD 教师伪标签'}


def parse(name):
    """运行目录名 <方法>_<源域>_<步骤>[_<目标域>] → (方法, 源域, 步骤)"""
    p = name.split('_') + ['?', '?', '?']
    return p[0], p[1], p[2]


def order(n):
    m, src, step = parse(n)
    return (['sim', 'st', 'kd'].index(step) if step in STEP_NAME else 9, METHODS.index(m) if m in METHODS else 9, ['in', 'out', 'mix'].index(src) if src in SRC_NAME else 9, n)


if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(PKG, 'runs', '*', 'pred_*_test.pkl')))
    rows = {}
    for f in files:
        split = re.search(r'pred_(real_test|outdoor_test)\.pkl$', f.replace('\\', '/'))
        if not split:
            continue
        name = re.sub(r'_s\d+$', '', os.path.basename(os.path.dirname(f)))
        rows.setdefault((split.group(1), name), []).append(score(f, split.group(1)))
    lines = ['# 分数（同一方法多个种子：均值 ± 标准差）', '',
             '源域：in = 室内 UE5 仿真 sim_train；out = 室外 CARLA 仿真 sim_outdoor_train；mix = in + out + Town10 按画质筛过的帧。',
             '"只用仿真"的同一个模型同时测两个目标域；"+ 自训练 / + PCMD 教师伪标签"只测对应目标域（MAV6D 从 in 出发，AntiUAV3D 从 out 出发）。', '']
    for split in ('real_test', 'outdoor_test'):
        R = sorted([(n, v) for (s, n), v in rows.items() if s == split], key=lambda x: order(x[0]))
        if not R:
            continue
        unit = '相对' if CFG[split]['rel'] else 'm'
        bins = list(R[0][1][0].keys())[1:]
        lines += ['## 目标域 %s' % ('MAV6D 测试集（室内真实）' if split == 'real_test' else 'AntiUAV3D 80 m 内测试集（室外真实）'), '',
                  '| 方法 | 源域 | 步骤 | 种子数 | AOE (°)↓ | ATE (%s)↓ | ASE (cm)↓ | 3D AP↑ | UDS↑ | mDR | ' % unit + ' | '.join('AP %s' % b for b in bins) + ' |',
                  '|' + '---|' * (10 + len(bins))]
        for n, v in R:
            def f(k, part='all', nd=2):
                x = np.array([e[part][k] for e in v], float)
                return ('%.*f' % (nd, np.nanmean(x))) + (' ± %.*f' % (nd, np.nanstd(x)) if len(x) > 1 else '')
            m, src, step = parse(n)
            lines.append('| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |' % (m, SRC_NAME.get(src, src), STEP_NAME.get(step, step), len(v), f('AOE'), f('ATE', nd=3), f('ASE'), f('AP'), f('UDS'), f('mDR', nd=1), ' | '.join(f('AP', b) for b in bins)))
        lines.append('')
    open(os.path.join(PKG, 'runs', 'SCORES.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    print('\n'.join(lines))
