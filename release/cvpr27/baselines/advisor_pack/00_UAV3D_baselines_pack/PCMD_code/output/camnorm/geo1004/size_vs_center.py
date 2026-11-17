# -*- coding: utf-8 -*-
"""把位置误差拆成「框中心在画面里差几像素」和「框在画面里的宽度估得准不准（→ 深度）」（10-08，用户问：IR 帮不帮尺寸）。
只读已有逐帧记录，不跑模型。匹配与 pose_eval greedy 相同（3D 距离 < 1 m）。每个比较只在两边都检出的共同目标上算。
每个目标：
  center_px  预测中心与真值中心投影到画面的距离（像素）
  width_err  |log(预测框画面宽 / 真值框画面宽)|，框 = 3D 框 8 角点投影的水平跨度（模型深度 = 焦距 × 尺寸 ÷ 画面宽度，宽度只看水平）
  depth      |Z_预测 − Z_真值|（米）
  lateral    3D 误差中垂直于视线的部分（米）
  pos        3D 位置误差（米）
写 SIZE_VS_CENTER.md / size_vs_center.json。"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, '@@ROOT@@')
from uavdet3d.utils import pose_eval as pe  # noqa: E402

G = '@@ROOT@@/output/camnorm/geo1004'
S5 = range(5)
MET = ['center_px', 'lateral', 'width_err', 'size_err', 'zrel_err', 'width_bias', 'size_bias', 'zrel_bias', 'depth', 'pos']
LAB = {'center_px': '中心偏差（像素）', 'lateral': '横向误差（cm）', 'width_err': '画面宽度误差 %', 'size_err': '实际尺寸误差 %', 'zrel_err': '深度相对误差 %',
       'width_bias': '画面宽度偏差 %（正 = 估宽）', 'size_bias': '实际尺寸偏差 %（正 = 估大）', 'zrel_bias': '深度偏差 %（正 = 估远）',
       'depth': '深度误差（cm）', 'pos': '位置误差（cm）'}
SCALE = {'center_px': 1.0, 'lateral': 100.0, 'width_err': 100.0, 'size_err': 100.0, 'zrel_err': 100.0, 'width_bias': 100.0, 'size_bias': 100.0, 'zrel_bias': 100.0,
         'depth': 100.0, 'pos': 100.0}


def corners(b, seq):
    return (pe.PROTO8 * b[3:6]) @ R.from_euler(seq, b[6:9]).as_matrix().T + b[:3]


def uspan(c, K):
    p = (K @ c.T).T
    if (p[:, 2] <= 1e-6).any():
        return np.nan
    u = p[:, 0] / p[:, 2]
    return float(u.max() - u.min())


def per_target(path):
    """-> {(seq_id, frame_id, gt 下标): 指标字典}，只含匹配上的目标"""
    z = pickle.load(open(path, 'rb'))
    recs, seq, dt = z['records'], z.get('seq', 'xyz'), z.get('dist_thresh', 1.0)
    out = {}
    for r in recs:
        gt, pred, conf, K = r['gt'], r['pred'], r['conf'], r['K']
        if len(gt) == 0 or len(pred) == 0:
            continue
        used = np.zeros(len(gt), bool)
        for j in np.argsort(-conf):
            d = np.linalg.norm(gt[:, :3] - pred[j, :3], axis=1)
            d[used] = np.inf
            k = int(np.argmin(d))
            if d[k] < dt:
                used[k] = True
                p, g = pred[j], gt[k]
                up, ug = K @ p[:3], K @ g[:3]
                cpx = float(np.linalg.norm(up[:2] / up[2] - ug[:2] / ug[2]))
                wp, wg = uspan(corners(p, seq), K), uspan(corners(g, seq), K)
                e = p[:3] - g[:3]
                ray = g[:3] / np.linalg.norm(g[:3])
                lat = float(np.linalg.norm(e - (e @ ray) * ray))
                ew = float(np.log(wp / wg)) if wp > 0 and wg > 0 else np.nan     # 画面宽度（对数比）
                ez = float(np.log(p[2] / g[2])) if p[2] > 0 and g[2] > 0 else np.nan  # 深度（对数比）
                el = ew + ez                                                      # 隐含实际尺寸 = 画面宽 × 深度 / 焦距 的对数比
                out[(r['seq_id'], r['frame_id'], k)] = dict(center_px=cpx, lateral=lat, width_err=abs(ew), size_err=abs(el), zrel_err=abs(ez),
                                                            width_bias=ew, size_bias=el, zrel_bias=ez,
                                                            depth=float(abs(p[2] - g[2])), pos=float(np.linalg.norm(e)))
    return out


def med_common(a, b):
    keys = sorted(set(a) & set(b))
    ma = {m: np.nanmedian([a[k][m] for k in keys]) * SCALE[m] for m in MET}
    mb = {m: np.nanmedian([b[k][m] for k in keys]) * SCALE[m] for m in MET}
    return ma, mb, len(keys)


def boot(a, b, metric, n=1000, seed=0):
    """按飞行片段重抽样：共同目标上 median(a) − median(b) 的 95% 区间（单个配对）"""
    keys = sorted(set(a) & set(b))
    by = {}
    for k in keys:
        by.setdefault(k[0], []).append(k)
    seqs = sorted(by)
    rng = np.random.default_rng(seed)
    va = {s: np.array([a[k][metric] for k in by[s]]) for s in seqs}
    vb = {s: np.array([b[k][metric] for k in by[s]]) for s in seqs}
    out = []
    for _ in range(n):
        ss = rng.choice(seqs, size=len(seqs), replace=True)
        out.append((np.nanmedian(np.concatenate([va[s] for s in ss])) - np.nanmedian(np.concatenate([vb[s] for s in ss]))) * SCALE[metric])
    return np.percentile(out, [2.5, 97.5])


def compare(name, pairs, L, res):
    """pairs: [(a_path, b_path)]（按种子配对）"""
    rows = []
    for pa, pb in pairs:
        A, B = per_target(pa), per_target(pb)
        ma, mb, n = med_common(A, B)
        rows.append((ma, mb, n, A, B))
    n_mean = np.mean([r[2] for r in rows])
    L += ['### %s（%d 组配对，每组共同目标约 %d 个）' % (name, len(rows), n_mean), '',
          '| 指标 | 前者 | 后者 | 差（前 − 后） | 前者更好的配对数 | 按飞行片段 95% 区间（第 1 组） |', '|---|---|---|---|---|---|']
    res[name] = {}
    for m in MET:
        xa = np.mean([r[0][m] for r in rows])
        xb = np.mean([r[1][m] for r in rows])
        d = [r[0][m] - r[1][m] for r in rows]
        ci = boot(rows[0][3], rows[0][4], m)
        better = sum(abs(r[0][m]) < abs(r[1][m]) for r in rows) if m.endswith('_bias') else sum(x < 0 for x in d)
        L.append('| %s | %.1f | %.1f | %+.1f | %d/%d | [%+.1f, %+.1f] |' % (LAB[m], xa, xb, np.mean(d), better, len(d), ci[0], ci[1]))
        res[name][m] = dict(a=xa, b=xb, diff=float(np.mean(d)), per=[float(x) for x in d], ci_first=[float(ci[0]), float(ci[1])])
    L.append('')


def main():
    L = ['# 位置误差拆成「中心」和「宽度（尺寸 → 深度）」（共同目标上比，中位数）', '',
         '画面宽度 = 3D 框 8 角点投影的水平跨度；实际尺寸 = 画面宽 × 深度 ÷ 焦距（隐含的物理宽度）。模型的深度 = 焦距 × 实际尺寸 ÷ 画面宽度，',
         '所以 深度对数误差 = 实际尺寸对数误差 − 画面宽度对数误差：宽度估宽 1% 让深度估近 1%，尺寸估大 1% 让深度估远 1%。百分数都是对数比 × 100。',
         '「偏差」行是带符号的中位数（看系统性偏向）；这几行的「更好」= 绝对值更小。', '']
    res = {}
    SF = G + '/sim_full_recs/%s.pkl'
    L += ['## 仿真全量 test（4646 帧）', '']
    if os.path.exists(SF % 'TIR_s0'):
        compare('红外教师（看真红外）− 普通彩色模型', [(SF % 'TIR_s0', SF % ('P0_s%d' % k)) for k in S5], L, res)
    gp = G + '/gray/sim_recs/replace_gray_s0.pkl'
    if os.path.exists(gp):
        compare('红外教师看灰度 − 普通彩色模型（种子 0）', [(gp, SF % 'P0_s0')], L, res)
    compare('只学教师版 − 普通彩色模型', [(SF % ('DHs_s%d' % k), SF % ('P0_s%d' % k)) for k in S5], L, res)
    compare('只学教师版 − 对照组', [(SF % ('DHs_s%d' % k), SF % ('D0_s%d' % k)) for k in S5], L, res)
    compare('既学教师又收任务 − 对照组', [(SF % ('DH_s%d' % k), SF % ('D0_s%d' % k)) for k in S5], L, res)
    if all(os.path.exists(SF % ('PIO_s%d' % k)) for k in S5):
        compare('输出蒸馏（教师的宽度等输出当答案）− 普通彩色模型', [(SF % ('PIO_s%d' % k), SF % ('P0_s%d' % k)) for k in S5], L, res)
    if all(os.path.exists(SF % ('PIM_s%d' % k)) for k in S5):
        compare('红外剪影当辅助答案 − 普通彩色模型', [(SF % ('PIM_s%d' % k), SF % ('P0_s%d' % k)) for k in S5], L, res)
    RD = lambda a, k: (G + '/irloc_recs/%s_s%d.pkl' if a == 'P0' else G + '/dual_recs/%s_s%d.pkl') % (a, k)
    L += ['## 真实 test（4800 帧，标准口径）', '']
    compare('只学教师版 − 普通彩色模型（真实）', [(RD('DHs', k), RD('P0', k)) for k in S5], L, res)
    compare('只学教师版 − 对照组（真实）', [(RD('DHs', k), RD('D0', k)) for k in S5], L, res)
    with open(G + '/SIZE_VS_CENTER.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    with open(G + '/size_vs_center.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
