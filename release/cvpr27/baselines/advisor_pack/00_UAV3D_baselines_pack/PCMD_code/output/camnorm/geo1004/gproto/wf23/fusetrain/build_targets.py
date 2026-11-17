# -*- coding: utf-8 -*-
"""wf23 第 2 条 · 第二步：把第二段真实训练裁窗的伪目标换成"两种教师合并后的距离"（不读标签；真值只做诊断打印）。
原伪目标：δ_伪 = log(z_伪 / z_P0)，z_伪 = 全严格单帧伪标签 st2sfnb 的距离（红外教的测距网络修正过的检测）。
新伪目标：log z_合 = (1−w)·log z_伪 + w·(log z_kp − b_k)，δ_新 = log(z_合 / z_P0) = δ_伪 + w·(log z_kp − log z_伪 − b_k)。
  z_kp = LiDAR 教的角点测距（ztrain.py，部署同法）；只在 wsfnb 检测与伪标签相距 < 0.5 m 的帧上用（同一架飞机）。
  b_k = 颜色原型 k 上 median(log z_kp / z_伪)，无标签训练集统计；没有 z_kp 的裁窗保持原伪目标。
另存"两教师一致度"逐样本权重：a = exp(−r²/2τ_k²)，r = log z_kp − log z_伪 − b_k，τ_k = 1.4826·MAD_k(r)（数据自身尺度，不调）。
用法：python build_targets.py <ztrain pkl> [w,w,...]   输出 selftrain/real_s{j}_nbmdp_fz{100w}.npy、real_s{j}_nbmdw_ag.npy"""
import os, sys, pickle, json
os.environ.setdefault('FE_CONST', 'train')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../mirror')
import numpy as np
import chain_eval_m as CE   # noqa: E402
ROOT = '@@ROOT@@'; G = ROOT + '/output/camnorm/geo1004'; S = G + '/gproto/wf21/strict'
sys.path.insert(0, ROOT)
from scipy.spatial.transform import Rotation as R   # noqa: E402
from uavdet3d.utils import pose_eval as pe   # noqa: E402
ZT = sys.argv[1]
W = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '0.15,0.3').split(',')]
SUF = os.environ.get('BT_SUF', '')
b = '@@MMCACHE@@/mav6d_cn_obj/train'
d = pickle.load(open(b + '/index.pkl', 'rb'))
key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(d['metas'])}
ix = pickle.load(open(b + '/index_pseudo_st2sfnb.pkl', 'rb'))
ps = {(str(ix['metas'][i]['seq']), str(ix['metas'][i]['frame'])): np.asarray(ix['metas'][i]['boxes9d'], np.float64)[0] for i in ix['valid_idx']}
Z = pickle.load(open(ZT, 'rb'))
K = CE.constants('train'); lum = CE.load_unc('train')['lum']
cc = np.asarray(K['col_c'])
cls = lambda k: int(np.argmin(np.abs(cc - lum[k]))) if k in lum else -1
dd = {}
for k, z in Z.items():
    if k in ps and z['det'] is not None and z['z_kp'] > 0 and np.linalg.norm(z['det'][:3] - ps[k][:3]) < 0.5:
        dd[k] = float(np.log(z['z_kp'] / ps[k][2]))
B, TAU = {}, {}
for c in (0, 1):
    v = np.array([x for k, x in dd.items() if cls(k) == c])
    B[c] = float(np.median(v)); TAU[c] = float(1.4826 * np.median(np.abs(v - B[c])))
print('有 z_kp 且与伪标签同一架的训练帧 %d / 伪标签帧 %d / z_kp 帧 %d' % (len(dd), len(ps), len(Z)))
print('b_k（无标签训练集）', {c: round(v, 4) for c, v in B.items()}, '  τ_k', {c: round(v, 4) for c, v in TAU.items()})


def zgt(k):
    return float(np.asarray(d['metas'][key[k]]['boxes9d'], np.float64)[0][2])


# 真值只诊断：帧级两种测距的误差、相关、合并后误差
rob = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))
diag = {}
for c in (0, 1):
    ks = [k for k in dd if cls(k) == c]
    ep = np.array([np.log(ps[k][2] / zgt(k)) for k in ks]); ek = np.array([dd[k] - B[c] + np.log(ps[k][2] / zgt(k)) for k in ks])
    row = dict(n=len(ks), e_ps_bias=float(np.median(ep)), e_ps_rob=rob(ep), e_kp_rob=rob(ek), corr=float(np.corrcoef(ep - np.median(ep), ek - np.median(ek))[0, 1]))
    for w in [0.0] + W + [0.5]:
        ef = (1 - w) * ep + w * ek
        row['mae_w%02d' % int(round(100 * w))] = float(np.mean(np.abs(ef - np.median(ef))))
    diag[c] = row
    print('[诊断] 颜色类 %d（%s） n=%d | 伪标签误差 偏差 %+.1f%% 去偏散布 %.1f%% | 角点误差 去偏散布 %.1f%% | 两者相关 %.2f | 去偏平均绝对误差 w=0 %.2f%% → %s' % (
        c, '深色' if c == 0 else '亮色', row['n'], 100 * row['e_ps_bias'], 100 * row['e_ps_rob'], 100 * row['e_kp_rob'], row['corr'],
        100 * row['mae_w00'], ' '.join('w=%.2f %.2f%%' % (w, 100 * row['mae_w%02d' % int(round(100 * w))]) for w in W + [0.5])))

# 逐裁窗：重放 extract_nb.py 的筛选规则拿到每个裁窗属于哪一帧（不读图像），再按 extract_nbm.py 只留有伪目标的
for j in range(5):
    recs = pickle.load(open('%s/recs/train_P0nb_s%d.pkl' % (S, j), 'rb'))['records']
    kk, gz = [], []
    for r in recs:
        if len(r['pred']) == 0:
            continue
        pi = int(np.argmax(r['conf']))
        if r['conf'][pi] < 0.5:
            continue
        i = key[(r['seq_id'], r['frame_id'])]
        Km = np.asarray(d['metas'][i]['K_in'], np.float64)
        g = r['pred'][pi]
        if g[2] <= 1e-3:
            continue
        c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
        q = (Km @ c.T).T
        if (q[:, 2] <= 1e-3).any():
            continue
        uv = q[:, :2] / q[:, 2:3]
        w_ = uv[:, 0].max() - uv[:, 0].min()
        if not (6.0 <= w_ <= 170.0):
            continue
        kk.append((str(r['seq_id']), str(r['frame_id']))); gz.append(float(g[2]))
    dp_all = np.load('%s/selftrain/real_s%d_nbdp_sfnb.npy' % (G, j))
    assert len(dp_all) == len(kk), (len(dp_all), len(kk))
    m = np.isfinite(dp_all)
    kk = [k for k, mm in zip(kk, m) if mm]; gz = np.array(gz)[m]
    dp = np.load('%s/selftrain/real_s%d_nbmdp_sfnb.npy' % (G, j))
    assert len(dp) == len(kk)
    assert np.allclose(dp, [np.log(ps[k][2] / z) for k, z in zip(kk, gz)], atol=1e-4)
    r = np.array([dd[k] - B[cls(k)] if (k in dd and cls(k) in B) else np.nan for k in kk])
    has = np.isfinite(r)
    tau = np.array([TAU.get(cls(k), np.nan) for k in kk])
    a = np.where(has, np.exp(-0.5 * (np.nan_to_num(r) / tau) ** 2), np.nan)
    a = np.where(has, a, np.nanmean(a)).astype(np.float32)
    np.save('%s/selftrain/real_s%d_nbmdw_ag%s.npy' % (G, j, SUF), a)
    np.save('%s/selftrain/real_s%d_nbmdr_fz%s.npy' % (G, j, SUF), r.astype(np.float32))   # 两教师去偏分歧（缺失 NaN），供原型再分配
    dg = np.array([np.log(zgt(k) / z) for k, z in zip(kk, gz)])
    msg = []
    for w in W:
        dn = (dp + w * np.nan_to_num(r)).astype(np.float32)
        np.save('%s/selftrain/real_s%d_nbmdp_fz%02d%s.npy' % (G, j, int(round(100 * w)), SUF), dn)
        msg.append('w=%.2f |δ−δ真| %.4f（有角点的裁窗 %.4f）' % (w, np.mean(np.abs(dn - dg)), np.mean(np.abs(dn[has] - dg[has]))))
    print('s%d：裁窗 %d，有 z_kp %d（%.0f%%）| [诊断] 原伪目标 |δ−δ真| %.4f（有角点的裁窗 %.4f）| %s | 一致度权重 均值 %.2f' % (
        j, len(kk), int(has.sum()), 100 * has.mean(), np.mean(np.abs(dp - dg)), np.mean(np.abs(dp[has] - dg[has])), ' | '.join(msg), float(a.mean())), flush=True)
json.dump(dict(B=B, TAU=TAU, W=W, diag=diag, n=len(dd)), open('%s/build_targets%s.json' % (HERE, SUF), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
