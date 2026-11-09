# -*- coding: utf-8 -*-
"""wf21/strict 第 2 步：第二段的真实训练裁窗与伪目标换成单帧版本（新文件，不覆盖旧文件）。
(a) 裁窗：规则与 selftrain_extract.py 逐字相同（P0_s{j} 每帧最高分检测，置信 ≥0.5、投影宽 ∈ [6,170]，128×128 + 5 个标量），
    只把检测记录换成**不开热图时序背景**的 P0_s{j} 推理（recs/train_P0nb_s{j}.pkl）→ selftrain/real_s{j}_nb{rgb,scal}.npy
(b) 伪目标：规则与 ssst_targets.py 相同（δ_伪 = log(伪距离 z / P0 检测 z)，只用伪标签接受的帧、3D 距离 < 0.5 m），
    只把伪标签从 ST2++（st2pp：含因果滚动中位校正 + 前后 3 帧位置筛选，跨帧）换成全严格单帧伪标签 index_pseudo_st2sfnb
    （ST2 五集成无背景检测 × 第二段修正、全训练集常数 ×0.75、第二段自估不确定度筛选）→ selftrain/real_s{j}_nbdp_sfnb.npy
不读真实标注。"""
import pickle, sys
import numpy as np
ROOT = '@@ROOT@@'; G = ROOT + '/output/camnorm/geo1004'; S = G + '/gproto/wf21/strict'
sys.path.insert(0, ROOT); sys.path.insert(0, G)
from scipy.spatial.transform import Rotation as R
from uavdet3d.utils import pose_eval as pe
import cropw_extract as ce
import silw_train as sw
ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
b = '@@MMCACHE@@/mav6d_cn_obj/train'
arr = np.load(b + '/rgb.npy', mmap_mode='r')
d = pickle.load(open(b + '/index.pkl', 'rb'))
key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(d['metas'])}
ix = pickle.load(open(b + '/index_pseudo_st2sfnb.pkl', 'rb'))
ps = {(str(ix['metas'][i]['seq']), str(ix['metas'][i]['frame'])): np.asarray(ix['metas'][i]['boxes9d'], np.float64)[0] for i in ix['valid_idx']}
print('全严格伪标签帧 %d' % len(ps), flush=True)
for j in [int(a) for a in (sys.argv[1:] or ['0', '1', '2', '3', '4'])]:
    recs = pickle.load(open('%s/recs/train_P0nb_s%d.pkl' % (S, j), 'rb'))['records']
    views, scal, dp = [], [], []
    for r in recs:
        if len(r['pred']) == 0:
            continue
        pi = int(np.argmax(r['conf']))
        if r['conf'][pi] < 0.5:
            continue
        i = key[(r['seq_id'], r['frame_id'])]
        K = np.asarray(d['metas'][i]['K_in'], np.float64)
        g = r['pred'][pi]
        if g[2] <= 1e-3:
            continue
        c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
        q = (K @ c.T).T
        if (q[:, 2] <= 1e-3).any():
            continue
        uv = q[:, :2] / q[:, 2:3]
        u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
        w = u1 - u0
        if not (6.0 <= w <= 170.0):
            continue
        img = np.ascontiguousarray(arr[i])
        views.append(sw.view_img(ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), ce.KW * w), S0, 0.0, 0.0, 1.0, w, False))
        scal.append([np.log(w / K[0, 0]), np.log(g[2])] + list(np.log(np.maximum(g[3:6], 1e-3))))
        k = (str(r['seq_id']), str(r['frame_id']))
        v = np.nan
        if k in ps and np.linalg.norm(ps[k][:3] - g[:3]) < 0.5:
            v = float(np.log(ps[k][2] / g[2]))
        dp.append(v)
    np.save('%s/selftrain/real_s%d_nbrgb.npy' % (G, j), np.stack(views))
    np.save('%s/selftrain/real_s%d_nbscal.npy' % (G, j), np.array(scal, np.float32))
    dp = np.array(dp, np.float32); np.save('%s/selftrain/real_s%d_nbdp_sfnb.npy' % (G, j), dp)
    fin = np.isfinite(dp)
    old = np.load('%s/selftrain/real_s%d_scal.npy' % (G, j))
    print('j=%d：%d 帧 → %d 个裁窗（旧带背景版 %d）；有伪目标 %.1f%%，δ_伪 中位 %+.3f（四分位 %+.3f ~ %+.3f）' % (
        j, len(recs), len(views), len(old), 100 * fin.mean(), np.median(dp[fin]), np.percentile(dp[fin], 25), np.percentile(dp[fin], 75)), flush=True)
