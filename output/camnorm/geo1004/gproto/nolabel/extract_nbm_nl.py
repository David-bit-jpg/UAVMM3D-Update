# -*- coding: utf-8 -*-
"""wf21/strict/extract_nb.py + extract_nbm.py 的无标签链副本（11-10）：第二段真实训练裁窗 = P0_s{j} 单帧最高分检测（置信 ≥0.5、投影宽 ∈ [6,170]）
且与本帧位置伪标签 index_pseudo_<arm>（只按置信度筛帧，make_nolabel_pseudo.py）相距 < 0.5 m；伪目标 δ_伪 = log(伪距离 / P0 检测距离)。
P0 单帧检测记录（recs/train_P0nb_s{j}.pkl，纯仿真检测器、不依赖新检测器）与其裁窗（selftrain/real_s{j}_nb{rgb,scal}.npy）原样复用：
按 extract_nb.py 的同一规则重放每个裁窗属于哪一帧（只读检测记录的 pred / conf / K，不读 gt），再按新伪标签挑行。
输出 selftrain/real_s{j}_nbm_<arm>_{rgb,scal}.npy、real_s{j}_nbm_<arm>_dp.npy（种子 5、6 用 j = 0、1）。
用法：python extract_nbm_nl.py <arm> [j ...]"""
import pickle, sys, os
import numpy as np
ROOT = '@@ROOT@@'; G = ROOT + '/output/camnorm/geo1004'; S = G + '/gproto/wf21/strict'
sys.path.insert(0, ROOT)
from scipy.spatial.transform import Rotation as R
from uavdet3d.utils import pose_eval as pe
arm = sys.argv[1]
js = [int(a) for a in sys.argv[2:]] or [0, 1]
ix = pickle.load(open('@@MMCACHE@@/mav6d_cn_obj/train/index_pseudo_%s.pkl' % arm, 'rb'))
assert ix.get('pseudo', {}).get('pipeline', '').startswith('无标签版'), '只接受 make_nolabel_pseudo.py 写的伪标签'
ps = {(str(ix['metas'][i]['seq']), str(ix['metas'][i]['frame'])): np.asarray(ix['metas'][i]['boxes9d'], np.float64)[0] for i in ix['valid_idx']}
print('%s 位置伪标签帧 %d' % (arm, len(ps)), flush=True)
for j in js:
    pre = '%s/selftrain/real_s%d_nbm_%s_' % (G, j, arm)
    if os.path.exists(pre + 'dp.npy'):
        print('exists', pre); continue
    recs = pickle.load(open('%s/recs/train_P0nb_s%d.pkl' % (S, j), 'rb'))['records']
    kk, gz = [], []
    for r in recs:
        if len(r['pred']) == 0:
            continue
        pi = int(np.argmax(r['conf']))
        if r['conf'][pi] < 0.5:
            continue
        K = np.asarray(r['K'], np.float64)
        g = r['pred'][pi]
        if g[2] <= 1e-3:
            continue
        c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
        q = (K @ c.T).T
        if (q[:, 2] <= 1e-3).any():
            continue
        uv = q[:, :2] / q[:, 2:3]
        w = uv[:, 0].max() - uv[:, 0].min()
        if not (6.0 <= w <= 170.0):
            continue
        kk.append(((str(r['seq_id']), str(r['frame_id'])), g[:3].copy()))
    rgb = np.load('%s/selftrain/real_s%d_nbrgb.npy' % (G, j), mmap_mode='r')
    sc = np.load('%s/selftrain/real_s%d_nbscal.npy' % (G, j))
    assert len(rgb) == len(sc) == len(kk), (len(rgb), len(sc), len(kk))
    assert np.allclose(sc[:, 1], np.log([p[2] for _, p in kk]), atol=1e-4)      # 重放的检测与裁窗标量逐行一致
    dp = np.full(len(kk), np.nan, np.float32)
    for n, (k, p) in enumerate(kk):
        if k in ps and np.linalg.norm(ps[k][:3] - p) < 0.5:
            dp[n] = np.log(ps[k][2] / p[2])
    m = np.isfinite(dp)
    np.save(pre + 'rgb.npy', np.ascontiguousarray(rgb[m]))
    np.save(pre + 'scal.npy', sc[m])
    np.save(pre + 'dp.npy', dp[m])
    pickle.dump([k for (k, _), mm in zip(kk, m) if mm], open(pre + 'keys.pkl', 'wb'))
    print('j=%d：P0 单帧裁窗 %d → 与 %s 伪标签一致 %d；δ_伪 中位 %+.3f' % (j, len(kk), arm, int(m.sum()), float(np.median(dp[m]))), flush=True)
