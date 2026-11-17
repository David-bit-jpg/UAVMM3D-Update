# -*- coding: utf-8 -*-
"""PREREG_ssst.md（sha256 38f4a437）第 1 步：给 selftrain/real_s{j} 每张真实小图配去偏伪目标 δ_伪 = log(伪距离 / P0 检测距离)。
小图顺序按 selftrain_extract.py 同规则重建（逐张核对数量）；伪距离来自 real_train_preds_st2pp.pkl，只用 index_pseudo_st2pp 接受的帧、3D 距离 < 0.5 m。
输出 selftrain/real_s{j}_dpseudo.npy（float32，无伪目标为 NaN）。不读真实标注。"""
import pickle, sys
import numpy as np
ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT); sys.path.insert(0, G)
from scipy.spatial.transform import Rotation as R
from uavdet3d.utils import pose_eval as pe

b = '@@MMCACHE@@/mav6d_cn_obj/train'
d = pickle.load(open(b + '/index.pkl', 'rb'))
key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(d['metas'])}
ix = pickle.load(open(b + '/index_pseudo_st2pp.pkl', 'rb'))
ok = set((str(ix['metas'][i]['seq']), str(ix['metas'][i]['frame'])) for i in ix['valid_idx'])
dump = pickle.load(open(ROOT + '/output/camnorm/st/real_train_preds_st2pp.pkl', 'rb'))
ps = {}
for f in dump['frames']:
    p = np.asarray(f['pred']).reshape(-1, 9)
    if len(p):
        ps[(str(f['seq']), str(f['frame']))] = p[int(np.argmax(np.asarray(f['conf']).reshape(-1)))]
for j in range(5):
    recs = pickle.load(open('%s/irloc_train_recs/train_P0_s%d.pkl' % (G, j), 'rb'))['records']
    n_ref = len(np.load('%s/selftrain/real_s%d_scal.npy' % (G, j)))
    out = []
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
        w = uv[:, 0].max() - uv[:, 0].min()
        if not (6.0 <= w <= 170.0):
            continue
        k = (r['seq_id'], r['frame_id'])
        v = np.nan
        if k in ok and k in ps and np.linalg.norm(ps[k][:3] - g[:3]) < 0.5:
            v = float(np.log(ps[k][2] / g[2]))
        out.append(v)
    assert len(out) == n_ref, (j, len(out), n_ref)
    out = np.array(out, np.float32)
    np.save('%s/selftrain/real_s%d_dpseudo.npy' % (G, j), out)
    fin = np.isfinite(out)
    print('j=%d：小图 %d，有伪目标 %.1f%%，δ_伪 中位 %+.3f（四分位 %+.3f ~ %+.3f）' % (j, len(out), 100 * fin.mean(), np.median(out[fin]), np.percentile(out[fin], 25), np.percentile(out[fin], 75)), flush=True)
