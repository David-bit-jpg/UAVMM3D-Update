# -*- coding: utf-8 -*-
"""[review1113 rgbst_equal] 'RGB self-training under equal conditions' pseudo labels, written in the ourdet_baseproto index format.
Rule = PST step 1 (srcx/lib/make_pseudo_x.py) with NO depth-refiner correction (c_bar = 0, delta = 0) = review1113/B4 make_pseudo_naive.py:
  frames whose 3-detector ensemble (+flip) top box has conf >= 0.5; box = (t_det, s_bar, R_det); no gate, no teacher orientation.
Input: srcx/<combo>/recs/det0.pkl (the sim-only ensemble recs PST step 1 uses; no real label is read).
Output: <baseproto cache copy>/train/index_naive.pkl (copy of the cache train index, boxes9d replaced, float64, no 'pseudo' flag = all heads
supervised, as index_st/index_pst) + labels/<combo>_naive.json.  Cross-check for in2mav: identical frames/boxes to B4's
index_pseudo_srcx_in2mav_naive.pkl (up to its float32 cast)."""
import json, pickle, sys
import numpy as np
COMBO = sys.argv[1]
G = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
C = {'in2mav': dict(cache='F:/ourdet_baseproto_cache/mav6d_cn_obj', recs=G + '/srcx/in2mav/recs/det0.pkl', n_train=12866,
                    b4='E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav_naive.pkl'),
     'out2antifs': dict(cache='D:/S2MUAV3D/ourdet_baseproto_cache/outdoor_real_fs', recs=G + '/srcx/out2antifs/recs/det0.pkl', n_train=3293, b4=None)}[COMBO]
CONF = 0.5
D = pickle.load(open(C['recs'], 'rb')); recs = D['recs']['train']; lwh = np.asarray(D['size'], np.float64)
B = C['cache'] + '/train'; idx = pickle.load(open(B + '/index.pkl', 'rb')); assert len(idx['metas']) == C['n_train']
k2i = {(m['seq'], m['frame']): i for i, m in enumerate(idx['metas'])}
metas = [None] * len(idx['metas']); valid = []
for r in recs:
    if r['box'] is None or r['conf'] < CONF:
        continue
    p = np.asarray(r['box'], np.float64)
    i = k2i[(r['seq'], r['frame'])]; src = idx['metas'][i]; assert metas[i] is None
    m = {k: src[k] for k in ('seq', 'frame', 'cls', 'K_in', 'K_raw', 'raw_wh', 'undistorted', 'raw_path') if k in src}
    m['boxes9d'] = np.concatenate([p[:3], lwh, p[6:9]])[None].astype(np.float64)
    m['names'] = ['drone']; m['qualified'] = np.array([True])
    metas[i] = m; valid.append(i)
out = {k: v for k, v in idx.items() if k != 'metas'}
out['metas'] = metas; out['valid_idx'] = np.asarray(sorted(valid))
out['ourdet_baseproto'] = dict(mode='naive', source=C['recs'], conf=CONF, s_bar=lwh.tolist())
pickle.dump(out, open(B + '/index_naive.pkl', 'wb'))
res = dict(combo=COMBO, mode='naive', n=len(valid), n_train=len(recs), index=B + '/index_naive.pkl', source=C['recs'], conf=CONF, s_bar=lwh.tolist())
if C['b4']:
    b4 = pickle.load(open(C['b4'], 'rb')); kb = {(m['seq'], m['frame']): m['boxes9d'] for m in b4['metas'] if m is not None}
    mine = {(m['seq'], m['frame']): m['boxes9d'] for m in metas if m is not None}
    assert set(kb) == set(mine), (len(kb), len(mine))
    res['b4_max_abs_diff'] = float(max(np.abs(np.asarray(kb[k], np.float64) - mine[k]).max() for k in mine))
json.dump(res, open(G + '/review1113/rgbst_equal/labels/%s_naive.json' % COMBO, 'w'), indent=1); print(res)
