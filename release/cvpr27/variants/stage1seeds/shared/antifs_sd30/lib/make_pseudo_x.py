# [review1113 copy of srcx/lib/make_pseudo_x.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/antifs_sd30/lib] sys.path -> redirected sx; code unchanged
# -*- coding: utf-8 -*-
"""[outdoor->srcx] Module 3 pseudo-labels (no labels read; copy of the rule in nolabel/make_nolabel_pseudo.py):
t_pl = t_det * exp(delta_hat - LAM * c_bar)  (LAM = 0.75 frozen; c_bar = median delta_hat over train frames with |delta_hat| <= log 1.4;
frames with |delta_hat| > log 1.4 or no crop keep the detector depth); keep frames with conf >= 0.5; pseudo box = (t_pl, s_bar, R_det).
usage: python make_pseudo_out.py <det recs.pkl> <refiner infer train.pkl> <tag>  -> D:/mmcache/outdoor_real_out/train/index_pseudo_<tag>.pkl"""
import pickle, sys, json, numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/antifs_sd30/lib'); import sx  # noqa  [srcx] per source/target constants
LAM, CONF, GUARD = 0.75, 0.5, float(np.log(1.4))
D = pickle.load(open(sys.argv[1], 'rb')); U = pickle.load(open(sys.argv[2], 'rb')); TAG = sys.argv[3]
recs = D['recs']['train']; lwh = np.asarray(D['size'], np.float64)
d = np.array([U[(r['seq'], r['frame'])]['d'] for r in recs if (r['seq'], r['frame']) in U])
cbar = float(np.median(d[np.abs(d) <= GUARD]))
B = sx.REAL + '/train'
idx = pickle.load(open(B + '/index.pkl', 'rb'))
k2i = {(m['seq'], m['frame']): i for i, m in enumerate(idx['metas'])}
metas = [None] * len(idx['metas']); valid = []; ncor = 0
for r in recs:
    if r['box'] is None or r['conf'] < CONF:
        continue
    p = np.asarray(r['box'], np.float64).copy(); u = U.get((r['seq'], r['frame']))
    if u is not None and abs(u['d']) <= GUARD:
        p[:3] *= np.exp(u['d'] - LAM * cbar); ncor += 1
    i = k2i[(r['seq'], r['frame'])]; src = idx['metas'][i]
    m = {k: src[k] for k in ('seq', 'frame', 'K_in', 'K_raw', 'raw_wh', 'undistorted', 'raw_path') if k in src}   # [11-13] MAV6D metas have no raw_path
    m['boxes9d'] = np.concatenate([p[:3], lwh, p[6:9]]).astype(np.float32)[None]
    m['names'] = ['drone']; m['qualified'] = np.array([True]); m['pseudo'] = True
    metas[i] = m; valid.append(i)
out = {k: v for k, v in idx.items() if k != 'metas'}
out['metas'] = metas; out['valid_idx'] = np.asarray(sorted(valid)); out['pseudo'] = dict(lam=LAM, c_bar=cbar, conf=CONF, calib_lwh=lwh.tolist())
pickle.dump(out, open('%s/index_pseudo_%s.pkl' % (B, TAG), 'wb'))
info = dict(tag=TAG, n=len(valid), n_corrected=ncor, c_bar=cbar, lam=LAM, n_train=len(recs))
json.dump(info, open('%s/pseudo_%s.json' % (sx.O, TAG), 'w'), indent=1); print(info)
