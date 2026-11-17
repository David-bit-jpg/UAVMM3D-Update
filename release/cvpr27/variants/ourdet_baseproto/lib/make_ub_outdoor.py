# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto] UPPER-BOUND row only (user-approved 11-16): OutdoorUAV3D real train labels from the SEALED train file
D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/train_gt.pkl (box9d_ours = the cache/scorer convention) -> index_ub_gt.pkl in the cache
COPY D:/S2MUAV3D/ourdet_baseproto_cache/outdoor_real_fs/train/. Used for nothing else."""
import pickle, json, numpy as np
B = 'D:/S2MUAV3D/ourdet_baseproto_cache/outdoor_real_fs/train'
idx = pickle.load(open(B + '/index.pkl', 'rb')); k2i = {(m['seq'], m['frame']): i for i, m in enumerate(idx['metas'])}
G = pickle.load(open('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/train_gt.pkl', 'rb'))
metas = [None] * len(idx['metas']); valid = []; kerr = 0.0
for g in G:
    i = k2i[(g['seq'], g['frame'])]; src = idx['metas'][i]
    K = np.asarray(g['K'], float); s = g['W'] / 512.0
    kerr = max(kerr, float(np.abs(np.asarray(src['K_raw'], float) - K).max()))
    m = {k: src[k] for k in ('seq', 'frame', 'K_in', 'K_raw', 'raw_wh', 'undistorted', 'raw_path') if k in src}
    m['boxes9d'] = np.asarray(g['box9d_ours'], np.float64)[None]; m['names'] = ['drone']; m['qualified'] = np.array([True])
    metas[i] = m; valid.append(i)
out = {k: v for k, v in idx.items() if k != 'metas'}; out['metas'] = metas; out['valid_idx'] = np.asarray(sorted(valid))
out['ourdet_baseproto'] = 'upper bound: sealed real train labels'
pickle.dump(out, open(B + '/index_ub_gt.pkl', 'wb'))
r = dict(n=len(valid), n_train=len(idx['metas']), K_raw_vs_label_K_max=kerr); print(r)
json.dump(r, open('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/labels/out2antifs_ub.json', 'w'))
