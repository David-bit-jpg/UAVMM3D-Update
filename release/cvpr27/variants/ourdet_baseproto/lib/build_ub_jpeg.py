# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto, 11-17] full-resolution JPEG store of the RealDrone3D (code 'antifs') TRAIN frames for the upper-bound row
with view augmentation (VIEW_AUG). Same layout as tools/build_mm_cache.py --store jpeg (the sim cache format):
  rgb_jpg.bin (concatenated JPEG bytes, RGB channel order as cv2.imencode of an RGB array, quality 95 = builder default),
  rgb_jpg_index.npy (N,2 offset/length), index file with store='jpeg', H/W = raw 1080/1920, metas with K_in = K_raw (full-resolution
  pinhole intrinsics; raw frames are already undistorted, the 512x288 npy cache = INTER_AREA resize of them, checked bit-exact).
Labels = the same sealed-train-label index index_ub_gt.pkl (user-approved for the upper-bound row only); frames and boxes unchanged.
usage: python build_ub_jpeg.py <src npy cache train dir> <out train dir>"""
import os, sys, pickle, copy
import numpy as np, cv2

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
idx = pickle.load(open(os.path.join(src, 'index_ub_gt.pkl'), 'rb'))
metas = idx['metas']
n = len(metas)
offs = np.zeros((n, 2), np.int64)
pos = 0
with open(os.path.join(out, 'rgb_jpg.bin'), 'wb') as f:
    for i, m in enumerate(metas):
        im = cv2.imread(m['raw_path'], cv2.IMREAD_COLOR)
        assert im is not None and im.shape[:2] == (1080, 1920) and tuple(m['raw_wh']) == (1920, 1080), m['raw_path']
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        b = buf.tobytes(); f.write(b); offs[i] = (pos, len(b)); pos += len(b)
        if i % 500 == 0:
            print(i, n, pos / 1e9, flush=True)
np.save(os.path.join(out, 'rgb_jpg_index.npy'), offs)
nidx = copy.deepcopy(idx)
nidx['store'] = 'jpeg'; nidx['H'], nidx['W'] = 1080, 1920
for m in nidx['metas']:
    m['K_in'] = np.array(m['K_raw'], np.float64).copy()
nidx['ourdet_baseproto_ubzoom'] = 'full-res jpeg of the train frames, K_in = K_raw; labels = index_ub_gt.pkl (sealed train labels, UB row only)'
pickle.dump(nidx, open(os.path.join(out, 'index_ub_gt.pkl'), 'wb'))
print('done', n, pos / 1e9, 'GB')
