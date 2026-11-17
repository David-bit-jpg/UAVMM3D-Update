# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto] pseudo-label indexes for OUR detector under the baselines' protocol (no real label is read).
  pst <combo>          : the baselines' +PST table, unchanged frames and boxes (center, lwh, euler_xyz) -> index_pst.pkl
                         in2mav     : D:/UAV3D_pack_fix_1114/data/pcmd_teacher_pseudo_real_train_clean.pkl (5,227 frames)
                         out2antifs : D:/UAV3D_pack_update_1112/data/cmpfull/pcmd_teacher_pseudo_outdoor_train.pkl (249 frames)
  st <combo> <recs.pkl>: = cmpfull/make_pseudo.py rule on our seed-0 sim-only detector: top-1 box per real train frame, keep frames whose
                         score >= quantile(scores, 1 - 0.8) (top 80 %), center / size / orientation all from its own box -> index_st.pkl
Index = copy of the target cache's train index.pkl with metas[i]['boxes9d'] replaced (float64, so the round trip is exact; the dataset
casts to float32 exactly as for every other label), valid_idx = kept frames, no 'pseudo' flag (plain labels: every head supervised).
Written into the cache COPY D:/S2MUAV3D/ourdet_baseproto_cache/<cache>/train/ (original caches untouched).
Round-trip check: 8 box corners projected with the baselines' labels.pkl K, table box vs box read back from the written index, max |du,dv|;
plus the camera check K_in (cache) vs labels K (MAV6D identical; AntiUAV3D: K_in = K_raw * 512/1920 with the half-pixel centre convention)."""
import json, os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

COMBO = sys.argv[2]; MODE = sys.argv[1]
C = {'in2mav': dict(cache='F:/ourdet_baseproto_cache/mav6d_cn_obj', table='D:/UAV3D_pack_fix_1114/data/pcmd_teacher_pseudo_real_train_clean.pkl',
                    labels='D:/cmpfull/real_train/labels.pkl', n_pst=5227, n_train=12866),
     'out2antifs': dict(cache='D:/S2MUAV3D/ourdet_baseproto_cache/outdoor_real_fs', table='D:/UAV3D_pack_update_1112/data/cmpfull/pcmd_teacher_pseudo_outdoor_train.pkl',
                        labels='D:/UAV3D_pack_update_1112/data/cmpfull/outdoor_train/labels.pkl', n_pst=249, n_train=3293)}[COMBO]
OUT = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/labels'
P8 = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])


def corners(center, lwh, eul, K):
    X = (Rot.from_euler('xyz', eul).as_matrix() @ (P8 * lwh).T).T + center
    uv = (K @ X.T).T
    return uv[:, :2] / uv[:, 2:]


B = C['cache'] + '/train'
idx = pickle.load(open(B + '/index.pkl', 'rb'))
assert len(idx['metas']) == C['n_train']
k2i = {(m['seq'], m['frame']): i for i, m in enumerate(idx['metas'])}
LAB = {(l['seq'], l['frame']): l for l in pickle.load(open(C['labels'], 'rb'))}
assert set(LAB) == set(k2i), 'labels.pkl frames != cache train frames'

# 11-17 pstu: AntiUAV3D table without the agreement-check filter (909 frames, same rule as MAV6D) -> index_pstu.pkl
if MODE == 'pstu':
    assert COMBO == 'out2antifs'
    C['table'] = 'D:/UAV3D_pack_fix_1117/data/pcmd_teacher_pseudo_outdoor_train_ungated.pkl'; C['n_pst'] = 909
# 11-17b pstg: MAV6D table restricted to frames passing the agreement check (2,126 frames) -> index_pstg.pkl
if MODE == 'pstg':
    assert COMBO == 'in2mav'
    C['table'] = 'D:/UAV3D_pack_fix_1117c/data/pcmd_teacher_pseudo_real_train_clean_gated.pkl'; C['n_pst'] = 2126
if MODE in ('pst', 'pstu', 'pstg'):
    rows = [(r['seq'], r['frame'], np.asarray(r['center'], np.float64), np.asarray(r['lwh'], np.float64), np.asarray(r['euler_xyz'], np.float64))
            for r in pickle.load(open(C['table'], 'rb'))]
    assert len(rows) == C['n_pst'], len(rows)
    info = dict(source=C['table'])
else:
    R = pickle.load(open(sys.argv[3], 'rb'))['recs']['train']
    assert len(R) == C['n_train']
    sc = np.array([r['conf'] if r['box'] is not None else -np.inf for r in R])
    thr = float(np.quantile(sc[np.isfinite(sc)], 1 - 0.8))
    rows = [(r['seq'], r['frame'], np.asarray(r['box'][:3], np.float64), np.asarray(r['box'][3:6], np.float64), np.asarray(r['box'][6:9], np.float64))
            for r, s in zip(R, sc) if r['box'] is not None and s >= thr]
    info = dict(source=sys.argv[3], score_threshold=thr, keep=0.8, n_no_box=int((~np.isfinite(sc)).sum()))

metas = [None] * len(idx['metas']); valid = []
for seq, fr, c, s, e in rows:
    i = k2i[(seq, fr)]; src = idx['metas'][i]; assert metas[i] is None, 'duplicate frame'
    m = {k: src[k] for k in ('seq', 'frame', 'cls', 'K_in', 'K_raw', 'raw_wh', 'undistorted', 'raw_path') if k in src}
    m['boxes9d'] = np.concatenate([c, s, e])[None].astype(np.float64)
    m['names'] = ['drone']; m['qualified'] = np.array([True])
    metas[i] = m; valid.append(i)
out = {k: v for k, v in idx.items() if k != 'metas'}
out['metas'] = metas; out['valid_idx'] = np.asarray(sorted(valid)); out['ourdet_baseproto'] = dict(mode=MODE, **info)
name = 'index_%s.pkl' % MODE
pickle.dump(out, open(B + '/' + name, 'wb'))

# ---- round trip (read the written file back)
back = pickle.load(open(B + '/' + name, 'rb'))
err, err32, kerr = 0.0, 0.0, 0.0
for seq, fr, c, s, e in rows:
    m = back['metas'][k2i[(seq, fr)]]; L = LAB[(seq, fr)]; K = np.asarray(L['K'], np.float64)
    b = np.asarray(m['boxes9d'], np.float64).reshape(9)
    u0 = corners(c, s, e, K)
    err = max(err, float(np.abs(corners(b[:3], b[3:6], b[6:9], K) - u0).max()))
    b32 = b.astype(np.float32).astype(np.float64)        # what the dataset hands to the encoder
    err32 = max(err32, float(np.abs(corners(b32[:3], b32[3:6], b32[6:9], K) - u0).max()))
    Kin = np.asarray(m['K_in'], np.float64); sw = L['W'] / 512.0
    Kexp = K.copy() if L['W'] == 512 else np.array([[K[0, 0] / sw, 0, (K[0, 2] + .5) / sw - .5], [0, K[1, 1] / sw, (K[1, 2] + .5) / sw - .5], [0, 0, 1]])
    kerr = max(kerr, float(np.abs(Kin - Kexp).max()))
res = dict(combo=COMBO, mode=MODE, n=len(valid), n_train=len(idx['metas']), index=B + '/' + name,
           roundtrip_max_px=err, roundtrip_float32_max_px=err32, K_in_vs_labels_K_max=kerr, **info)
assert err < 1e-6, res
json.dump(res, open('%s/%s_%s.json' % (OUT, COMBO, MODE), 'w'), indent=1); print(res)
