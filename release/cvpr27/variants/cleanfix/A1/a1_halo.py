# -*- coding: utf-8 -*-
"""[cleanfix A1] Halo share removed by the LiDAR veto, on the simulated orientation crops the CLEAN chain actually composites with.
The clean students (srcx/lib/ou_train_x.py, KP.CacheDS with MATTE='v') read precomputed alphas from sx.S[src]['sim_crops']/a_v.npy
(IR+LiDAR cross-verified) and a_ir.npy (IR only), both written by wf20/design/extract_kz_crops.py with kzpaste_train.matte() and stored as
uint8(255 x Gaussian-feathered binary mask, sigma 1 px). Removed halo per crop = |M_ir \ M_v| / |M_ir| with M = alpha >= 128 (recovers the
binary mask before feathering up to boundary pixels); computed on crops with has_matte. Dev figure (X_suppl:276, 1.3 %) = median of
'halo_bg' in wf20/design/matte_stats_test.json (prep_sim.py, 552 sim TEST crops, same rules)."""
import json, pickle, sys
import numpy as np
OLDSCR = 'C:/Users/jiang/AppData/Local/Temp/claude/0fa2f3b7e932577a/26739ba4-3497-41f0-8f86-fdad1ac856ee/scratchpad'
SETS = {'in (MAV6D chain, crops_6000)': OLDSCR + '/kz/crops_6000', 'out (AntiUAV3D chain, crops_5367)': 'D:/S2MUAV3D/pcmd_out_work/crops_5367'}
res = {}
for name, d in SETS.items():
    meta = pickle.load(open(d + '/meta.pkl', 'rb'))
    AV = np.load(d + '/a_v.npy', mmap_mode='r'); AI = np.load(d + '/a_ir.npy', mmap_mode='r')
    fr, anyrm = [], []
    for i, m in enumerate(meta):
        if not m.get('has_matte'):
            continue
        mi = np.asarray(AI[i]) >= 128; mv = np.asarray(AV[i]) >= 128
        a = int(mi.sum())
        if a == 0:
            continue
        r = float((mi & ~mv).sum()) / a
        fr.append(r)
    fr = np.array(fr)
    res[name] = dict(n_crops=len(meta), n_matte=int(len(fr)), median_pct=100 * float(np.median(fr)), mean_pct=100 * float(fr.mean()),
                     p90_pct=100 * float(np.percentile(fr, 90)), frac_gt10pct=100 * float((fr > 0.10).mean()), frac_zero=100 * float((fr == 0).mean()))
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/A1/a1_halo.json', 'w'), indent=1)
