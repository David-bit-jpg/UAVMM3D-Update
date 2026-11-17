# -*- coding: utf-8 -*-
"""[review1113 boxhull] put the box-hull "silhouettes" into <O>/ref/sim_sil.npy for the SAME simulated samples as the clean chain.
usage: python align_hull.py <O>      (<O> = review1113/boxhull/<combo>; needs ref/{sim_meta,sim_rgb,clean_sim_sil}.npy|pkl copied from the clean
                                       chain and ref/regen_sim_* from the patched `refiner_x.py sim`)
The regenerated sample list (same sim-only detector s0, same frames, same matching) must reproduce the clean list: same length, same
(seq, frame) order, crop width |dw| < 0.5 px and |d delta*| < 0.01 for every sample. Then the clean sim_rgb / sim_meta stay and only the
silhouette array is swapped (aligned). Otherwise (GPU non-determinism changed the sample set) the regenerated rgb / meta / hull set replaces
all three, consistently (logged as FALLBACK; BH_STRICT=1 makes this an error). Writes ref/align_hull.json."""
import os, sys, json, pickle
import numpy as np
O = sys.argv[1]; R = O + '/ref'
mc = pickle.load(open(R + '/sim_meta.pkl', 'rb')); mr = pickle.load(open(R + '/regen_sim_meta.pkl', 'rb'))
hull = np.load(R + '/regen_sim_sil_hull.npy'); sir = np.load(R + '/regen_sim_sil_ir.npy', mmap_mode='r'); sc = np.load(R + '/clean_sim_sil.npy', mmap_mode='r')
info = dict(n_clean=len(mc), n_regen=len(mr))
same = len(mc) == len(mr) and all(a['seq'] == b['seq'] and a['frame'] == b['frame'] for a, b in zip(mc, mr))
if same:
    dw = np.abs(np.array([a['w'] for a in mc]) - np.array([b['w'] for b in mr])); dd = np.abs(np.array([a['delta'] for a in mc]) - np.array([b['delta'] for b in mr]))
    info.update(max_dw_px=float(dw.max()), max_ddelta=float(dd.max()))
    same = bool(dw.max() < 0.5 and dd.max() < 0.01)
    # crop-geometry check: the regenerated IR silhouettes must reproduce the clean ones
    a = np.asarray(sc) > 128; b = np.asarray(sir) > 128; inter = (a & b).sum((1, 2)); uni = (a | b).sum((1, 2)); ok = uni > 0
    info.update(ir_sil_iou_median=float(np.median(inter[ok] / uni[ok])) if ok.any() else None, ir_sil_identical_frac=float(np.mean(np.all(a == b, axis=(1, 2)))))
if same:
    np.save(R + '/sim_sil.npy', hull); info['mode'] = 'ALIGNED (clean sim_rgb / sim_meta kept, silhouette array = box hull)'
else:
    if os.environ.get('BH_STRICT') == '1':
        json.dump(info, open(R + '/align_hull.json', 'w'), indent=1); sys.exit('regenerated sim samples differ from the clean chain (BH_STRICT=1): %s' % info)
    os.replace(R + '/sim_meta.pkl', R + '/clean_sim_meta.pkl'); os.replace(R + '/sim_rgb.npy', R + '/clean_sim_rgb.npy')
    np.save(R + '/sim_rgb.npy', np.load(R + '/regen_sim_rgb.npy')); pickle.dump(mr, open(R + '/sim_meta.pkl', 'wb')); np.save(R + '/sim_sil.npy', hull)
    info['mode'] = 'FALLBACK (regenerated rgb / meta / hull used together; clean copies renamed clean_sim_*)'
h = hull > 128; s = np.asarray(sc) > 128
info.update(hull_area_frac_mean=float(h.mean()), clean_ir_sil_area_frac_mean=float(s.mean()), clean_ir_sil_empty_frac=float(np.mean(~s.any((1, 2)))),
            hull_empty_frac=float(np.mean(~h.any((1, 2)))), composite_eligible_frac_hull=float(np.mean(hull.max((1, 2)) > 128)))
json.dump(info, open(R + '/align_hull.json', 'w'), indent=1); print('align_hull', json.dumps(info))
