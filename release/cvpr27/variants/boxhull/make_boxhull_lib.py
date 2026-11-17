# -*- coding: utf-8 -*-
"""[review1113 boxhull] isolated copies of srcx/lib for the BOX-HULL rendered-truth control of the depth refiner (Module 4).
usage: python make_boxhull_lib.py           (idempotent; writes only review1113/boxhull/lib)

Control: the refiner's IR-silhouette target (sim_sil.npy, built by `refiner_x.py sim` = softkd_extract.sil_full: IR pixels above half the
hull/ring contrast INSIDE the projected-GT-box convex hull) is replaced by the FILLED convex hull of the projected 3-D GT box (255 inside),
cropped with the same deployment crop (crop_at(cx, cy, w) of the matched sim detection) for the same samples. Everything else = clean chain.

Patches (all anchors asserted; method code otherwise byte-identical to srcx/lib):
  sx.py        O -> review1113/boxhull/<combo>, W -> O/work (nothing under srcx/ or D:/S2MUAV3D/srcx_work is written)
  refiner_x.py sup_patch.refiner (env batches REF_BS / REF_ST1_BS / REF_TRAIN_BS for OOM halving, resume states; NO seed shift),
               cmd_sim: sim-loader batch = $REF_SIM_BS (default 8 = deployed), and it ALSO builds the box-hull crop of every sample;
               it writes regen_sim_{rgb,sil_ir,sil_hull}.npy + regen_sim_meta.pkl + regen_sim_geom.pkl instead of sim_*.
               (align_hull.py then puts the hull array in ref/sim_sil.npy, aligned to the clean chain's sim samples.)
  ori_x.py     batch = $ORI_BS (unused here: chain_x only imports ori_x for its zL ensemble helper)
  chain_x / score_x / avg_inf / to_pred: sys.path -> this copy only."""
import os, sys
HERE = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull'
sys.path.insert(0, os.path.dirname(HERE))
import sup_patch as P  # noqa

D = HERE + '/lib'
t = P.rd('sx.py')
t = P.rep(t, P.SX_OW, "O = '%s/' + COMBO          # [boxhull] control outputs\nW = O + '/work'\nO_ORIG = HOME + '/' + COMBO; W_ORIG = 'D:/S2MUAV3D/srcx_work/' + COMBO   # read only\n" % HERE)
t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)", "assert TEACH == 'full', TEACH   # [boxhull] control of the PST (full) chain only\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)")
P.wr(D, 'sx.py', t, 'outputs -> review1113/boxhull/<combo> (+ /work)')

t = P.refiner(P.path(P.rd('refiner_x.py'), D), seedshift=False)
t = P.rep(t, "ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=lg, training=False)",
          "ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=int(os.environ.get('REF_SIM_BS', '8')), dist=False, workers=0, logger=lg, training=False)")
t = P.rep(t, "    rgbs, sils, meta = [], [], []\n    for r in recs:", "    rgbs, sils, meta = [], [], []; hulls, geom = [], []   # [boxhull]\n    for r in recs:")
t = P.rep(t, """            sv = crop_at(s, cx, cy, w) if s is not None else np.zeros((128, 128), np.uint8)
            rgbs.append(v); sils.append(sv)""", """            sv = crop_at(s, cx, cy, w) if s is not None else np.zeros((128, 128), np.uint8)
            hf = np.zeros(ir.shape[:2], np.uint8); cv2.fillConvexPoly(hf, cv2.convexHull(np.round(uvg).astype(np.int32)), 255)   # [boxhull] filled hull of the projected GT box
            hulls.append(crop_at(hf, cx, cy, w)); geom.append((cx, cy, w))
            rgbs.append(v); sils.append(sv)""")
t = P.rep(t, "    np.save(RO + '/sim_rgb.npy', np.stack(rgbs)); np.save(RO + '/sim_sil.npy', np.stack(sils)); pickle.dump(meta, open(RO + '/sim_meta.pkl', 'wb'))",
          "    np.save(RO + '/regen_sim_rgb.npy', np.stack(rgbs)); np.save(RO + '/regen_sim_sil_ir.npy', np.stack(sils)); np.save(RO + '/regen_sim_sil_hull.npy', np.stack(hulls))\n"
          "    pickle.dump(geom, open(RO + '/regen_sim_geom.pkl', 'wb')); pickle.dump(meta, open(RO + '/regen_sim_meta.pkl', 'wb'))   # [boxhull] never overwrites sim_*")
P.wr(D, 'refiner_x.py', t, 'env batches + resume (sup_patch, no seed shift); sim mode also writes the box-hull crops (regen_sim_*)')
P.wr(D, 'ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'batch = $ORI_BS (default 16 = deployed)')
for f in ('chain_x.py', 'score_x.py', 'avg_inf.py', 'to_pred.py'):
    P.wr(D, f, P.path(P.rd(f), D), 'sys.path -> redirected sx; code unchanged')
P.infer_wrap(D)
for f in os.listdir(D):
    if f.endswith('.py'):
        assert 'gproto/srcx/lib' not in open(D + '/' + f, encoding='utf-8').read().split('\n', 1)[1], f
print('ok', D)
