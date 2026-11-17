# -*- coding: utf-8 -*-
"""[review1113 stage1seeds] isolated copies of srcx/lib for ONE full-reseed run (stage 1 re-trained with seed set SD, downstream SX_SEEDSHIFT=SD).
usage: python make_s1_lib.py <group prep|full|E1|E4> <tgt mav|antifs> <sd 30|40>      (idempotent; writes only <run dir>/lib)
Built on review1113/sup_patch.py exactly as seeds/make_seeds_lib.py (PST) and sensorfree/make_sf_lib.py (E1 / E4); the ONLY additions:
  * sx.SOURCES[src] stage-1 paths -> the re-seeded networks of seed set SD (s1paths.nets: det x3, IR+LiDAR x2, IR-distilled x2, LiDAR x2,
    height net, plain-RGB x2), asserted to differ from the reported ones;
  * group 'prep' (shared, per target and seed set): teach=full copy whose `refiner_x.py sim` ALSO writes ref/sim_hull.npy = filled convex hull
    of the projected 3-D GT box, cropped with the same deployment crop of the same samples (= review1113/boxhull patch, but next to the
    normal sim_* files instead of regen_*). E1 uses it as its refiner silhouette/matte array: aligned by construction (same pass).
Groups:
  full  = PST (SX_TEACH=full): refiner / students seed-shifted (sup_patch), env batches, resume states.            names *_s1fullsd<SD>
  E1    = sensor-free control (SX_TEACH=none + box-hull mattes): refiner = make_sf_lib E1 patch (matte hook, SF_COMPOSITE=True, SILW=1.0),
          students = make_sf_lib E1 patch (composite alpha = D:/S2MUAV3D/srcx_work/sensorfree/hull_<src>.npy).        names *_s1E1sd<SD>
  E4    = pure RGB self-training (SX_TEACH=none, no composites: refiner none-arm unchanged, students run with --paste 0).  names *_s1E4sd<SD>
Outputs: O = review1113/stage1seeds/runs/<tgt>_<group>_s1sd<SD> (prep: stage1seeds/shared/<tgt>_sd<SD>), W = D:/S2MUAV3D/srcx_work/s1seeds/<same>;
configs srcx/{real,st}_<C><suffix>.yaml, pseudo tag srcx_<C><suffix>, self-trained detectors output/models/.../srcx/st_<C><suffix>/."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
sys.path.insert(0, GP + '/review1113'); sys.path.insert(0, GP + '/review1113/sensorfree'); sys.path.insert(0, HERE)
import sup_patch as P  # noqa
import make_sf_lib as SF  # noqa  (constants only; its main is guarded)
import s1paths as SP  # noqa

SRCOF = {'mav': 'in', 'antifs': 'out'}
TEACH = {'prep': 'full', 'full': 'full', 'E1': 'none', 'E4': 'none'}


def names(group, tgt, sd):
    src = SRCOF[tgt]; C = '%s2%s' % (src, tgt) + ('' if TEACH[group] == 'full' else '_none')
    suf = '_s1%ssd%d' % (group, sd)
    run = ('%s_sd%d' % (tgt, sd)) if group == 'prep' else ('%s_%s_s1sd%d' % (tgt, group, sd))
    B = HERE + ('/shared/' if group == 'prep' else '/runs/') + run
    W = 'D:/S2MUAV3D/srcx_work/s1seeds/' + ('prep_' if group == 'prep' else '') + run
    return src, C, suf, run, B, W


HULL_A = "    rgbs, sils, meta = [], [], []\n    for r in recs:"
HULL_B = "    rgbs, sils, meta = [], [], []; hulls = []   # [stage1seeds prep] + box hull per sample\n    for r in recs:"
HULL_C = """            sv = crop_at(s, cx, cy, w) if s is not None else np.zeros((128, 128), np.uint8)
            rgbs.append(v); sils.append(sv)"""
HULL_D = """            sv = crop_at(s, cx, cy, w) if s is not None else np.zeros((128, 128), np.uint8)
            hf = np.zeros(ir.shape[:2], np.uint8); cv2.fillConvexPoly(hf, cv2.convexHull(np.round(uvg).astype(np.int32)), 255)   # [stage1seeds prep] = boxhull patch
            hulls.append(crop_at(hf, cx, cy, w))
            rgbs.append(v); sils.append(sv)"""
HULL_E = "    np.save(RO + '/sim_rgb.npy', np.stack(rgbs)); np.save(RO + '/sim_sil.npy', np.stack(sils)); pickle.dump(meta, open(RO + '/sim_meta.pkl', 'wb'))"
HULL_F = HULL_E + "\n    np.save(RO + '/sim_hull.npy', np.stack(hulls))   # [stage1seeds prep] box hull aligned to sim_meta (same pass); used by E1 only"
SIMBS_A = "ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=lg, training=False)"
SIMBS_B = "ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=int(os.environ.get('REF_SIM_BS', '8')), dist=False, workers=0, logger=lg, training=False)"

if __name__ == '__main__':
    group, tgt, sd = sys.argv[1], sys.argv[2], int(sys.argv[3])
    assert group in TEACH and tgt in SRCOF and sd in (30, 40), sys.argv
    src, C, suf, run, B, W = names(group, tgt, sd)
    teach = TEACH[group]; D = B + '/lib'
    NETS = SP.nets(src, sd)

    t = P.rd('sx.py')
    t = P.rep(t, P.SX_OW, "O = '%s'          # [stage1seeds] %s %s seed set %d\nW = '%s'\n" % (B, group, tgt, sd, W))
    t = P.rep(t, "REAL_CFG = CFGD + 'srcx/real_%s.yaml' % COMBO", "REAL_CFG = CFGD + 'srcx/real_%%s%s.yaml' %% COMBO" % suf)
    t = P.rep(t, "ST_CFG = CFGD + 'srcx/st_%s.yaml' % COMBO", "ST_CFG = CFGD + 'srcx/st_%%s%s.yaml' %% COMBO" % suf)
    t = P.rep(t, "PSEUDO_TAG = 'srcx_' + COMBO", "PSEUDO_TAG = 'srcx_' + COMBO + '%s'" % suf)
    t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)",
              "assert (SRC, TGT, TEACH) == (%r, %r, %r), (SRC, TGT, TEACH)   # [stage1seeds] this copy is for %s only\n"
              "_REP = dict(SOURCES[%r])   # reported stage-1 networks (read only, for the assertion below)\n"
              "SOURCES[%r] = dict(SOURCES[%r], **json.loads(%r))   # [stage1seeds] RE-SEEDED stage-1 networks, seed set %d (s1paths.nets)\n"
              "assert all(SOURCES[%r][k] != _REP[k] for k in ('det', 'irlid', 'kdir', 'lid', 'kps')), 'stage-1 paths not replaced'\n"
              "S1_SEEDSET = %d; S1_GROUP = %r\n"
              "S = SOURCES.get(SRC); T = TARGETS.get(TGT)" % (src, tgt, teach, run, src, src, src, json.dumps(NETS), sd, src, sd, group))
    t = P.rep(t, "for d in (O, W):\n    os.makedirs(d, exist_ok=True)", "for d in ((O, W) if os.environ.get('SX_NOMKDIR') != '1' else ()):\n    os.makedirs(d, exist_ok=True)")
    P.wr(D, 'sx.py', t, '[stage1seeds %s %s sd%d teach %s] O -> %s, W -> %s; stage-1 nets re-seeded' % (group, tgt, sd, teach, B, W))

    # ---- refiner
    t = P.refiner(P.path(P.rd('refiner_x.py'), D), seedshift=True)
    if group == 'prep':
        t = P.rep(t, SIMBS_A, SIMBS_B); t = P.rep(t, HULL_A, HULL_B); t = P.rep(t, HULL_C, HULL_D); t = P.rep(t, HULL_E, HULL_F)
        hdr = 'env batches (+REF_SIM_BS); sim mode also writes ref/sim_hull.npy (box hull, same samples)'
    elif group == 'E1':   # = make_sf_lib.py arm E1, byte-identical patch text
        t = P.rep(t, SF.SB_OLD, SF.SB_NEW)
        t = P.rep(t, "            a = m[n][..., None]", "            a = mt[n][..., None]   # [sensorfree] alpha from the matte array")
        t = P.rep(t, "def sim_batch(", SF.SB_DEF)
        SW = "SILW = 0.0 if sx.NOIR else 1.0   # [11-12] no-teacher control: IR silhouette losses off without IR"
        t = P.rep(t, SW, SW + "\nSF_COMPOSITE = True   # [sensorfree] composites without IR (E1: box-hull matte)")
        t = t.replace(SW, SW + "\nSILW = 1.0   # [sensorfree E1, 11-15] region head supervised by the box hull (ref/sim_sil.npy = hull); no sensor", 1)
        hdr = 'SX_SEEDSHIFT; env batches; resume; [sensorfree E1] matte hook, SF_COMPOSITE=True, SILW=1.0 (ref/sim_sil.npy = box hull)'
    else:
        hdr = 'SX_SEEDSHIFT on training seeds; env batches; resume states'
    P.wr(D, 'refiner_x.py', t, hdr)

    # ---- orientation students
    t = P.ou_train(P.path(P.rd('ou_train_x.py'), D), seedshift=True)
    if group == 'E1':
        anchor = "src = (FgCacheDS if (a.fg_col > 0 or a.fg_blur > 0 or a.proto_xp) else KP.CacheDS)(cfg.DATA_CONFIG, sx.S['sim_crops'], training=True, logger=logger, xbg=xbg, seed=SEED)\n"
        t = P.rep(t, anchor, anchor + (
            "    _HM = %r %% sx.SRC   # [sensorfree E1] student composite alpha = filled projected-GT-box hull (prep_hull.py), not the IR x LiDAR matte\n"
            "    _H = np.load(_HM, mmap_mode='r'); _R = np.load(sx.S['sim_crops'] + '/rgb.npy', mmap_mode='r')\n"
            "    assert _H.shape == _R.shape[:3] and _H.shape[0] == len(src.cmeta), (_H.shape, _R.shape, len(src.cmeta))\n"
            "    src._c = {'rgb': _R, 'a_v': _H, 'a_ir': _H}   # CacheDS loads _c lazily only when None (workers 0) -> the cache's a_v / a_ir are never read\n"
            "    logger.info('[sensorfree E1] student composite matte = %%s (box hull), paste prob %%.2f' %% (_HM, a.paste))\n") % SF.HULL_STU)
    P.wr(D, 'ou_train_x.py', t, 'SX_SEEDSHIFT on every RNG seed; resume state; recipe unchanged' + ('; [E1] composite matte = box hull' if group == 'E1' else ''))
    P.wr(D, 'ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'batch = $ORI_BS (default 16 = deployed)')
    P.wr(D, 'det_infer_x.py', P.det_bs(P.path(P.rd('det_infer_x.py'), D)), 'batch = $DET_BS (default 8 = deployed)')
    for f in ('chain_x.py', 'make_pseudo_x.py', 'mkcfg.py', 'wsim_x.py', 'score_x.py', 'avg_inf.py', 'to_pred.py'):
        P.wr(D, f, P.path(P.rd(f), D), 'sys.path -> redirected sx; code unchanged')
    P.infer_wrap(D)
    for f in os.listdir(D):
        if f.endswith('.py'):
            assert 'gproto/srcx/lib' not in open(D + '/' + f, encoding='utf-8').read().split('\n', 1)[1], f
    print('ok', group, tgt, sd, run, D)
