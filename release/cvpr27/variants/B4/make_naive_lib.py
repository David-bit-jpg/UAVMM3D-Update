# -*- coding: utf-8 -*-
"""[review1113 B4] isolated copies of srcx/lib for the naive self-training baseline (in2mav, TEACH=full, no seed shift) -> review1113/B4/lib.
  sx.py            O -> review1113/B4/run, W -> D:/S2MUAV3D/srcx_work/in2mav_naive, REAL_CFG / ST_CFG -> srcx/real_in2mav_naive.yaml /
                   srcx/st_in2mav_naive.yaml, PSEUDO_TAG -> srcx_in2mav_naive (=> E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav_naive.pkl,
                   detectors output/models/uavdet_3d/camnorm/srcx/st_in2mav_naive/DET1_s{0,1,2}); all new names, nothing existing is overwritten.
                   + PSEUDO_TAG_ST3 = 'srcx_in2mav' (the reported run's IR-corrected Module-3 labels, read only; used ONLY by chain_x targets
                   when the rest of the PST chain is run on top of the naive detector, so that steps 2-3 are exactly PST's).
  make_pseudo_naive.py   = make_pseudo_x.py with the refiner correction removed: refiner file 'none' -> no delta, c_bar = 0; frames with
                   conf >= 0.5 keep the sim-only detector's OWN box (t_det, s_bar, R_det). Same frames / size / rotation as PST's Module-3 labels;
                   only the IR-refiner depth correction is gone.
  chain_x.py       targets read index_pseudo_<PSEUDO_TAG_ST3> (see above); deploy unchanged.
  refiner_x / ou_train_x / ori_x / det_infer_x / the rest: sup_patch (env batch sizes for OOM halving, resume states, sys.path -> copy);
                   NO seed shift (seeds = the chain's: refiner 5/6, detectors 0/1/2, students 0/1).
Run: python make_naive_lib.py   (idempotent; SX_NOMKDIR=1 skips the import-time makedirs)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sup_patch as P  # noqa

B = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/run'
D = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/lib'
WK = 'D:/S2MUAV3D/srcx_work/in2mav_naive'

t = P.rd('sx.py')
t = P.rep(t, P.SX_OW, "O = '%s'          # [B4] naive self-training baseline\nW = '%s'\nO_ORIG = HOME + '/' + COMBO; W_ORIG = 'D:/S2MUAV3D/srcx_work/' + COMBO   # read only\n" % (B, WK))
t = P.rep(t, "REAL_CFG = CFGD + 'srcx/real_%s.yaml' % COMBO", "REAL_CFG = CFGD + 'srcx/real_%s_naive.yaml' % COMBO")
t = P.rep(t, "ST_CFG = CFGD + 'srcx/st_%s.yaml' % COMBO", "ST_CFG = CFGD + 'srcx/st_%s_naive.yaml' % COMBO")
t = P.rep(t, "PSEUDO_TAG = 'srcx_' + COMBO", "PSEUDO_TAG = 'srcx_' + COMBO + '_naive'\nPSEUDO_TAG_ST3 = 'srcx_' + COMBO          # [B4] PST Module-3 labels (read only), stage-3 targets only")
t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)",
          "assert (SRC, TGT, TEACH) == ('in', 'mav', 'full'), (SRC, TGT, TEACH)   # [B4] this copy is for in2mav only\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)")
t = P.rep(t, "for d in (O, W):\n    os.makedirs(d, exist_ok=True)", "for d in ((O, W) if os.environ.get('SX_NOMKDIR') != '1' else ()):\n    os.makedirs(d, exist_ok=True)")
P.wr(D, 'sx.py', t, 'outputs -> %s + %s; configs / pseudo tag *_naive; PSEUDO_TAG_ST3 = PST labels for stage 3' % (B, WK))
P.wr(D, 'refiner_x.py', P.refiner(P.path(P.rd('refiner_x.py'), D)), 'env batches; resume states; no seed shift')
P.wr(D, 'ou_train_x.py', P.ou_train(P.path(P.rd('ou_train_x.py'), D)), 'resume state; recipe unchanged; no seed shift')
P.wr(D, 'ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'batch = $ORI_BS (default 16 = deployed)')
P.wr(D, 'det_infer_x.py', P.det_bs(P.path(P.rd('det_infer_x.py'), D)), 'batch = $DET_BS (default 8 = deployed)')
t = P.path(P.rd('chain_x.py'), D)
t = P.rep(t, "ix = pickle.load(open('%s/train/index_pseudo_%s.pkl' % (sx.REAL, sx.PSEUDO_TAG), 'rb'))",
          "ix = pickle.load(open('%s/train/index_pseudo_%s.pkl' % (sx.REAL, sx.PSEUDO_TAG_ST3), 'rb'))   # [B4] PST Module-3 labels")
P.wr(D, 'chain_x.py', t, 'sys.path -> redirected sx; stage-3 targets use the PST Module-3 labels (PSEUDO_TAG_ST3); deploy unchanged')
t = P.path(P.rd('make_pseudo_x.py'), D)
t = P.rep(t, "U = pickle.load(open(sys.argv[2], 'rb'))", "U = ({} if sys.argv[2] == 'none' else pickle.load(open(sys.argv[2], 'rb')))")   # [B4] 'none' = no IR refiner (no comment: same line as TAG)
t = P.rep(t, "cbar = float(np.median(d[np.abs(d) <= GUARD]))", "cbar = float(np.median(d[np.abs(d) <= GUARD])) if len(d) else 0.0")
t = P.rep(t, "info = dict(tag=TAG, n=len(valid), n_corrected=ncor,", "assert sys.argv[2] != 'none' or ncor == 0\ninfo = dict(tag=TAG, refiner=sys.argv[2], n=len(valid), n_corrected=ncor,")
P.wr(D, 'make_pseudo_naive.py', t, "make_pseudo_x with refiner 'none': own detector box for conf >= 0.5 frames, no IR-refiner depth correction")
for f in ('mkcfg.py', 'wsim_x.py', 'score_x.py', 'avg_inf.py', 'to_pred.py'):
    P.wr(D, f, P.path(P.rd(f), D), 'sys.path -> redirected sx; code unchanged')
P.infer_wrap(D)
for f in os.listdir(D):
    if f.endswith('.py'):
        assert 'gproto/srcx/lib' not in open(D + '/' + f, encoding='utf-8').read().split('\n', 1)[1], f
print('ok', D)
