# -*- coding: utf-8 -*-
"""[review1113 hparam_1117] isolated lib copies of srcx/lib for ONE lambda run of the whole PST chain (lambda = fraction of the median
step-1 refiner bias removed in the Module-3 pseudo-box depth, make_pseudo_x.py LAM, deployed 0.75).
usage: python make_lam_lib.py <lam tag, e.g. 050> <tgt mav|antifs> <shift 0|10|20>   (idempotent; writes only hparam_1117/lam/<tgt>_L<tag>_s<shift>/lib)
Built like review1113/heads/make_heads_lib.py FULL mode (sup_patch: SX_SEEDSHIFT on every training seed, env batches for OOM halving, resume
states, sys.path -> copy); shift 0 = the nominal seeds of the reported run.
  sx.O -> hparam_1117/lam/<tgt>_L<tag>_s<shift>; sx.W -> D:/S2MUAV3D/srcx_work/hp_<C>_L<tag>_s<shift>;
  REAL_CFG / ST_CFG / PSEUDO_TAG -> *_hpL<tag>s<shift> (new names, nothing existing overwritten);
  make_pseudo_x.py: LAM = <tag>/100 (the ONLY method change). Everything else byte-identical to srcx/lib (+ sup_patch)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
sys.path.insert(0, GP + '/review1113')
import sup_patch as P  # noqa

SRC = {'mav': 'in', 'antifs': 'out'}
tag, tgt, shift = sys.argv[1], sys.argv[2], int(sys.argv[3])
lam = int(tag) / 100.0
assert tgt in SRC and shift in (0, 10, 20) and len(tag) == 3 and lam != 0.75, sys.argv
C = '%s2%s' % (SRC[tgt], tgt)
B = '%s/lam/%s_L%s_s%d' % (HERE, tgt, tag, shift)
W = 'D:/S2MUAV3D/srcx_work/hp_%s_L%s_s%d' % (C, tag, shift)
D = B + '/lib'
SUF = 'hpL%ss%d' % (tag, shift)

t = P.rd('sx.py')
t = P.rep(t, P.SX_OW, "O = '%s'          # [hparam lambda %.2f %s shift %d]\nW = '%s'\n" % (B, lam, tgt, shift, W))
t = P.rep(t, "REAL_CFG = CFGD + 'srcx/real_%s.yaml' % COMBO", "REAL_CFG = CFGD + 'srcx/real_%%s_%s.yaml' %% COMBO" % SUF)
t = P.rep(t, "ST_CFG = CFGD + 'srcx/st_%s.yaml' % COMBO", "ST_CFG = CFGD + 'srcx/st_%%s_%s.yaml' %% COMBO" % SUF)
t = P.rep(t, "PSEUDO_TAG = 'srcx_' + COMBO", "PSEUDO_TAG = 'srcx_' + COMBO + '_%s'" % SUF)
t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)",
          "assert (SRC, TGT, TEACH) == (%r, %r, 'full'), (SRC, TGT, TEACH)   # [hparam] this copy is for %s lambda %.2f s%d only\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)"
          % (SRC[tgt], tgt, tgt, lam, shift))
t = P.rep(t, "for d in (O, W):\n    os.makedirs(d, exist_ok=True)", "for d in ((O, W) if os.environ.get('SX_NOMKDIR') != '1' else ()):\n    os.makedirs(d, exist_ok=True)")
P.wr(D, 'sx.py', t, '[hparam lambda %.2f %s s%d] O -> %s, W -> %s, tags *_%s' % (lam, tgt, shift, B, W, SUF))
t = P.path(P.rd('make_pseudo_x.py'), D)
t = P.rep(t, "LAM, CONF, GUARD = 0.75, 0.5, float(np.log(1.4))", "LAM, CONF, GUARD = %r, 0.5, float(np.log(1.4))   # [hparam] lambda (deployed 0.75)" % lam)
P.wr(D, 'make_pseudo_x.py', t, 'LAM = %.2f (only change)' % lam)
P.wr(D, 'refiner_x.py', P.refiner(P.path(P.rd('refiner_x.py'), D), seedshift=True), 'SX_SEEDSHIFT on training seeds; env batches; resume states')
P.wr(D, 'ou_train_x.py', P.ou_train(P.path(P.rd('ou_train_x.py'), D), seedshift=True), 'SX_SEEDSHIFT on every RNG seed; resume state; recipe unchanged')
P.wr(D, 'ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'batch = $ORI_BS (default 16 = deployed)')
P.wr(D, 'det_infer_x.py', P.det_bs(P.path(P.rd('det_infer_x.py'), D)), 'batch = $DET_BS (default 8 = deployed)')
for f in ('chain_x.py', 'mkcfg.py', 'wsim_x.py', 'score_x.py', 'avg_inf.py', 'to_pred.py'):
    P.wr(D, f, P.path(P.rd(f), D), 'sys.path -> redirected sx; code unchanged')
P.infer_wrap(D)
for f in os.listdir(D):
    if f.endswith('.py'):
        s = open(D + '/' + f, encoding='utf-8').read()
        assert 'gproto/srcx/lib' not in s.split('\n', 1)[1], f
print('ok lambda', lam, tgt, shift, D, SUF)
