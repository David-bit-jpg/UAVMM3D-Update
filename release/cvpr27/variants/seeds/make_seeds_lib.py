# -*- coding: utf-8 -*-
"""[review1113 seeds] generalisation of sup3/make_seed_lib.py: isolated copies of srcx/lib for ONE seed-shifted clean-chain run.
usage: python make_seeds_lib.py <teach full|none> <src in|out> <tgt mav|antifs> <shift>      (idempotent; writes only <run dir>/lib)
  run name  RUN  = <tgt>_<teach>_s<shift>                  -> lib + outputs in review1113/seeds/<RUN>/
  combo     C    = <src>2<tgt>[_none]  (as srcx/sx.py)       ; seed tag  TAG = C + '_sd<shift>'
  sx.py          O -> review1113/seeds/<RUN>, W -> D:/S2MUAV3D/srcx_work/<TAG>, REAL_CFG / ST_CFG -> srcx/real_<TAG>.yaml / srcx/st_<TAG>.yaml,
                 PSEUDO_TAG -> srcx_<TAG> (=> index_pseudo_srcx_<TAG>.pkl in the target cache, self-trained detector checkpoints under
                 output/models/uavdet_3d/camnorm/srcx/st_<TAG>/). All names carry the tag: nothing existing is overwritten.
                 Asserts (SRC, TGT, TEACH) == the arguments. SX_NOMKDIR=1 skips the import-time makedirs (dry config print).
  refiner_x / ou_train_x / ori_x / det_infer_x / the rest: exactly the sup3 patches (review1113/sup_patch.py): SX_SEEDSHIFT on every training
                 RNG seed of refiner stages 1-3 and the orientation students, env batch sizes (OOM halving), resume states, sys.path -> copy.
Method constants untouched; data-construction seeds that are not training seeds stay at rng 0 (as sup3)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sup_patch as P  # noqa

teach, src, tgt, shift = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
assert teach in ('full', 'none') and src in ('in', 'out') and tgt in ('mav', 'antifs') and shift > 0, sys.argv
RUN = '%s_%s_s%d' % (tgt, teach, shift)
C = '%s2%s' % (src, tgt) + ('' if teach == 'full' else '_' + teach)
TAG = '%s_sd%d' % (C, shift)
B = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/' + RUN
D = B + '/lib'
WK = 'D:/S2MUAV3D/srcx_work/' + TAG

t = P.rd('sx.py')
t = P.rep(t, P.SX_OW, "O = '%s'          # [seeds] seed-shifted run %s\nW = '%s'\nO_ORIG = HOME + '/' + COMBO; W_ORIG = 'D:/S2MUAV3D/srcx_work/' + COMBO   # read only\n" % (B, RUN, WK))
t = P.rep(t, "REAL_CFG = CFGD + 'srcx/real_%s.yaml' % COMBO", "REAL_CFG = CFGD + 'srcx/real_%%s_sd%d.yaml' %% COMBO" % shift)
t = P.rep(t, "ST_CFG = CFGD + 'srcx/st_%s.yaml' % COMBO", "ST_CFG = CFGD + 'srcx/st_%%s_sd%d.yaml' %% COMBO" % shift)
t = P.rep(t, "PSEUDO_TAG = 'srcx_' + COMBO", "PSEUDO_TAG = 'srcx_' + COMBO + '_sd%d'" % shift)
t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)",
          "assert (SRC, TGT, TEACH) == (%r, %r, %r), (SRC, TGT, TEACH)   # [seeds] this copy is for %s only\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)" % (src, tgt, teach, RUN))
t = P.rep(t, "for d in (O, W):\n    os.makedirs(d, exist_ok=True)", "for d in ((O, W) if os.environ.get('SX_NOMKDIR') != '1' else ()):\n    os.makedirs(d, exist_ok=True)")
P.wr(D, 'sx.py', t, 'outputs -> %s + %s; configs / pseudo tag *_sd%d' % (B, WK, shift))
P.wr(D, 'refiner_x.py', P.refiner(P.path(P.rd('refiner_x.py'), D), seedshift=True), 'SX_SEEDSHIFT on training seeds; env batches; resume states')
P.wr(D, 'ou_train_x.py', P.ou_train(P.path(P.rd('ou_train_x.py'), D), seedshift=True), 'SX_SEEDSHIFT on every RNG seed; resume state; recipe unchanged')
P.wr(D, 'ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'batch = $ORI_BS (default 16 = deployed)')
P.wr(D, 'det_infer_x.py', P.det_bs(P.path(P.rd('det_infer_x.py'), D)), 'batch = $DET_BS (default 8 = deployed)')
for f in ('chain_x.py', 'make_pseudo_x.py', 'mkcfg.py', 'wsim_x.py', 'score_x.py', 'avg_inf.py', 'to_pred.py'):
    P.wr(D, f, P.path(P.rd(f), D), 'sys.path -> redirected sx; code unchanged')
P.infer_wrap(D)
# every copied file must point at this copy, never at srcx/lib (a stray import of the original sx would write into srcx/<combo>)
for f in os.listdir(D):
    if f.endswith('.py'):
        assert 'gproto/srcx/lib' not in open(D + '/' + f, encoding='utf-8').read().split('\n', 1)[1], f
print('ok', RUN, TAG, D)
