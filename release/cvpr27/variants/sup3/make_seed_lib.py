# -*- coding: utf-8 -*-
"""[sup3] copies of srcx/lib for the second clean-chain run with shifted seeds -> review1113/sup3/lib (nothing else is written).
  sx.py          O -> review1113/sup3/run, W -> D:/S2MUAV3D/srcx_work/in2mav_seed2, REAL_CFG / ST_CFG -> srcx/real_in2mav_seed2.yaml /
                 srcx/st_in2mav_seed2.yaml, PSEUDO_TAG -> srcx_in2mav_seed2 (new files; nothing of in2mav is overwritten). Method constants untouched.
  refiner_x      SX_SEEDSHIFT added to the training seed (torch.manual_seed / numpy rng 1000+seed) of stages 1-3; files keep s5/s6 names;
                 env batches (deployed defaults) + resume states (sup_patch.refiner)
  ou_train_x     SX_SEEDSHIFT added to every RNG seed (set_random_seed, sim-crop dataset, loader generators, target pool bag 1000+seed);
                 init member and run-dir name keep the nominal seed 0/1; resume state every 250 steps (sup_patch.ou_train)
  ori_x          batch = $ORI_BS (16); NO crop fallback to in2mav (crops are re-cut from the new detector)
  det_infer_x    batch = $DET_BS (8)
  chain_x, make_pseudo_x, mkcfg, wsim_x, score_x, avg_inf, to_pred   sys.path -> this copy (redirected sx), otherwise unchanged
Data-construction seeds that are not training seeds stay as in srcx/lib (refiner sim-sample / bank selection rng 0, BN-subset rng 0).
The self-trained detectors' seed is shifted in run_seed2.sh (train.py --seed k+SHIFT). Run: python make_seed_lib.py (idempotent)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sup_patch as P  # noqa

D = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/lib'
OUT = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run'
WK = 'D:/S2MUAV3D/srcx_work/in2mav_seed2'
os.makedirs(D, exist_ok=True)

t = P.rd('sx.py')
t = P.rep(t, P.SX_OW, "O = '%s'          # [sup3] second seed run\nW = '%s'\nO_ORIG = HOME + '/' + COMBO; W_ORIG = 'D:/S2MUAV3D/srcx_work/' + COMBO   # read only\n" % (OUT, WK))
t = P.rep(t, "REAL_CFG = CFGD + 'srcx/real_%s.yaml' % COMBO", "REAL_CFG = CFGD + 'srcx/real_%s_seed2.yaml' % COMBO")
t = P.rep(t, "ST_CFG = CFGD + 'srcx/st_%s.yaml' % COMBO", "ST_CFG = CFGD + 'srcx/st_%s_seed2.yaml' % COMBO")
t = P.rep(t, "PSEUDO_TAG = 'srcx_' + COMBO", "PSEUDO_TAG = 'srcx_' + COMBO + '_seed2'")
t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)", "assert (SRC, TGT, TEACH) == ('in', 'mav', 'full'), (SRC, TGT, TEACH)   # [sup3] this copy is for in2mav only\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)")
P.wr(D, 'sx.py', t, 'outputs -> review1113/sup3/run + D:/S2MUAV3D/srcx_work/in2mav_seed2; configs / pseudo tag *_seed2')
P.wr(D, 'refiner_x.py', P.refiner(P.path(P.rd('refiner_x.py'), D), seedshift=True), 'SX_SEEDSHIFT on training seeds; env batches; resume states')
P.wr(D, 'ou_train_x.py', P.ou_train(P.path(P.rd('ou_train_x.py'), D), seedshift=True), 'SX_SEEDSHIFT on every RNG seed; resume state; recipe unchanged')
P.wr(D, 'ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'batch = $ORI_BS (default 16 = deployed)')
P.wr(D, 'det_infer_x.py', P.det_bs(P.path(P.rd('det_infer_x.py'), D)), 'batch = $DET_BS (default 8 = deployed)')
for f in ('chain_x.py', 'make_pseudo_x.py', 'mkcfg.py', 'wsim_x.py', 'score_x.py', 'avg_inf.py', 'to_pred.py'):
    P.wr(D, f, P.path(P.rd(f), D), 'sys.path -> redirected sx; code unchanged')
P.infer_wrap(D)
print('ok', D)
