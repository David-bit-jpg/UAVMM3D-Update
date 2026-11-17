# -*- coding: utf-8 -*-
"""[review1113 hparam_1117] isolated lib copies for the hyperparameter-sensitivity cells that reuse an EXISTING PST run
(gamma: deployment only; gate / lambda_mir: orientation students -> deployment, stage-3 refiners and w frozen, = sup2 / initabl protocol).
usage: python make_hp_lib.py <run key> <out dir>      (idempotent; writes only <out dir>/lib)
  run keys: mav_s0  = srcx/in2mav (srcx/lib + sup_patch resume/env batches)  mav_s10 = review1113/sup3/run  mav_s20 = review1113/seeds/mav_full_s20
            anti_s0 = srcx/out2antifs (srcx/lib + sup_patch)                 anti_s10 / anti_s20 = review1113/seeds/antifs_full_s10 / _s20
Changes vs the run's own lib (everything else byte-identical):
  sx.O -> <out dir>; sx.W stays the run's crop dir (READ ONLY: no ori_x crops/bank/pseudo call is ever made with this lib);
  chain_x.py: ALPHA = $HP_GAMMA (default 0.5 = deployed); cmd_deploy reads det1 / stage-3 refiner outputs / height readings from $HP_IN
              (default sx.O) and the student readings from $HP_STU (default sx.O); outputs (preds, deploy json) go to sx.O;
  ou_train_x.py: the pseudo-orientation file is $HP_PSEUDO if set (default sx.W/<--pseudo>.npz, unchanged)."""
import os, sys
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
sys.path.insert(0, GP + '/review1113')
import sup_patch as P  # noqa

RUNS = {'mav_s0': (None, GP + '/srcx/in2mav', 'D:/S2MUAV3D/srcx_work/in2mav', ('in', 'mav')),
        'mav_s10': (GP + '/review1113/sup3/lib', GP + '/review1113/sup3/run', 'D:/S2MUAV3D/srcx_work/in2mav_seed2', ('in', 'mav')),
        'mav_s20': (GP + '/review1113/seeds/mav_full_s20/lib', GP + '/review1113/seeds/mav_full_s20', 'D:/S2MUAV3D/srcx_work/in2mav_sd20', ('in', 'mav')),
        'anti_s0': (None, GP + '/srcx/out2antifs', 'D:/S2MUAV3D/srcx_work/out2antifs', ('out', 'antifs')),
        'anti_s10': (GP + '/review1113/seeds/antifs_full_s10/lib', GP + '/review1113/seeds/antifs_full_s10', 'D:/S2MUAV3D/srcx_work/out2antifs_sd10', ('out', 'antifs')),
        'anti_s20': (GP + '/review1113/seeds/antifs_full_s20/lib', GP + '/review1113/seeds/antifs_full_s20', 'D:/S2MUAV3D/srcx_work/out2antifs_sd20', ('out', 'antifs'))}
run, B = sys.argv[1], sys.argv[2].replace(chr(92), '/').rstrip('/')
SRCLIB, RUNDIR, WK, (src, tgt) = RUNS[run]
assert B.startswith(GP + '/review1113/hparam_1117/'), B
D = B + '/lib'
os.makedirs(D, exist_ok=True)


def wr(f, t, hdr):
    assert 'gproto/srcx/lib' not in t, (f, 'still points at srcx/lib')
    if SRCLIB:
        assert SRCLIB not in t, (f, 'still points at the run lib')
    open(D + '/' + f, 'w', encoding='utf-8').write('# [review1113/hparam_1117 copy, run %s] %s\n' % (run, hdr) + t)


def chain(t):
    t = P.rep(t, "ALPHA = 0.5\n", "ALPHA = float(os.environ.get('HP_GAMMA', '0.5'))   # [hparam] gamma (deployed 0.5)\n"
              "HP_IN = os.environ.get('HP_IN', O); HP_STU = os.environ.get('HP_STU', O)   # [hparam] read-only inputs of cmd_deploy\n")
    t = P.rep(t, "D = pickle.load(open(O + '/recs/det1.pkl', 'rb')); sbar", "D = pickle.load(open(HP_IN + '/recs/det1.pkl', 'rb')); sbar")
    t = P.rep(t, "pickle.load(open(V + '/ref/inf_fin_det1_%s.pkl' % sp, 'rb'))", "pickle.load(open(HP_IN + '/ref/inf_fin_det1_%s.pkl' % sp, 'rb'))")
    t = P.rep(t, "pickle.load(open(O + '/kps_%s.pkl' % sp, 'rb'))", "pickle.load(open(HP_IN + '/kps_%s.pkl' % sp, 'rb'))")
    t = P.rep(t, "zL_table('val1', V + '/ou/read_val1_stu.pkl'), 'test': zL_table('te1', V + '/ou/read_te1_stu.pkl')",
              "zL_table('val1', HP_STU + '/ou/read_val1_stu.pkl'), 'test': zL_table('te1', HP_STU + '/ou/read_te1_stu.pkl')")
    return t


def oupseudo(t):
    return P.rep(t, "P = dict(np.load(sx.W + '/%s.npz' % a.pseudo, allow_pickle=False))",
                 "P = dict(np.load(os.environ.get('HP_PSEUDO') or (sx.W + '/%s.npz' % a.pseudo), allow_pickle=False))   # [hparam] gate variants")


if SRCLIB is None:   # s0 = reported run's code (srcx/lib) + sup_patch resume / env batches, no seed shift
    t = P.rd('sx.py')
    t = P.rep(t, P.SX_OW, "O = '%s'          # [hparam] run %s\nW = '%s'          # [hparam] the run's crop dir, READ ONLY\n" % (B, run, WK))
    t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)",
              "assert (SRC, TGT, TEACH) == (%r, %r, 'full'), (SRC, TGT, TEACH)   # [hparam]\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)" % (src, tgt))
    wr('sx.py', t, 'O -> %s; W = %s (read only)' % (B, WK))
    wr('ou_train_x.py', oupseudo(P.ou_train(P.path(P.rd('ou_train_x.py'), D))), 'sup_patch resume; $HP_PSEUDO; recipe unchanged')
    wr('ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'sup_patch: batch = $ORI_BS (default 16)')
    wr('chain_x.py', chain(P.path(P.rd('chain_x.py'), D)), '$HP_GAMMA / $HP_IN / $HP_STU')
    wr('score_x.py', P.path(P.rd('score_x.py'), D), 'unchanged')
    P.infer_wrap(D)
else:
    def rd(f):
        return open(SRCLIB + '/' + f, encoding='utf-8').read()
    t = rd('sx.py')
    ol = [l for l in t.split('\n') if l.startswith("O = '")]; assert len(ol) == 1 and RUNDIR in ol[0], ol
    wl = [l for l in t.split('\n') if l.startswith("W = '")]; assert len(wl) == 1 and ("'%s'" % WK) in wl[0], wl
    t = t.replace(ol[0], "O = '%s'          # [hparam] run %s (W below = the run's crop dir, READ ONLY)" % (B, run))
    wr('sx.py', t.replace(SRCLIB, D), 'O -> %s; W = %s (read only)' % (B, WK))
    wr('ou_train_x.py', oupseudo(rd('ou_train_x.py').replace(SRCLIB, D)), '$HP_PSEUDO; code = run lib (SX_SEEDSHIFT, resume)')
    wr('ori_x.py', rd('ori_x.py').replace(SRCLIB, D), 'code = run lib')
    wr('chain_x.py', chain(rd('chain_x.py').replace(SRCLIB, D)), '$HP_GAMMA / $HP_IN / $HP_STU')
    wr('score_x.py', rd('score_x.py').replace(SRCLIB, D), 'code = run lib')
    open(D + '/infer_wrap.py', 'w', encoding='utf-8').write(rd('infer_wrap.py'))
print('ok', run, D, 'base', RUNDIR)
