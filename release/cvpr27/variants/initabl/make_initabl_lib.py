# -*- coding: utf-8 -*-
"""[review1113 initabl] isolated lib copies for the student-initialisation ablation (IR+LiDAR teacher init vs RGB-only init).
usage: python make_initabl_lib.py <s0|s10|s20>      (idempotent; writes only review1113/initabl/<run>/lib)
  s0  = srcx/in2mav                    (reported run; lib = srcx/lib + sup_patch resume/env-batch patches, NO seed shift)
  s10 = review1113/sup3/run            (lib = review1113/sup3/lib as the run used it; SX_SEEDSHIFT 10 on the student RNG seeds)
  s20 = review1113/seeds/mav_full_s20  (lib = review1113/seeds/mav_full_s20/lib; SX_SEEDSHIFT 20)
Only change vs the run's own lib: sx.O -> review1113/initabl/<run>, sys.path -> this copy. sx.W stays = the run's crop dir
(oc_*.npy/pkl, ou_bank.npy, pseudo_tr1.npz) and is used READ ONLY: the ablation never calls ori_x crops/bank/pseudo (the only writers of W);
ou_train_x reads pseudo/crops/bank, ori_x adabn/read and chain_x deploy read crops; their outputs go to explicit paths under O.
Method code untouched (student init comes from $OU_INIT, which ou_train_x already supports)."""
import os, sys
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
sys.path.insert(0, GP + '/review1113')
import sup_patch as P  # noqa

RUNS = {'s0': (None, GP + '/srcx/in2mav', 'D:/S2MUAV3D/srcx_work/in2mav'),
        's10': (GP + '/review1113/sup3/lib', GP + '/review1113/sup3/run', 'D:/S2MUAV3D/srcx_work/in2mav_seed2'),
        's20': (GP + '/review1113/seeds/mav_full_s20/lib', GP + '/review1113/seeds/mav_full_s20', 'D:/S2MUAV3D/srcx_work/in2mav_sd20')}
run = sys.argv[1]
SRCLIB, RUNDIR, WK = RUNS[run]
B = GP + '/review1113/initabl/' + run
D = B + '/lib'
os.makedirs(D, exist_ok=True)
FILES = ('sx.py', 'ou_train_x.py', 'ori_x.py', 'chain_x.py', 'score_x.py')


def wr(f, t, hdr):
    body = t
    assert 'gproto/srcx/lib' not in body, (f, 'still points at srcx/lib')
    if SRCLIB:
        assert SRCLIB not in body, (f, 'still points at the seed-run lib')
    open(D + '/' + f, 'w', encoding='utf-8').write('# [review1113/initabl copy for run %s] %s\n' % (run, hdr) + body)


if SRCLIB is None:   # s0: = the reported run's code (srcx/lib, unchanged since 11-13 04:55 < run start 06:48) + resume / env batches
    t = P.rd('sx.py')
    t = P.rep(t, P.SX_OW, "O = '%s'          # [initabl] student-init ablation, run %s\nW = '%s'          # [initabl] the run's crop dir, READ ONLY\n" % (B, run, WK))
    t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)",
              "assert (SRC, TGT, TEACH) == ('in', 'mav', 'full'), (SRC, TGT, TEACH)   # [initabl] in2mav only\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)")
    wr('sx.py', t, 'O -> %s; W = %s (read only); everything else = srcx/lib/sx.py' % (B, WK))
    wr('ou_train_x.py', P.ou_train(P.path(P.rd('ou_train_x.py'), D)), 'sup_patch: resume state every 250 steps; no seed shift; recipe unchanged')
    wr('ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'sup_patch: batch = $ORI_BS (default 16 = deployed)')
    for f in ('chain_x.py', 'score_x.py'):
        wr(f, P.path(P.rd(f), D), 'sys.path -> this copy; code unchanged')
    P.infer_wrap(D)
else:
    def rd(f):
        return open(SRCLIB + '/' + f, encoding='utf-8').read()
    t = rd('sx.py')
    lines = [l for l in t.split('\n') if l.startswith("O = '")]
    assert len(lines) == 1 and RUNDIR in lines[0], lines
    wl = [l for l in t.split('\n') if l.startswith("W = '")]
    assert len(wl) == 1 and ("'%s'" % WK) in wl[0], wl
    t = t.replace(lines[0], "O = '%s'          # [initabl] student-init ablation, run %s (W below = the run's crop dir, READ ONLY)" % (B, run))
    wr('sx.py', t.replace(SRCLIB, D), 'O -> %s; W = %s (read only); everything else = %s/sx.py' % (B, WK, SRCLIB))
    for f in ('ou_train_x.py', 'ori_x.py', 'chain_x.py', 'score_x.py'):
        wr(f, rd(f).replace(SRCLIB, D), 'sys.path -> this copy; code = %s/%s (SX_SEEDSHIFT, resume, env batches as the run used them)' % (SRCLIB, f))
    open(D + '/infer_wrap.py', 'w', encoding='utf-8').write(rd('infer_wrap.py'))
for f in os.listdir(D):
    if f.endswith('.py'):
        s = open(D + '/' + f, encoding='utf-8').read()
        assert 'gproto/srcx/lib' not in s.split('\n', 1)[1], f
print('ok', run, D)
