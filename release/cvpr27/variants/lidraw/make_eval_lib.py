# -*- coding: utf-8 -*-
"""[review1113 lidraw] isolated copies of the clean chain's orientation reading code for the teacher-level evaluation (idempotent).
Writes ONLY review1113/lidraw/eval/lib: sx.py (O -> review1113/lidraw/eval; W = the reported in2mav run's crop dir D:/S2MUAV3D/srcx_work/in2mav,
READ ONLY: oc_tr1 (unlabeled real train crops, for the teachers' BN re-estimation) and oc_te1 (test crops)), ori_x.py (sys.path -> this copy;
batch = $ORI_BS, default 16 = deployed) and infer_wrap.py (sup_patch). Code otherwise = srcx/lib (the code of the reported run srcx/in2mav,
as review1113/initabl/make_initabl_lib.py s0). Only ori_x adabn / read and ori_x.ensemble / sym are used; crops / bank / pseudo (the writers of W)
are never called."""
import os, sys
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
sys.path.insert(0, GP + '/review1113')
import sup_patch as P  # noqa
B = GP + '/review1113/lidraw/eval'
D = B + '/lib'
WK = 'D:/S2MUAV3D/srcx_work/in2mav'
os.makedirs(D, exist_ok=True)


def wr(f, t, hdr):
    assert 'gproto/srcx/lib' not in t, (f, 'still points at srcx/lib')
    open(D + '/' + f, 'w', encoding='utf-8').write('# [review1113/lidraw eval copy] %s\n' % hdr + t)


t = P.rd('sx.py')
t = P.rep(t, P.SX_OW, "O = '%s'          # [lidraw] teacher-level orientation eval\nW = '%s'          # [lidraw] the reported in2mav run's crop dir, READ ONLY\n" % (B, WK))
t = P.rep(t, "S = SOURCES.get(SRC); T = TARGETS.get(TGT)",
          "assert (SRC, TGT, TEACH) == ('in', 'mav', 'full'), (SRC, TGT, TEACH)   # [lidraw] in2mav only\nS = SOURCES.get(SRC); T = TARGETS.get(TGT)")
wr('sx.py', t, 'O -> %s; W = %s (read only); everything else = srcx/lib/sx.py' % (B, WK))
wr('ori_x.py', P.ori_bs(P.path(P.rd('ori_x.py'), D)), 'sup_patch: sys.path -> this copy; batch = $ORI_BS (default 16 = deployed); code otherwise = srcx/lib/ori_x.py')
P.infer_wrap(D)
print('[lidraw] eval lib written to', D)
