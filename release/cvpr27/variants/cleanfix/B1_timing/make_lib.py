# -*- coding: utf-8 -*-
"""[cleanfix B1] isolated copies of srcx/lib (sx, refiner_x, ori_x, det_infer_x) into B1_timing/lib. Only changes: sys.path -> this copy,
sx.O -> B1_timing/out (no file of the run dir is written; W = the run's crop dir, not used by timing). Method code untouched.
Run with SX_SRC=in SX_TGT=mav SX_TEACH=full (idempotent)."""
import os
S = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib'
B = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B1_timing'
D = B + '/lib'
os.makedirs(D, exist_ok=True)
OLDP = "sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')"; NEWP = "sys.path.insert(0, '%s')" % D
for f in ('sx.py', 'refiner_x.py', 'ori_x.py', 'det_infer_x.py'):
    t = open(S + '/' + f, encoding='utf-8').read()
    if f == 'sx.py':
        a = "O = HOME + '/' + COMBO                      # per-combo outputs (recs, ref, ou, preds, logs)\n"
        assert a in t
        t = t.replace(a, "O = '%s/out'   # [cleanfix B1] redirected (timing writes nothing else)\nO_RUN = HOME + '/' + COMBO\n" % B)
    else:
        assert OLDP in t, f
        t = t.replace(OLDP, NEWP)
    assert 'gproto/srcx/lib' not in t, f
    open(D + '/' + f, 'w', encoding='utf-8').write('# [cleanfix/B1_timing copy of srcx/lib/%s] sys.path/O redirected only\n' % f + t)
print('ok', D)
