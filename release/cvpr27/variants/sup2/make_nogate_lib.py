# -*- coding: utf-8 -*-
"""[sup2] copies of srcx/lib for nogate_commands.sh -> review1113/sup2/nogate/lib (nothing else is written).
Same transformations as review1113/clean3/make_nomir_lib.py (sx.O/W -> $CL_AO, crop arrays read in place from the deployed work dir via
_wf(), env batch sizes ORI_BS / REF_BS / REF_TRAIN_BS with the deployed defaults), plus (sup_patch.py): resume states for the orientation
student and refiner training loops, REF_ST1_BS, and infer_wrap.py (memory fraction 0.2 for inference). No seed shift.
Run: python make_nogate_lib.py   (idempotent)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sup_patch as P  # noqa

D = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2/nogate/lib'
os.makedirs(D, exist_ok=True)

t = P.rd('sx.py')
P.wr(D, 'sx.py', P.rep(t, P.SX_OW, "O = os.environ['CL_AO']\nW = O + '/work'\nW_ORIG = 'D:/S2MUAV3D/srcx_work/' + COMBO\nO_ORIG = HOME + '/' + COMBO\n"),
     'O/W redirected to $CL_AO (writes only under review1113/sup2/nogate); PSEUDO_TAG unchanged (index_pseudo read only, FULL=1)')
P.wr(D, 'ou_train_x.py', P.ou_train(P.path(P.rd('ou_train_x.py'), D)), 'sx = redirected copy; resume state every 250 steps; training recipe unchanged (workers 0)')
t = P.path(P.rd('ori_x.py'), D)
a = "ROOT = 'E:/Open3DUAVDet'; O = sx.O; W = sx.W\n"
t = P.rep(t, a, a + "\n\ndef _wf(p):\n    return p if os.path.exists(p) or not p.startswith(sx.W + '/') else sx.W_ORIG + p[len(sx.W):]\n")
for pat in ["np.load('%s/oc_%s.npy' % (W, name), mmap_mode='r')", "open('%s/oc_%s.pkl' % (W, name), 'rb')", "np.load('%s/oc_%s.npy' % (W, names[0]), mmap_mode='r')",
            "open('%s/oc_%s.pkl' % (W, names[0]), 'rb')", "np.load('%s/oc_%s.npy' % (W, nm_), mmap_mode='r')", "open('%s/oc_%s.pkl' % (W, nm_), 'rb')"]:
    inner = re.search(r"'%s/oc_%s\.(npy|pkl)' % \(W, \w+(\[0\])?\)", pat).group(0)
    t = t.replace(pat, pat.replace(inner, '_wf(%s)' % inner))
P.wr(D, 'ori_x.py', P.ori_bs(t), 'sx = redirected copy; _wf crop fallback to deployed W (read only); batch = $ORI_BS (default 16 = deployed)')
P.wr(D, 'refiner_x.py', P.refiner(P.path(P.rd('refiner_x.py'), D)), 'sx = redirected copy; env batches (deployed defaults); resume states')
P.wr(D, 'chain_x.py', P.path(P.rd('chain_x.py'), D), 'sx = redirected copy (reads/writes under $CL_AO)')
P.wr(D, 'score_x.py', P.rd('score_x.py'), 'unchanged (sx imported from this dir)')
P.infer_wrap(D)
print('ok', D)
