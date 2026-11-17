# -*- coding: utf-8 -*-
"""[review1113/cleanfix B3/B4/B6] = review1113/clean3/make_nomir_lib.py with the lib dir taken from argv[1]; transformations identical (nothing else is written).
Same transformations as review1113/clean/lib/make_copies.py, except that the inference batch sizes are env-controlled (for OOM halving)
and keep the deployed defaults (ori_x 16, refiner 128):
  sx.py        O/W -> $CL_AO, $CL_AO/work (+ O_ORIG / W_ORIG = deployed, read only)
  ou_train_x   imports the redirected sx (default runs dir / pseudo file under $CL_AO); training code unchanged (DataLoader workers are 0 already)
  ori_x        redirected sx; _wf(): crop arrays fall back to the deployed work dir (read only, not copied: 5.7 + 2.1 + 1.0 GB); batch = $ORI_BS (16)
  refiner_x    redirected sx; inference batch = $REF_BS (128)
  chain_x / score_x   redirected sx
Run: python make_cf_lib.py <lib dir>   (idempotent)."""
import os, re
S = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib'
import sys
D = sys.argv[1].replace(chr(92), '/')   # [cleanfix] target lib dir (B3 / B4 / B6)
os.makedirs(D, exist_ok=True)
OLDP = "sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')"; NEWP = "sys.path.insert(0, '%s')" % D


def rd(f):
    return open(S + '/' + f, encoding='utf-8').read()


def wr(f, t, hdr):
    open(D + '/' + f, 'w', encoding='utf-8').write('# [review1113/cleanfix copy of srcx/lib/%s] %s\n' % (f, hdr) + t)


t = rd('sx.py')
a = "O = HOME + '/' + COMBO                      # per-combo outputs (recs, ref, ou, preds, logs)\nW = 'D:/S2MUAV3D/srcx_work/' + COMBO        # per-combo large crop arrays\n"
assert a in t
wr('sx.py', t.replace(a, "O = os.environ['CL_AO']\nW = O + '/work'\nW_ORIG = 'D:/S2MUAV3D/srcx_work/' + COMBO\nO_ORIG = HOME + '/' + COMBO\n"),
   'O/W redirected to $CL_AO (writes only under review1113/cleanfix)')
t = rd('ou_train_x.py'); assert t.count(OLDP) == 1
wr('ou_train_x.py', t.replace(OLDP, NEWP), 'sx = redirected copy; training code unchanged')
t = rd('ori_x.py'); assert OLDP in t; t = t.replace(OLDP, NEWP)
a = "ROOT = 'E:/Open3DUAVDet'; O = sx.O; W = sx.W\n"; assert a in t
t = t.replace(a, a + "\n\ndef _wf(p):\n    return p if os.path.exists(p) or not p.startswith(sx.W + '/') else sx.W_ORIG + p[len(sx.W):]\n", 1)
for pat in ["np.load('%s/oc_%s.npy' % (W, name), mmap_mode='r')", "open('%s/oc_%s.pkl' % (W, name), 'rb')", "np.load('%s/oc_%s.npy' % (W, names[0]), mmap_mode='r')",
            "open('%s/oc_%s.pkl' % (W, names[0]), 'rb')", "np.load('%s/oc_%s.npy' % (W, nm_), mmap_mode='r')", "open('%s/oc_%s.pkl' % (W, nm_), 'rb')"]:
    inner = re.search(r"'%s/oc_%s\.(npy|pkl)' % \(W, \w+(\[0\])?\)", pat).group(0)
    t = t.replace(pat, pat.replace(inner, '_wf(%s)' % inner))
BS = "int(os.environ.get('ORI_BS', '16'))"
for x, y in (('def batches(X, idx, bs=16):', 'def batches(X, idx, bs=None):\n    bs = bs or %s' % BS), ('np.array(el), 16)', 'np.array(el), %s)' % BS),
             ('np.arange(len(info)), 16)', 'np.arange(len(info)), %s)' % BS)):
    assert x in t, x
    t = t.replace(x, y)
wr('ori_x.py', t, 'sx = redirected copy; _wf crop fallback to deployed W (read only); batch = $ORI_BS (default 16 = deployed)')
t = rd('refiner_x.py'); assert OLDP in t and 'if len(buf) == 128:' in t
TB = "int(os.environ.get('REF_TRAIN_BS', '64'))"
for x in ('rng.choice(len(S), 64, replace=False)', 'ri = rng.choice(pool, 64, replace=len(pool) < 64)'):
    assert t.count(x) == 1, x
t = t.replace('rng.choice(len(S), 64, replace=False)', 'rng.choice(len(S), %s, replace=False)' % TB)
t = t.replace('ri = rng.choice(pool, 64, replace=len(pool) < 64)', 'ri = rng.choice(pool, %s, replace=len(pool) < %s)' % (TB, TB))
wr('refiner_x.py', t.replace(OLDP, NEWP).replace('if len(buf) == 128:', "if len(buf) == int(os.environ.get('REF_BS', '128')):"),
   'sx = redirected copy; inference batch = $REF_BS (default 128 = deployed); stage-2/3 train batch = $REF_TRAIN_BS sim + real (default 64 = deployed)')
t = rd('chain_x.py'); assert OLDP in t; wr('chain_x.py', t.replace(OLDP, NEWP), 'sx = redirected copy (reads/writes under $CL_AO)')
wr('score_x.py', rd('score_x.py'), 'unchanged (sx imported from this dir)')
# [cleanfix] + resume state every 250 steps for the students (= review1113/sup_patch.ou_train, as used by initabl; recipe unchanged, no seed shift)
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113')
import sup_patch as SP  # noqa
t = open(D + '/ou_train_x.py', encoding='utf-8').read()
if 'RESUME from step' not in t:
    open(D + '/ou_train_x.py', 'w', encoding='utf-8').write(SP.ou_train(t))
SP.infer_wrap(D)   # inference under a CUDA memory cap (SX_INFER_MEMFRAC, 0.2), as initabl
print('ok', D)
