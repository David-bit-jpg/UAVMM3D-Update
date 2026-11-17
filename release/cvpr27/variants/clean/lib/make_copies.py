# [clean] make inference-only copies of srcx/lib (current, post 11-13 glue) whose sx.O / sx.W point under review1113/clean (env CL_AO).
# Changes: sx O/W redirect (+ W_ORIG/O_ORIG read-only fallbacks); ori_x _wf() fallback to deployed crops; batch 16/128 -> 8; det_infer M12_NOFLIP.
import re, os
S = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib'; D = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean/lib'
OLDP = "sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')"; NEWP = "sys.path.insert(0, '%s')" % D
def rd(f): return open(S + '/' + f, encoding='utf-8').read()
def wr(f, t, hdr): open(D + '/' + f, 'w', encoding='utf-8').write('# [clean copy of srcx/lib/%s] %s\n' % (f, hdr) + t)
t = rd('sx.py')
a = "O = HOME + '/' + COMBO                      # per-combo outputs (recs, ref, ou, preds, logs)\nW = 'D:/S2MUAV3D/srcx_work/' + COMBO        # per-combo large crop arrays\n"
assert a in t
t = t.replace(a, "O = os.environ['CL_AO']\nW = O + '/work'\nW_ORIG = 'D:/S2MUAV3D/srcx_work/' + COMBO\nO_ORIG = HOME + '/' + COMBO\n")
wr('sx.py', t, 'O/W redirected to $CL_AO (writes only under review1113/clean); O_ORIG/W_ORIG = deployed (read only)')
t = rd('ori_x.py'); assert OLDP in t; t = t.replace(OLDP, NEWP)
t = t.replace("ROOT = 'E:/Open3DUAVDet'; O = sx.O; W = sx.W\n", "ROOT = 'E:/Open3DUAVDet'; O = sx.O; W = sx.W\n\n\ndef _wf(p):\n    return p if os.path.exists(p) or not p.startswith(sx.W + '/') else sx.W_ORIG + p[len(sx.W):]\n", 1)
for pat in ["np.load('%s/oc_%s.npy' % (W, name), mmap_mode='r')", "open('%s/oc_%s.pkl' % (W, name), 'rb')", "np.load('%s/oc_%s.npy' % (W, names[0]), mmap_mode='r')",
            "open('%s/oc_%s.pkl' % (W, names[0]), 'rb')", "np.load('%s/oc_%s.npy' % (W, nm_), mmap_mode='r')", "open('%s/oc_%s.pkl' % (W, nm_), 'rb')"]:
    inner = re.search(r"'%s/oc_%s\.(npy|pkl)' % \(W, \w+(\[0\])?\)", pat).group(0)
    t = t.replace(pat, pat.replace(inner, '_wf(%s)' % inner))
t = t.replace('def batches(X, idx, bs=16):', 'def batches(X, idx, bs=8):').replace('np.array(el), 16)', 'np.array(el), 8)').replace('np.arange(len(info)), 16)', 'np.arange(len(info)), 8)')
wr('ori_x.py', t, 'sx = clean copy; _wf crop fallback to deployed W; batches 16 -> 8')
t = rd('refiner_x.py'); assert OLDP in t; t = t.replace(OLDP, NEWP).replace('if len(buf) == 128:', 'if len(buf) == 8:')
wr('refiner_x.py', t, 'sx = clean copy; inference batch 128 -> 8')
t = rd('det_infer_x.py'); assert OLDP in t; t = t.replace(OLDP, NEWP)
assert '    EC.install_flip_hook(model)\n' in t
t = t.replace('    EC.install_flip_hook(model)\n', "    if os.environ.get('M12_NOFLIP') != '1':\n        EC.install_flip_hook(model)\n")
wr('det_infer_x.py', t, 'sx = clean copy; M12_NOFLIP=1 skips flip TTA')
t = rd('chain_x.py'); assert OLDP in t; wr('chain_x.py', t.replace(OLDP, NEWP), 'sx = clean copy (reads/writes under $CL_AO)')
t = rd('score_x.py'); wr('score_x.py', t, 'unchanged (sx imported from this dir)')
print('ok')
