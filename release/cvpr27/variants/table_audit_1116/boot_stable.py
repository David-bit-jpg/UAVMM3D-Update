# Re-run review1113/heads/heads_boot.py unchanged except fast_ap uses a STABLE sort (tie rule of the 11-16 pack scorer).
# usage: same arguments as heads_boot.py
import sys, re
src = open('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/heads_boot.py', encoding='utf-8').read()
n = src.count("o = np.argsort(-sc); t = hit[o]")
assert n == 1, n
src = src.replace("o = np.argsort(-sc); t = hit[o]", "o = np.argsort(-sc, kind='stable'); t = hit[o]")
sys.argv = ['heads_boot.py'] + sys.argv[1:]
g = {'__name__': '__main__', '__file__': 'heads_boot_stable.py'}
exec(compile(src, 'heads_boot_stable.py', 'exec'), g)
