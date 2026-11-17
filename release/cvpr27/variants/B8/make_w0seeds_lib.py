# -*- coding: utf-8 -*-
"""[review1113 B8] IR-only fusion control (omega = 0 stage-3 retrain, = clean2/cost/w0_commands.sh) on the PST seed runs 2 and 3.
usage: python make_w0seeds_lib.py <s10|s20>      (idempotent; writes only review1113/B8/<run>/lib)
  s10 = review1113/sup3/run            (SX_SEEDSHIFT 10, lib review1113/sup3/lib,           crops D:/S2MUAV3D/srcx_work/in2mav_seed2)
  s20 = review1113/seeds/mav_full_s20  (SX_SEEDSHIFT 20, lib review1113/seeds/mav_full_s20/lib, crops D:/S2MUAV3D/srcx_work/in2mav_sd20)
The copies start from THAT seed run's own lib (so PSEUDO_TAG = the run's Module-3 index *_seed2 / *_sd20, refiner seeds = nominal 5/6 + SX_SEEDSHIFT,
env batch sizes, resume states, exactly as the run used them). Only change: sx.O -> $CL_AO, sx.W -> $CL_AO/work (W_SEED = the run's crop dir, read
only). chain_x / ori_x / refiner_x / score_x / avg_inf / infer_wrap point at this copy. Method code untouched."""
import os, re, sys

GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
RUNS = {'s10': (GP + '/review1113/sup3/lib', 'D:/S2MUAV3D/srcx_work/in2mav_seed2'),
        's20': (GP + '/review1113/seeds/mav_full_s20/lib', 'D:/S2MUAV3D/srcx_work/in2mav_sd20')}
run = sys.argv[1]
SRC, WSEED = RUNS[run]
D = GP + '/review1113/B8/%s/lib' % run
os.makedirs(D, exist_ok=True)


def rd(f):
    return open(SRC + '/' + f, encoding='utf-8').read()


def wr(f, t, hdr):
    assert SRC not in t, (f, 'still points at the seed-run lib')
    assert 'gproto/srcx/lib' not in t.split('\n', 1)[1], (f, 'points at srcx/lib')
    open(D + '/' + f, 'w', encoding='utf-8').write('# [review1113/B8 copy of %s/%s] %s\n' % (SRC, f, hdr) + t)


t = rd('sx.py')
mo = re.findall(r"^O = '[^']*'.*$", t, flags=re.M); mw = re.findall(r"^W = '[^']*'.*$", t, flags=re.M)
assert len(mo) == 1 and len(mw) == 1, (mo, mw)
assert ("'%s'" % WSEED) in mw[0], (mw, WSEED)
t = t.replace(mo[0], "O = os.environ['CL_AO']                        # [B8] outputs only under review1113/B8/%s" % run)
t = t.replace(mw[0], "W = O + '/work'\nW_SEED = '%s'   # [B8] seed run crop dir (read only)" % WSEED)
assert t.count("PSEUDO_TAG = 'srcx_' + COMBO + '_") == 1   # keeps the seed run's own Module-3 pseudo-label index
wr('sx.py', t.replace(SRC, D), 'O/W -> $CL_AO (writes only under review1113/B8/%s); PSEUDO_TAG / configs = the seed run' % run)
for f in ('refiner_x.py', 'chain_x.py', 'ori_x.py', 'score_x.py', 'avg_inf.py', 'infer_wrap.py'):
    s = rd(f)
    wr(f, s.replace(SRC, D), 'sys.path -> this copy; code = the seed run lib (refiner: SX_SEEDSHIFT on the training seed, env batches, resume)')
print('ok', run, D)
