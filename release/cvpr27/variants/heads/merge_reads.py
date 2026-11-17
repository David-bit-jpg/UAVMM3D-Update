# -*- coding: utf-8 -*-
"""[review1113 heads H4/H5] build the teacher-reading file of an orientation-teacher ablation from the reference run's readings.
usage: python merge_reads.py <ref read_tr1_teach.pkl> <new reads.pkl | -> <out.pkl> <my ou dir (key prefix)> <spec>
  spec = comma list name=src:member, src = ref (the reference run's reading of teacher T_<member>.pth) or new (this run's reading of
  T_<member>.pth). Output keys are '<my ou dir>/T_<name>.pth' (the paths the run passes to ori_x pseudo).
Unchanged teachers keep the reference run's readings BIT-IDENTICAL (same checkpoint, same AdaBN file, same crops) so that the arm differs from
the reported run only in the replaced members. An alias (two names -> the same ref member) is exact: AdaBN of the same checkpoint on the same
crops gives the same network. Asserts: identical read cells (= identical crops) between ref and new readings."""
import os, sys, pickle
import numpy as np

ref_p, new_p, out_p, odir, spec = sys.argv[1:6]
R = pickle.load(open(ref_p, 'rb'))
N = pickle.load(open(new_p, 'rb')) if new_p != '-' else None


def find(D, member):
    ks = [k for k in D['reads'] if os.path.basename(k.replace('\\', '/')) == 'T_%s.pth' % member]
    assert len(ks) == 1, (member, list(D['reads']))
    return D['reads'][ks[0]]


if N is not None:
    assert len(R['cells']) == len(N['cells']) and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(R['cells'], N['cells'])), 'crops differ'
reads, order = {}, []
for item in spec.split(','):
    name, sm = item.split('='); src, mem = sm.split(':')
    key = '%s/T_%s.pth' % (odir, name)
    reads[key] = find(R if src == 'ref' else N, mem); order.append(key)
    print('%s <- %s reading of T_%s.pth' % (key, src, mem))
pickle.dump(dict(reads=reads, order=order, cells=R['cells']), open(out_p, 'wb'))
print('merged %d teacher readings on %d crops -> %s' % (len(reads), len(R['cells']), out_p))
