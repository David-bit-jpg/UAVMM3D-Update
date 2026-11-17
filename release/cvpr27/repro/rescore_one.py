# Score one or more prediction files on one target with the pack scorer and print per-run and mean metrics (CPU only).
# usage: python rescore_one.py mav|anti pred1.pkl [pred2.pkl ...]       (RealDrone3D ATE printed in % of the distance)
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from rescore_all import score_file, K6
tgt, files = sys.argv[1], sys.argv[2:]
per = [score_file(p, tgt) for p in files]
for p, r in zip(files, per): print(' '.join('%s %.4f' % (k, r[k]) for k in K6), p)
if len(per) > 1:
    u = [r['UDS'] for r in per]
    print('mean ' + ' '.join('%s %.4f' % (k, np.mean([r[k] for r in per])) for k in K6), '| UDS sample sd %.4f' % np.std(u, ddof=1))
