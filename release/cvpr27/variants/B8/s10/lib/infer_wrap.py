# [review1113/B8 copy of E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/lib/infer_wrap.py] sys.path -> this copy; code = the seed run lib (refiner: SX_SEEDSHIFT on the training seed, env batches, resume)
# -*- coding: utf-8 -*-
"""[review1113 sup] run an inference script (argv[1], its own argv after it) as __main__ under a CUDA memory cap:
torch.cuda.set_per_process_memory_fraction($SX_INFER_MEMFRAC, default 0.2). No other change."""
import os, runpy, sys
import torch
fr = float(os.environ.get('SX_INFER_MEMFRAC', '0.2')); assert 0 < fr <= 0.2, fr
torch.cuda.set_per_process_memory_fraction(fr)
script = os.path.abspath(sys.argv[1]); sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
print('[infer_wrap] memory fraction %.2f -> %s' % (fr, ' '.join(sys.argv)), flush=True)
runpy.run_path(script, run_name='__main__')
