# M12 launcher: caps this process's CUDA memory (torch.cuda.set_per_process_memory_fraction) then runs a script unchanged.
# usage: python memcap.py <script.py> [args...]   env MEMFRAC (default 0.3)
import os, sys, runpy
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEMFRAC', '0.3')))
script = os.path.abspath(sys.argv[1]); sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name='__main__')
