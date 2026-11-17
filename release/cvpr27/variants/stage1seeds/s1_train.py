# -*- coding: utf-8 -*-
"""[review1113 stage1seeds] training entry = the ORIGINAL trainer script, unchanged, with one redirect: the run directory goes to
$S1_EXPROOT (default D:/S2MUAV3D/stage1seeds_models) instead of E:/Open3DUAVDet/output (E: has < 60 GB free; ~60 GB of stage-1 checkpoints
+ ~12 GB of self-trained detectors would not fit). Same mechanism as review1113/lidraw/lidraw_train.py ($LIDRAW_EXP): tools/train.py builds
output_dir = ROOT_DIR / 'output' / EXP_GROUP_PATH / TAG / extra_tag; EXP_GROUP_PATH is prefixed with the absolute $S1_EXPROOT, so the run dir is
<S1_EXPROOT>/models/uavdet_3d/camnorm/<cfg stem>/<extra_tag> (pathlib: an absolute component replaces the left part). Nothing else changes:
tools/train.py main(), auto-resume from the newest checkpoint_epoch_*.pth in that dir, all arguments as given.
usage (from E:/Open3DUAVDet/tools):  python s1_train.py <trainer: tools/train.py | lidar/kzlid/train_mm.py | train_kz_lid.py | lidar/kpsize/train_mmk.py> <train.py args...>"""
import os, sys, runpy
sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, 'E:/Open3DUAVDet/tools')
ROOT = os.environ.get('S1_EXPROOT', 'D:/S2MUAV3D/stage1seeds_models').replace('\\', '/').rstrip('/')
assert os.path.isabs(ROOT), ROOT
import train as T  # noqa: E402  (tools/train.py; the trainer scripts import the same cached module)
_pc = T.parse_config


REMAP = [('E:/mmcache/full_v2', 'F:/mmcache/full_v2')]   # 11-16: E: (USB HDD) saturated -> byte-identical copy on the F: SSD


def _same_tree(a, b):   # use the copy only if every file exists with the same size (copy finished)
    for dp, _, fs in os.walk(a):
        for f in fs:
            p = os.path.join(dp, f); q = os.path.join(b, os.path.relpath(p, a))
            if not os.path.isfile(q) or os.path.getsize(q) != os.path.getsize(p):
                return False
    return True


def _remap(node, pairs):
    if isinstance(node, dict):
        for k in list(node.keys()):
            node[k] = _remap(node[k], pairs)
        return node
    if isinstance(node, list):
        return [_remap(v, pairs) for v in node]
    if isinstance(node, str):
        s = node.replace('\\', '/')
        for src, dst in pairs:
            if s.startswith(src):
                return dst + s[len(src):]
    return node


def parse_config():
    a, cfg = _pc()
    pairs = [(s, d) for s, d in REMAP if os.environ.get('S1_NO_REMAP') != '1' and os.path.isdir(d) and _same_tree(s, d)]
    if pairs:
        _remap(cfg, pairs); print('[s1_train] data path remap (identical copy): %s' % pairs, flush=True)
    cfg.EXP_GROUP_PATH = ROOT + '/' + cfg.EXP_GROUP_PATH
    print('[s1_train] run dir -> %s/%s/%s' % (cfg.EXP_GROUP_PATH, cfg.TAG, a.extra_tag), flush=True)
    return a, cfg


T.parse_config = parse_config
# 11-16 fix: Windows spawn re-imports this file in every DataLoader worker (as __mp_main__) with the trainer's argv, so everything that
# reads argv / starts training must run only in the main process (same pattern as lidraw/lidraw_train.py).
if __name__ == '__main__':
  trainer = os.path.abspath(sys.argv[1]); args = sys.argv[2:]
  assert os.path.exists(trainer), trainer
  sys.argv = ['train.py'] + args
  if os.path.basename(trainer) == 'train.py':
    T.main()
  else:   # train_mm.py / train_kz_lid.py / train_mmk.py: register their detector classes, then T.main() on the patched module
    sys.path.insert(0, os.path.dirname(trainer))
    runpy.run_path(trainer, run_name='__main__')
