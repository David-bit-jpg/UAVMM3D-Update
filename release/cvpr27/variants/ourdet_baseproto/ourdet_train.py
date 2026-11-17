# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto] training entry = the ORIGINAL tools/train.py main(), unchanged, with two additions:
  1. the run directory goes to $OD_EXPROOT (default D:/S2MUAV3D/ourdet_baseproto_models) instead of E:/Open3DUAVDet/output (E: saturated);
     same mechanism as review1113/stage1seeds/s1_train.py (absolute prefix on EXP_GROUP_PATH);
  2. dataset 'OurdetConcat' (lib/ourdet_concat.py: sim + real-pseudo concatenation) is registered into uavdet3d.datasets.__all__.
usage (cwd E:/Open3DUAVDet/tools):  python ourdet_train.py <train.py args...>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, 'E:/Open3DUAVDet/tools'); sys.path.insert(0, os.path.join(HERE, 'lib'))
ROOT = os.environ.get('OD_EXPROOT', 'D:/S2MUAV3D/ourdet_baseproto_models').replace('\\', '/').rstrip('/')
import uavdet3d.datasets as UD  # noqa: E402
import ourdet_concat as OC  # noqa: E402
UD.__all__[OC.NAME] = OC.build
import train as T  # noqa: E402
_pc = T.parse_config


def parse_config():
    a, cfg = _pc()
    cfg.EXP_GROUP_PATH = ROOT + '/' + cfg.EXP_GROUP_PATH
    print('[ourdet_train] run dir -> %s/%s/%s' % (cfg.EXP_GROUP_PATH, cfg.TAG, a.extra_tag), flush=True)
    return a, cfg


T.parse_config = parse_config
# 11-16: training batches via a thread pool (lib/thread_loader.py; OD_THREADS threads, 0 = plain DataLoader) instead of spawn workers
# (light slot; --workers 0 alone was CPU-bound at ~1 s/it). Same sampling (seeded permutation per epoch, no drop_last), same collate.
import thread_loader as TL  # noqa: E402
_bd = T.build_dataloader


def build_dataloader(dataset_cfg, batch_size, dist, root_path=None, workers=4, seed=None, logger=None, training=True):
    ds, dl, sp = _bd(dataset_cfg=dataset_cfg, batch_size=batch_size, dist=dist, root_path=root_path, workers=workers, seed=seed,
                     logger=logger, training=training)
    nt = int(os.environ.get('OD_THREADS', '6'))
    if training and nt > 0 and workers == 0:
        dl = TL.ThreadLoader(ds, batch_size, shuffle=True, drop_last=False, threads=nt, seed=seed or 0)
        print('[ourdet_train] thread loader: %d threads, %d batches/epoch' % (nt, len(dl)), flush=True)
    return ds, dl, sp


T.build_dataloader = build_dataloader
if __name__ == '__main__':
    sys.argv = ['train.py'] + sys.argv[1:]
    T.main()
