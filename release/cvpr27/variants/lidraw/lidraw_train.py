# -*- coding: utf-8 -*-
"""[review1113 lidraw] training entry = geo1004/lidar/kzlid/train_kz_lid.py (the KZ-LiD entry) + two things:
  1. registers CenterDetLiD (copied unchanged, arm a) and CenterDetLidPts (raw-point head, arm b) from this directory;
  2. redirects the output dir to review1113/lidraw/runs/<cfg stem>/<extra_tag> (tools/train.py otherwise derives it from the cfg path,
     which would write into the shared model folder) - via $LIDRAW_EXP (EXP_GROUP_PATH relative to E:/Open3DUAVDet/output).
tools/train.py itself is called unchanged (auto-resume from the newest checkpoint_epoch_*.pth in the run's ckpt dir).
    cd E:/Open3DUAVDet/tools && python <this> --cfg_file <cfg> --seed k --extra_tag ... (COMMON args of run_kzlid_train.sh)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'E:/Open3DUAVDet')
sys.path.insert(0, 'E:/Open3DUAVDet/tools')
sys.path.insert(0, HERE)
import uavdet3d.model.detectors as dets          # noqa: E402
from center_det_lid import CenterDetLiD          # noqa: E402
from center_det_lidpts import CenterDetLidPts    # noqa: E402
dets.__all__['CenterDetLiD'] = CenterDetLiD
dets.__all__['CenterDetLidPts'] = CenterDetLidPts
import train as T                                # noqa: E402

_pc = T.parse_config


def _remap(node):   # 11-17: E: (USB HDD) saturated -> byte-identical copy of the indoor sim cache on the F: SSD
    if isinstance(node, dict):
        for k in list(node.keys()): node[k] = _remap(node[k])
        return node
    if isinstance(node, list): return [_remap(v) for v in node]
    if isinstance(node, str):
        t = node.replace(chr(92), '/')
        if t.startswith('E:/mmcache/full_v2') and os.path.isdir('F:/mmcache/full_v2'): return 'F:/mmcache/full_v2' + t[len('E:/mmcache/full_v2'):]
    return node


def parse_config():
    args, cfg = _pc()
    _remap(cfg)
    cfg.EXP_GROUP_PATH = os.environ.get('LIDRAW_EXP', 'camnorm/geo1004/gproto/review1113/lidraw/runs')
    return args, cfg


T.parse_config = parse_config

if __name__ == '__main__':
    sys.argv = ['train.py'] + sys.argv[1:]
    T.main()
