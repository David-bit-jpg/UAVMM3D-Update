# -*- coding: utf-8 -*-
"""IR + LiDAR 辅助答案训练入口：注册 CenterDetMM / CenterDetLiD 后原样调用 tools/train.py。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '@@ROOT@@')
sys.path.insert(0, '@@ROOT@@/tools')
sys.path.insert(0, HERE)
import uavdet3d.model.detectors as dets          # noqa: E402
from center_det_mm import CenterDetMM            # noqa: E402
from center_det_lid import CenterDetLiD          # noqa: E402
dets.__all__['CenterDetMM'] = CenterDetMM
dets.__all__['CenterDetLiD'] = CenterDetLiD
import train as T                                # noqa: E402

if __name__ == '__main__':
    sys.argv = ['train.py'] + sys.argv[1:]
    T.main()
