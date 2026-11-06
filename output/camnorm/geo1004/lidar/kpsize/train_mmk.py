# -*- coding: utf-8 -*-
"""KZ-KPS 训练入口：注册 CenterDetMMK（+ CenterDetMM）后原样调用 tools/train.py。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '@@ROOT@@')
sys.path.insert(0, '@@ROOT@@/tools')
sys.path.insert(0, '@@ROOT@@/output/camnorm/geo1004/lidar/kzlid')
sys.path.insert(0, '@@ROOT@@/output/camnorm/geo1004/lidar/kzbody')
sys.path.insert(0, HERE)
import uavdet3d.model.detectors as dets          # noqa: E402
from center_det_mm import CenterDetMM            # noqa: E402
from center_det_mmk import CenterDetMMK          # noqa: E402
dets.__all__['CenterDetMM'] = CenterDetMM
dets.__all__['CenterDetMMK'] = CenterDetMMK
import train as T                                # noqa: E402

if __name__ == '__main__':
    sys.argv = ['train.py'] + sys.argv[1:]
    T.main()
