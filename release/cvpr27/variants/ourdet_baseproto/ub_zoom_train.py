# -*- coding: utf-8 -*-
"""[review1113 ourdet_baseproto, 11-17] = ourdet_train.py (original tools/train.py main(), run dir on D:, thread loader) + registers the
UBZoom_Det_Dataset (lib/ub_zoom_dataset.py) for the upper-bound-with-view-augmentation row. usage (cwd E:/Open3DUAVDet/tools) as ourdet_train.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ourdet_train as OT  # noqa: E402  (patches train.py; its main() only runs under __main__)
import uavdet3d.datasets as UD  # noqa: E402
import ub_zoom_dataset as UZ  # noqa: E402
UD.__all__[UZ.NAME] = UZ.build
if __name__ == '__main__':
    sys.argv = ['train.py'] + sys.argv[1:]
    OT.T.main()
