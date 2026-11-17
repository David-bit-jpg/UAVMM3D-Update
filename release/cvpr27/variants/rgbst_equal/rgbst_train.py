# -*- coding: utf-8 -*-
"""[review1113 rgbst_equal] thin wrapper = review1113/ourdet_baseproto/ourdet_train.py run unchanged (runpy), under a separate file name so that
gpu_slot.sh does not classify this supplementary job as an urgent main-table job (its command line must not contain 'ourdet_baseproto')."""
import runpy, sys
runpy.run_path('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ourdet_train.py', run_name='__main__')
