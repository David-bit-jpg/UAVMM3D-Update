# -*- coding: utf-8 -*-
"""N1：在 MAV6D 验证集上重跑第一轮教师检测器（ST2 五集成 + 翻转，无时序背景，与 real_train_preds_st2nb.pkl 同口径）。
不改原脚本：限显存 0.15、把 build_dataloader 的 batch 压到 4，然后原样执行 tools/annot_audit/dump_real_preds.py。"""
import os, sys, runpy
import torch
torch.cuda.set_per_process_memory_fraction(0.15, 0)
sys.path.insert(0, 'E:/Open3DUAVDet'); sys.path.insert(0, 'E:/Open3DUAVDet/tools')
import uavdet3d.datasets as UDS
_b = UDS.build_dataloader
def _bd(*a, **k):
    if 'batch_size' in k: k['batch_size'] = min(k['batch_size'], 4)
    return _b(*a, **k)
UDS.build_dataloader = _bd
M = 'E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/st2_mav6d_r34_geo_full_tr'
OUT = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1/real_val_preds_st2nb.pkl'
sys.argv = ['dump_real_preds.py', '--split', 'val', '--bg', 'none', '--ckpt', M + '/ST2_s0/ckpt/checkpoint_epoch_3.pth',
            '--ensemble'] + [M + '/ST2_s%d/ckpt/checkpoint_epoch_3.pth' % k for k in range(1, 5)] + ['--out', OUT]
runpy.run_path('E:/Open3DUAVDet/tools/annot_audit/dump_real_preds.py', run_name='__main__')
