# -*- coding: utf-8 -*-
"""wf23 第 2 条：全严格位置检测器 wsfnb（3 集成 + 翻转，单帧）在真实 train 全部帧（间隔 1）上的检测 + 朝向钩子 zoom-95 裁窗窗口。
只推理、不读标签。用途：在每个真实训练帧上算 LiDAR 教的角点测距 z_kp，用来和红外教的伪距离合并成第二段的新伪目标。
复制自 wf21/strict/dump_train_wsfnb_kz.py，只改输出目录与标签名。用法（cwd = @@ROOT@@/tools）：python dump_train1.py"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '@@ROOT@@/output/camnorm/night1003/scratch/repl')
import dump_members_repl as M   # noqa: E402
RC = M.RC
CK = ['@@ROOT@@/output/models/uavdet_3d/camnorm/st2sfnbwa_mav6d_r34_geo_full_tr/WSFNB_s%d/ckpt/checkpoint_epoch_3.pth' % k for k in range(3)]
for c in CK:
    assert os.path.exists(c), c
M.POS_SETS['wsfnb'] = CK
_ba = M.build_argv
M.build_argv = lambda *a, **k: _ba(*a, **k) + ['--bg', 'none']
OUT = HERE + '/tags'
RC.under_out_root = lambda p: os.path.normcase(os.path.abspath(p)).startswith(os.path.normcase(os.path.abspath(OUT)))
M.main(['--split', 'train', '--pos', 'wsfnb', '--members', 'kz', '--tag', 'wsfnb_train1_kz', '--interval', '1', '--out-root', OUT])
