# -*- coding: utf-8 -*-
"""全严格位置检测器 wsfnb（WSFNB_s0..2 三集成 + 翻转，推理 --bg none = 单帧）在真实 train 上的检测 + 朝向钩子 zoom-95 裁窗窗口（间隔 3）。
只推理、封存真值、不读标签。用途：在无标签训练集上算"关键点高度比按颜色原型的中位"（替代原来取自另一划分的常数）。
改自 wf20/repro/dump_train_wanb.py（只换检测器权重与输出目录）。用法（cwd = @@ROOT@@/tools）：python dump_train_wsfnb_kz.py [interval]"""
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
iv = sys.argv[1] if len(sys.argv) > 1 else '3'
M.main(['--split', 'train', '--pos', 'wsfnb', '--members', 'kz', '--tag', 'wsfnb_train_kz', '--interval', iv, '--out-root', OUT])
