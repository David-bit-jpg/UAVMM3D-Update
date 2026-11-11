# -*- coding: utf-8 -*-
"""朝向五成员（kzt7q，已训好不动）在原型组检测器的检测框上推理。python dump_orient_p.py <val|test> <pf|pl|pfl>"""
import os, sys
G = '@@ROOT@@/output/camnorm/geo1004'
sp, NAME = sys.argv[1], sys.argv[2]
sys.path.insert(0, G + '/proto')
import run_proto as RP  # noqa: E402
CK = [RP.ck(NAME, k) for k in range(RP.NSEED.get(NAME, 5))]
sys.path.insert(0, '@@ROOT@@/output/camnorm/night1003/scratch/repl')
import dump_members_repl as M  # noqa: E402
M.POS_SETS[NAME] = CK
if NAME in RP.NOTBG:                                   # 11-07 夜修：无背景组的朝向推理也必须用无热图时序背景的检测（否则位置与集成记录对不上，F.apply 断言失败）
    _ba = M.build_argv
    M.build_argv = lambda *a, **k: _ba(*a, **k) + ['--bg', 'none']
LOO = [G + '/kzst7/train_q/LOO_s%d/ckpt/checkpoint_epoch_3.pth' % k for k in range(5)]
for c in CK + LOO:
    assert os.path.exists(c), c
M.main(['--split', sp, '--pos', NAME, '--ckpts'] + LOO + ['--tag', '%s_%s_kzt7q' % (NAME, sp)])
