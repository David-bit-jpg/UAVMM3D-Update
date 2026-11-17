# -*- coding: utf-8 -*-
"""最终对比用：在 night1003 的 dump_members_repl.py 上加一个位置集合 st2p（10-11 去偏自训练检测器 ST2P_s0..4），
记录 KZ（发布版）或 AdaBN KZ 五个成员的关键点窗口，供离线 SA-TM 融合。用法：python dump_members_st2p.py <test|val> <kz|adabn>"""
import sys
REPL = '@@ROOT@@/output/camnorm/night1003/scratch/repl'
sys.path.insert(0, REPL)
import dump_members_repl as M   # noqa: E402

M.POS_SETS['st2p'] = ['@@ROOT@@/output/models/uavdet_3d/camnorm/st2plus_mav6d_r34_geo_full_tr/ST2P_s%d/ckpt/checkpoint_epoch_3.pth' % i for i in range(5)]
split, mem = sys.argv[1], sys.argv[2]
sys.exit(M.main(['--split', split, '--pos', 'st2p', '--members', mem, '--tag', 'st2p_%s_%s' % (split, mem)]))
