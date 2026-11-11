# -*- coding: utf-8 -*-
"""wf21/strict/dump_train_wsfnb_kz.py / wf23/fusetrain/dump_train1.py 的无标签链副本（11-10）：只把检测器换成 <arm>（3 集成 + 翻转，--bg none 单帧），
输出到 nolabel/<arm>/tags/<tag>。dump_members_repl 推理后把真值封存进 sealed_gt_*.pkl（本链任何脚本都不读它），记录 pkl 只含无标签字段。
用法（cwd = @@ROOT@@/tools）：python dump_train_kz_nl.py <arm> <interval> <tag>"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, '@@ROOT@@/output/camnorm/night1003/scratch/repl')
sys.path.insert(0, '@@ROOT@@/output/camnorm/geo1004/proto')
import run_proto as RP   # noqa: E402
import dump_members_repl as M   # noqa: E402
RC = M.RC
arm, iv, tag = sys.argv[1], sys.argv[2], sys.argv[3]
CK = [RP.ck(arm, k) for k in range(RP.NSEED[arm])]
for c in CK:
    assert os.path.exists(c), c
M.POS_SETS[arm] = CK
_ba = M.build_argv
M.build_argv = lambda *a, **k: _ba(*a, **k) + ['--bg', 'none']
OUT = '@@ROOT@@/output/camnorm/geo1004/gproto/nolabel/%s/tags' % arm
os.makedirs(OUT, exist_ok=True)
RC.under_out_root = lambda p: os.path.normcase(os.path.abspath(p)).startswith(os.path.normcase(os.path.abspath(OUT)))
M.main(['--split', 'train', '--pos', arm, '--members', 'kz', '--tag', tag, '--interval', iv, '--out-root', OUT])
