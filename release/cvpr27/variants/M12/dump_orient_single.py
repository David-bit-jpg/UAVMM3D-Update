# -*- coding: utf-8 -*-
"""[M12] orientation-crop dump for the SINGLE-model variant (= nolabel run_proto.infer -> proto/dump_orient_p.py, unchanged recipe) but:
position detector = NL075_s0 only (no --ensemble members), --no-flip, --bg none (strict single frame, as the nl075 chain), output root = M12/M1/repl.
The 5 LOO KZ members are passed exactly as dump_orient_p.py does (they only fix the windows/replay check; the deployed orientation comes from
the recut with the S6I student). Ground-truth summary is suppressed for every split (pose_eval.summarize stubbed before main()).
usage (cwd = E:/Open3DUAVDet/tools): python memcap.py dump_orient_single.py <val|test>"""
import os, sys
sp = sys.argv[1]
OUTR = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/M1/repl'
os.makedirs(OUTR, exist_ok=True)
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/night1003/scratch/repl')
import repl_common as RC  # noqa
RC.OUT_ROOT = OUTR
sys.path.insert(0, 'E:/Open3DUAVDet/tools')
from uavdet3d.utils import pose_eval  # noqa
pose_eval.summarize = lambda *x, **k: {}
pose_eval.format_summary = lambda tag, s: '[M12] %s: ground-truth summary suppressed' % tag
import dump_members_repl as M  # noqa
sys.path.insert(0, 'E:/Open3DUAVDet/tools/annot_audit')
import re, builtins  # noqa
import dump_real_preds as DRP  # noqa  (same module object main() imports)
DRP.print = lambda *x, **k: builtins.print(re.sub(r'检出 [0-9.]+%', '检出 <masked>', ' '.join(str(v) for v in x)), **k)
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'
CK0 = 'E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/st2nl075wa_mav6d_r34_geo_full_tr/NL075_s0/ckpt/checkpoint_epoch_3.pth'
M.POS_SETS['single'] = [CK0]
_ba = M.build_argv
M.build_argv = lambda *a, **k: _ba(*a, **k) + ['--bg', 'none', '--no-flip']
LOO = [G + '/kzst7/train_q/LOO_s%d/ckpt/checkpoint_epoch_3.pth' % k for k in range(5)]
M.main(['--split', sp, '--pos', 'single', '--ckpts'] + LOO + ['--tag', 'nl075_%s_kzt7q' % sp, '--out-root', OUTR])
