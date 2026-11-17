# -*- coding: utf-8 -*-
"""[review1113 rgbst_equal] score one GUPNet prediction file with the baseline pack's score() (D:/UAV3D_baselines_pack/scripts/score.py, the
metric of the main-table baseline rows) WITHOUT running its __main__ (which would overwrite the pack's runs/SCORES.md). usage: <pred.pkl> <split>"""
import json, sys
sys.path.insert(0, 'D:/UAV3D_baselines_pack/scripts')
import score as S  # noqa: E402
r = S.score(sys.argv[1], sys.argv[2]); a = r['all']
print('UDS %.2f AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.1f' % (a['UDS'], a['AP'], a['ATE'], a['ASE'], a['AOE'], a['mDR']))
json.dump(r, open(sys.argv[1] + '.score_%s.json' % sys.argv[2], 'w'), indent=1, default=float)
