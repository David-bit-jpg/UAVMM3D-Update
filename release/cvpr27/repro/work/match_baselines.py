# Discovery helper: for every baseline row of tab_main, find the subset of saved runs whose mean reproduces all printed cells.
import json, re, itertools, sys
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/repro_1117')
from texparse import main_rows
SC = json.load(open('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/repro_1117/work/scan_baselines.json', encoding='utf-8'))
SC = {p: v for p, v in SC.items() if 'skip' not in v}
K6 = ['AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS']
STEP = {'': 'sim', '+ST': 'st', '+PST': 'pst', '+DANN': 'dann'}


def cands(m, step, tgt):
    out = []
    for p, v in SC.items():
        if v['target'] != tgt: continue
        d = p.split('/')[-2]
        pats = {('sim', 'mav'): [r'^%s_in_sim_s\d$', r'^%s_s0$'], ('sim', 'anti'): [r'^%s_out_sim_s\d$', r'^%s_out_sim_local_s0$'],
                ('st', 'mav'): [r'^%s_in_st_mav_s\d$', r'^%s_st_real_s0$'], ('st', 'anti'): [r'^%s_out_st_anti_s\d$'],
                ('pst', 'mav'): [r'^%s_in_kdc_mav_s\d$', r'^%s_kd_clean_real_s0$'], ('pst', 'anti'): [r'^%s_out_kd_anti_s\d$'],
                ('dann', 'mav'): [r'^%s_uda_dann_s0$'], ('dann', 'anti'): [r'^%s_out_dann_anti_s\d$']}[(step, tgt)]
        if any(re.match(q % m, d) for q in pats): out.append(p)
    u = {}
    for p in out: u.setdefault(SC[p]['sha256'], []).append(p)
    return [v[0] for v in u.values()], u


def fmt(v, s):
    d = len(s.split('.')[1]) if '.' in s else 0
    return ('%.' + str(d) + 'f') % v


for (row, tgt), cells in main_rows().items():
    mm = re.match(r'(GUPNet|MonoDETR|MonoCD|MonoDGP|MonoCoP)\s*(\+\w+)?', row)
    if not mm: continue
    m = mm.group(1).lower(); step = STEP[mm.group(2) or '']
    cs, dup = cands(m, step, tgt)
    hits = []
    for k in range(1, min(4, len(cs)) + 1):
        for sub in itertools.combinations(cs, k):
            vals = {c: np.mean([SC[p][c] * (100 if (c == 'ATE' and tgt == 'anti') else 1) for p in sub]) for c in K6}
            nok = sum(fmt(vals[c], cells[c]) == cells[c] for c in K6)
            hits.append((nok, sub))
    hits.sort(key=lambda h: -h[0])
    best = hits[0] if hits else (0, ())
    print('%-14s %-4s %d/6  %s' % (row, tgt, best[0], ' | '.join(p.split('/')[-2] + '@' + p.split('/')[1][:12] for p in best[1])))
    for h in hits[1:3]:
        if h[0] == best[0]: print('      tie:', ' | '.join(p.split('/')[-2] for p in h[1]))


if __name__ == '__main__' and len(sys.argv) > 1:
    out = {}
    for (row, tgt), cells in main_rows().items():
        mm = re.match(r'(GUPNet|MonoDETR|MonoCD|MonoDGP|MonoCoP)\s*(\+\w+)?', row)
        if not mm: continue
        cs, dup = cands(mm.group(1).lower(), STEP[mm.group(2) or ''], tgt); best = None
        for k in range(1, min(4, len(cs)) + 1):
            for sub in itertools.combinations(cs, k):
                vals = {c: np.mean([SC[p][c] * (100 if (c == 'ATE' and tgt == 'anti') else 1) for p in sub]) for c in K6}
                if all(fmt(vals[c], cells[c]) == cells[c] for c in K6) and best is None: best = sub
        out['%s|%s' % (row, tgt)] = dict(files=list(best), identical_copies={p: dup[SC[p]['sha256']] for p in best})
    json.dump(out, open(sys.argv[1], 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    with open('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/repro_1117/baseline_runs.py', 'w', encoding='utf-8') as f:
        f.write('# -*- coding: utf-8 -*-\n# Baseline rows of Tab. 1 -> saved prediction files (found by work/match_baselines.py: the unique run subset whose mean reproduces all six printed cells).\n')
        f.write('# Byte-identical copies of the same file under the 11-16 delivery (W) are listed in manifest/baseline_runs.json.\nBASELINE = {\n')
        for k, v in out.items():
            row, tgt = k.split('|'); f.write('    (%r, %r): %r,\n' % (row, tgt, v['files']))
        f.write('}\n')
