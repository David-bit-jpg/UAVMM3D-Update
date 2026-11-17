# R1: per-seed mean and range for detector-level feature-transfer variants (MAV6D real test, single model)
import json, os, re, sys
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/tools')
from pair_compare import load
C = 'E:/Open3DUAVDet/output/camnorm'
DIRS = {'kd': C + '/kd/json', 'n1003': C + '/night1003/json', 'irloc': C + '/geo1004/irloc_json', 'dual': C + '/geo1004/dual_json'}
def get(arm, d, seeds=range(5)):
    out = {}
    for s in seeds:
        p = os.path.join(DIRS[d], '%s_s%d.json' % (arm, s))
        if os.path.exists(p):
            out[s] = load(p)
    return out
ROWS = [('RAMP', 'kd', None), ('KDrgb', 'kd', 'RAMP'), ('KDmm', 'kd', 'RAMP'), ('KDmm2', 'kd', 'RAMP'), ('KDmm3', 'kd', 'RAMP'),
        ('MMH', 'kd', 'RAMP'), ('MMHP', 'kd', 'RAMP'), ('KDir', 'kd', 'RAMP'), ('MMHir', 'kd', 'RAMP'), ('KDF', 'n1003', 'RAMP'),
        ('ST2', 'kd', None), ('ST2KD', 'n1003', 'ST2'),
        ('P0', 'irloc', None), ('PIM', 'irloc', 'P0'), ('PIO', 'irloc', 'P0'), ('PIH', 'irloc', 'P0'),
        ('D0', 'dual', 'P0'), ('DH', 'dual', 'P0'), ('DHs', 'dual', 'P0')]
dirs_of = {r[0]: r[1] for r in ROWS}
M = [('pos_median', 100, 'pos cm'), ('det_rate', 100, 'det %'), ('ads_indoor', 1, 'ADS')]
res = {}
for arm, d, base in ROWS:
    A = get(arm, d)
    row = {'n': len(A), 'seeds': sorted(A)}
    for k, sc, _ in M:
        v = np.array([A[s].get(k, np.nan) * sc for s in sorted(A)], float)
        row[k] = dict(mean=float(np.nanmean(v)), min=float(np.nanmin(v)), max=float(np.nanmax(v)), vals=v.round(2).tolist())
        if base:
            B = get(base, dirs_of[base])
            ss = [s for s in sorted(A) if s in B]
            dd = np.array([A[s][k] * sc - B[s][k] * sc for s in ss])
            row[k]['diff'] = dict(mean=float(dd.mean()), min=float(dd.min()), max=float(dd.max()), vals=dd.round(2).tolist(),
                                 better=int((dd < 0).sum() if k == 'pos_median' else (dd > 0).sum()), n=len(ss))
    res[arm] = row
json.dump(res, open('r1_seeds.json', 'w'), indent=1, ensure_ascii=False)
for arm, d, base in ROWS:
    r = res[arm]
    s = '%-6s n=%d ' % (arm, r['n'])
    for k, sc, nm in M:
        x = r[k]
        s += '| %s %.2f [%.2f,%.2f] ' % (nm, x['mean'], x['min'], x['max'])
        if 'diff' in x:
            s += 'd %+.2f [%+.2f,%+.2f] %d/%d ' % (x['diff']['mean'], x['diff']['min'], x['diff']['max'], x['diff']['better'], x['diff']['n'])
    print(s)
