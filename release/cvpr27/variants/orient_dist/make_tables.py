# -*- coding: utf-8 -*-
"""tables for ORIENT_DIST.md from orient_dist.json, heading_modes.json, lidar_share.json (printed as markdown)."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(D + '/orient_dist.json')); H = json.load(open(D + '/heading_modes.json'))
L = json.load(open(D + '/lidar_share.json')) if os.path.exists(D + '/lidar_share.json') else None
f = lambda x, n=2: '%.*f' % (n, x)
def dci(d, n=2):
    return '%+.*f [%+.*f, %+.*f]' % (n, d['point'], n, d['ci'][0], n, d['ci'][1])
out = []
NAMES = {'mav': 'MAV6D test', 'antifs': 'AntiUAV3D v2 test'}
for tgt in ('mav', 'antifs'):
    out += ['### C1-%s. %s' % ('a' if tgt == 'mav' else 'b', NAMES[tgt]), '',
            '| Aircraft | n | Arm | TP | AOE-90° mean | AOE-180° mean | Heading err. median | Heading err. mean | heading <45° / 45–135° / ≥135° (%) |',
            '|---|---|---|---|---|---|---|---|---|']
    for sub, r in R['C1'][tgt].items():
        for k, arm in enumerate(('PST', 'RGB-only', 'Sim-only')):
            a = r[arm]; h = H[tgt][sub][arm]
            out.append('| %s | %s | %s | %d | %s | %s | %s | %s | %.0f / %.0f / %.0f |' % (sub if k == 0 else '', r['n'] if k == 0 else '', arm, a['n_tp'], f(a['AOE90']), f(a['AOE180']),
                       f(a['HEADmed']), f(a['HEADmean']), 100 * h['lt45'], 100 * h['mid'], 100 * h['ge135']))
        d = r['PST - RGB-only']
        out.append('| | | **PST − RGB-only** [95%% CI] | | %s | %s | %s | %s | |' % (dci(d['AOE90']), dci(d['AOE180']), dci(d['HEADmed']), dci(d['HEADmean'])))
    out.append('')
    out.append('Resampling: ' + '; '.join('%s = %s' % (s, r['resampling']) for s, r in R['C1'][tgt].items()) + '.')
    out.append('')
out += ['### C2. AntiUAV3D v2 test by GT range band', '',
        '| Band (m) | n (M4E/M400) | Arm | AP | ATE (rel.) | ASE (cm) | AOE-90° | mDR | UDS |', '|---|---|---|---|---|---|---|---|---|']
for b, r in R['C2'].items():
    for k, arm in enumerate(('PST', 'RGB-only', 'Sim-only')):
        a = r[arm]
        out.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (b if k == 0 else '', ('%d (%d/%d)' % (r['n'], r['n_M4E'], r['n_M400'])) if k == 0 else '', arm,
                   f(a['AP']), f(a['ATE'], 3), f(a['ASE']), f(a['AOE']), f(a['mDR'], 1), f(a['UDS'])))
    d = r['PST - RGB-only']
    out.append('| | | **PST − RGB-only** [95%% CI] | %s | %s | | %s | | %s |' % (dci(d['AP']), dci(d['ATE'], 3), dci(d['AOE90']), dci(d['UDS'])))
out.append('')
if L:
    out += ['### C2b. Simulated outdoor training objects with a non-empty LiDAR body map', '',
            '| Band (m) | all objects: n / share ≥5 returns / median returns | small airframes | large airframes | qualified objects |', '|---|---|---|---|---|']
    for b in ('0-20', '20-40', '40-60', '60+'):
        cells = []
        for g in ('all objects', 'small airframes', 'large airframes', 'qualified objects'):
            x = L[g][b]
            cells.append('%d / %.1f%% / %g' % (x['n_obj'], 100 * x['share_nonempty'], x['median_returns']) if x['n_obj'] else '0 / – / –')
        out.append('| %s | %s |' % (b, ' | '.join(cells)))
    out.append('')
print('\n'.join(out))
