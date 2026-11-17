# -*- coding: utf-8 -*-
"""Print markdown tables from a_block_boot.json and b_swap.json (pasted into A3_REPORT.md)."""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(OUT + '/a_block_boot.json')); Bs = json.load(open(OUT + '/b_swap.json', encoding='utf-8'))
f = lambda x, n=2: ('%+.*f' % (n, x))
ci = lambda c, n=2: '[%s, %s]' % (f(c[0], n), f(c[1], n))
print('## A')
for L in sorted(A['boot'], key=int):
    r = A['boot'][L]
    print('\nL =', L)
    print('| Arm | AP | 95% CI | UDS | 95% CI |'); print('|---|---|---|---|---|')
    for nm in ('PST', 'Sim-only', 'RGB-only teachers'):
        p = A['point'][nm]; print('| %s | %.2f | [%.2f, %.2f] | %.2f | [%.2f, %.2f] |' % (nm, p['AP'], *r[nm]['AP'], p['UDS'], *r[nm]['UDS']))
    for k in ('PST - Sim-only', 'PST - RGB-only teachers'):
        q = r[k]; print('| %s | %s | %s | %s | %s |' % (k, f(q['AP']['point']), ci(q['AP']['ci']), f(q['UDS']['point']), ci(q['UDS']['ci'])))
print('\n## B')
names = ['PST: LiDAR height ratio r_k'] + list(Bs['alternatives'])
for sp in ('test', 'val'):
    S = Bs['splits'][sp]; print('\n###', sp, 'rk', S['rk_from_deployed'], 'off', S['n_off_class_frames'], S['class_by_type'])
    print('| Height source | r dark / bright | UDS | ASE (cm) | mean \\|Δh\\| cm (% GT h) | Mavic 2 \\|Δh\\| cm (%) | Mavic 2 ASE | Phantom 4 \\|Δh\\| cm (%) | Phantom 4 ASE |')
    print('|---|---|---|---|---|---|---|---|---|')
    for nm in names:
        r = S['rows'][nm]; rr = Bs['alternatives'].get(nm, S['rk_from_deployed'])
        print('| %s | %.3f / %.3f | %.2f | %.2f | %.2f (%.1f) | %.2f (%.1f) | %.2f | %.2f (%.1f) | %.2f |' % (
            nm, rr[0], rr[1], r['UDS'], r['ASE'], r['all']['dh'], r['all']['dhr'], r['Mavic 2']['dh'], r['Mavic 2']['dhr'], r['Mavic 2']['ASE'],
            r['Phantom 4']['dh'], r['Phantom 4']['dhr'], r['Phantom 4']['ASE']))
    print('\nDiffs vs r_k (alt - r_k), seq bootstrap')
    print('| Height source | Group (seqs) | ΔUDS | ΔASE (cm) | Δ mean \\|Δh\\| (cm) | Δ mean \\|Δh\\| (% GT h) |'); print('|---|---|---|---|---|---|')
    for nm in Bs['alternatives']:
        for g in ('all', 'Mavic 2', 'Phantom 4'):
            r = S['rows'][nm]; r0 = S['rows'][names[0]]; b = S['boot'][g][nm]
            du = (r['UDS'] - r0['UDS']) if g == 'all' else (r[g]['UDS_sub'] - r0[g]['UDS_sub'])
            print('| %s | %s (%d) | %s %s | %s %s | %s %s | %s %s |' % (nm, g, S['boot'][g]['n_seq'], f(du), ci(b['UDS']), f(r[g]['ASE'] - r0[g]['ASE']), ci(b['ASE']),
                                                            f(r[g]['dh'] - r0[g]['dh']), ci(b['dh']), f(r[g]['dhr'] - r0[g]['dhr'], 1), ci(b['dhr'], 1)))
