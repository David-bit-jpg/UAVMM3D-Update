# -*- coding: utf-8 -*-
"""[clean] markdown tables for items 3 and 4 from height/height_clean.json and diag/ate_diag.json (prints to stdout)."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
H = json.load(open(D + '/height/height_clean.json', encoding='utf-8'))
A = json.load(open(D + '/diag/ate_diag.json', encoding='utf-8'))
f2 = lambda x: '%.2f' % x
ci = lambda v, s=1.0, n=2: ('%+.*f [%+.*f, %+.*f]' % (n, v[0] * s, n, v[1] * s, n, v[2] * s))
print('## item 3')
for sp in ('test', 'val'):
    S = H['splits'][sp]; print('\n**%s**\n' % sp)
    print('| Height source | r (dark / bright) | UDS | AP | ATE (m) | ASE (cm) | mean |Δh| cm (% GT h) | Mavic 2 |Δh| cm (%) | Mavic 2 ASE | Phantom 4 |Δh| cm (%) | Phantom 4 ASE |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    rv = {'PST: LiDAR height ratio r_k': '%.3f / %.3f' % (H['r_k']['0'], H['r_k']['1'])}
    rv.update({k: '%.3f' % v for k, v in H['alternatives'].items()})
    for nm, r in S['rows'].items():
        print('| %s | %s | %.2f | %.2f | %.3f | %.2f | %.2f (%.1f) | %.2f (%.1f) | %.2f | %.2f (%.1f) | %.2f |' % (
            nm, rv[nm], r['UDS'], r['AP'], r['ATE'], r['ASE'], r['all']['dh'], r['all']['dhr'], r['Mavic 2']['dh'], r['Mavic 2']['dhr'], r['Mavic 2']['ASE'],
            r['Phantom 4']['dh'], r['Phantom 4']['dhr'], r['Phantom 4']['ASE']))
print('\n| Split | Height source | Group (seqs) | ΔUDS | ΔASE (cm) | Δ mean |Δh| (cm) | Δ mean |Δh| (% GT h) |')
print('|---|---|---|---|---|---|---|')
for sp in ('test', 'val'):
    for nm in H['alternatives']:
        for g in ('all', 'Mavic 2', 'Phantom 4'):
            b = H['splits'][sp]['boot'][g]
            print('| %s | %s | %s (%d) | %s | %s | %s | %s |' % (sp, nm, g, b['n_seq'], ci(b[nm]['UDS']), ci(b[nm]['ASE']), ci(b[nm]['dh']), ci(b[nm]['dhr'], n=1)))
print('\nclass counts', {sp: H['splits'][sp]['n_class'] for sp in H['splits']}, {sp: H['splits'][sp]['class_by_type'] for sp in H['splits']})
print('\n## item 4')
for sp in ('test', 'val'):
    R = A[sp]['arms']; print('\n**%s**\n' % sp)
    print('| Arm | UDS | AP | ATE (m) | mDR | TPs | bias b = median log(z/z_gt) | spread (1.4826 MAD) | bias Mavic 2 / Phantom 4 | bias dark / bright class | ATE after −b | UDS after −b | AP after −b | UDS after per-aircraft −b |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for nm, r in R.items():
        s = r['score']; a = r['all']; t = r['by_type']; c = r['by_class']; d = r['debias_global']
        print('| %s | %.2f | %.2f | %.3f | %.1f | %d | %+.1f%% | %.1f%% | %+.1f%% / %+.1f%% | %+.1f%% / %+.1f%% | %.3f | %.2f | %.2f | %.2f |' % (
            nm, s['UDS'], s['AP'], s['ATE'], s['mDR'], a['n'], 100 * a['bias'], 100 * a['spread'], 100 * t['Mavic 2']['bias'], 100 * t['Phantom 4']['bias'],
            100 * c['0']['bias'], 100 * c['1']['bias'], d['ATE'], d['UDS'], d['AP'], r['debias_per_aircraft']['UDS']))
    print('\nCommon TP frames with sim-only (%d frames):\n' % A[sp]['common_tp_with_simonly']['clean: PST (+ corner fusion)']['n'])
    print('| Arm | ATE on sim-only TP frames | bias there | spread there | extra TP frames (not TP for sim-only) | ATE on extra frames |')
    print('|---|---|---|---|---|---|')
    for nm, r in A[sp]['common_tp_with_simonly'].items():
        print('| %s | %.3f | %+.1f%% | %.1f%% | %d | %s |' % (nm, r['ATE'], 100 * r['bias'], 100 * r['spread'], r['n_only_this'], '%.3f' % r['ATE_only_this'] if r['ATE_only_this'] else '–'))
    print('\nStages:', json.dumps(A[sp]['stages'], default=lambda x: round(float(x), 4)))
