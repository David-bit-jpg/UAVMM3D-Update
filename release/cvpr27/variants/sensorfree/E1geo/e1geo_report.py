# -*- coding: utf-8 -*-
"""[E1geo] write E1geo/RESULT.md from the per-run constants.json / boot_*.json and the pooled jsons (no labels read here)."""
import os, json
H = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
K = ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR')
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
f2 = lambda v: '%.3f' % v if abs(v) < 1 else '%.2f' % v
ci = lambda b, k: '%+.3f [%+.3f, %+.3f]' % (b['point'], *b['ci']) if k == 'ATE' else '%+.2f [%+.2f, %+.2f]' % (b['point'], *b['ci'])
fk = lambda d: ', '.join('%s: %.3f' % (k, float(v)) for k, v in sorted(d.items(), key=lambda x: str(x[0])) if v == v)
L = ['# E1geo: deploy-only variant of the sensor-free control E1 (MAV6D)', '',
     'E1geo = E1 with two DEPLOYMENT-only changes; every network, pseudo-label, w and stage-3 target is E1\'s own (read only).',
     '- **(a) two brightness classes**: PST\'s rule (1-D two-means of the silhouette body brightness `lum` over unlabeled train crops, nearest centre at',
     '  deployment), applied to E1\'s own hull-supervised region head; c_k (refiner offset, train) and beta\'_k (fusion offset, unlabeled val images)',
     '  recomputed per class exactly as `chain_x deploy`. **Training stays one-class**: the stage-3 fused targets (beta_k inside `chain_x targets`) were',
     '  built by E1 with one class and are NOT recomputed (that would need a stage-3 retrain).',
     '- **(b) per-class height ratio r_k** from box-corner PnP of the RGB-only KZ teachers (KZs0/1 after E1\'s AdaBN; H6 refit rule, l, w fixed at s_bar),',
     '  median over unlabeled train crops per class; h = r_k (l + w) / 2 as in `chain_x deploy`. Student-derived r_k is a diagnostic only (students are',
     '  trained toward s_bar-shaped pseudo boxes, so their aspect collapses to the constant). No IR/LiDAR-trained network is used.',
     '- Faithfulness: the patched chain copy with both switches off reproduces E1\'s own test predictions exactly (asserted per run).',
     '- Test scored once per run (pack score.py, real_test). Paired sequence bootstrap (10 test sequences, B = 2000) via heads/heads_boot.py.', '']
L += ['## Per run (test)', '', '| run | E1 UDS | E1geo UDS | PST UDS | E1geo - E1 UDS [95% CI] | E1geo - PST UDS [95% CI] | E1geo - E1 ASE | E1geo - E1 ATE [95% CI] |',
      '|---|---|---|---|---|---|---|---|']
CON = []
for s in (0, 10, 20):
    d = '%s/mav_s%d' % (H, s); a, b, c = J(d + '/boot_vsE1.json'), J(d + '/boot_vsPST.json'), J(d + '/constants.json')
    if os.path.exists(d + '/.skipped'):
        L.append('| mav_s%d | E1 run not finished (skipped) | | | | | | |' % s); continue
    if not (a and b):
        L.append('| mav_s%d | pending | | | | | | |' % s); continue
    pa, pb = a['pairs'][0], b['pairs'][0]
    L.append('| mav_s%d | %.2f | %.2f | %.2f | %s | %s | %+.2f | %s |' % (s, pa['ref']['UDS'], pa['arm']['UDS'], pb['ref']['UDS'],
             ci(dict(point=pa['diff']['UDS'], ci=pa['ci']['UDS']), 'UDS'), ci(dict(point=pb['diff']['UDS'], ci=pb['ci']['UDS']), 'UDS'),
             pa['diff']['ASE'], ci(dict(point=pa['diff']['ATE'], ci=pa['ci']['ATE']), 'ATE')))
    if c:
        CON.append('- mav_s%d: E1 one class c_k {%s} r_k {%s} beta\'_k {%s} | E1geo centres %s (train bright %.2f) c_k {%s} r_k {%s} beta\'_k {%s} | '
                   'student r_k (diag) {%s} | test frames without class %d' % (s, fk(c['E1']['c_k']), fk(c['E1']['r_k']), fk(c['E1']['b_k']),
                   [round(x, 1) for x in c['E1geo']['col']], c['E1geo']['train_frac_bright'], fk(c['E1geo']['c_k']), fk(c['E1geo']['r_k']),
                   fk(c['E1geo']['b_k']), fk(c['diag_student_r_k']), c['E1geo']['test_frames_without_class']))
L += ['', '## Pooled over finished seed pairs (mean of per-pair differences, pooled 95% CI, P(diff <= 0))', '']
for nm, lab in (('pooled_vsE1.json', 'E1geo - E1'), ('pooled_vsPST.json', 'E1geo - PST')):
    p = J(H + '/' + nm)
    if p:
        L.append('- **%s** (%d pairs): arm mean UDS %.2f, ref mean UDS %.2f | ' % (lab, len(p['pairs']), p['mean_arm']['UDS'], p['mean_ref']['UDS'])
                 + ' | '.join('%s %s (P<=0 %.3f)' % (k, ci(p['boot'][k], k), p['boot'][k]['frac_le0']) for k in K))
L += ['', '## Deployment constants', ''] + CON
L += ['', '## Reading', '', 'PST - E1 (the reported sensor-free gap) splits into PST - E1geo (what remains once E1 gets two brightness classes and a',
      'sensor-free per-class height ratio at deployment) plus E1geo - E1 (what those two non-sensor deployment pieces were worth inside E1).',
      'Caveat: E1geo cannot reproduce two-class stage-3 TRAINING targets (beta_k) or any upstream use; only deployment constants change.', '']
open(H + '/RESULT.md', 'w', encoding='utf-8').write('\n'.join(L))
print('RESULT.md written')
