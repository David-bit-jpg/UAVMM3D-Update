# Render manifest.json (written by rescore_all.py) as MANIFEST.md: table cell -> files -> aggregation -> command.
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
M = json.load(open(HERE + '/manifest.json', encoding='utf-8'))
SHORT = [('D:/advisor_results_1117b/runs/', 'A/'), ('C:/Users/jiang/OneDrive/文档/xwechat_files/wxid_pu1uzzbuzwsx22_b4ed/msg/file/2026-11/results_to_send_20261116_1459/runs/', 'W/'),
         ('D:/cmpfull_runs/', 'L/'), ('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/', 'X/'), ('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/', 'R/')]


def s(p):
    for a, b in SHORT: p = p.replace(a, b)
    return p


L = ['# MANIFEST: every number of Tab. 1-3 (and the main-text numbers) -> prediction files -> aggregation -> command', '',
     'Generated %s from `manifest.json` (written by `rescore_all.py`). Paper: D:/CVPR27_UAV (read-only).' % M['generated'], '',
     'Prefixes: `A/` = D:/advisor_results_1117b/runs/ (advisor server, 11-17 delivery), `L/` = D:/cmpfull_runs/ (local baseline runs),',
     '`X/` = E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/ (clean chain), `R/` = E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ (variants).', '',
     'Scorer `%s` (sha256 %s), most confident box per frame, `S.metrics(..., np.arange(N), S.CFG[split])`.' % (M['scorer'], M['scorer_sha256'][:16]),
     'GT: ' + '; '.join('%s = `%s` (sha256 %s)' % (k, v['path'], v['sha256'][:16]) for k, v in M['metas'].items()), '',
     'Commands (CPU only, `D:/Miniconda3/envs/city/python.exe`):', '- everything: `python rescore_all.py` -> `rescore_report.md`, `manifest.json`',
     '- one cell group: `python rescore_one.py <mav|anti> <pred files of the row>` (mean over the listed runs; RealDrone3D ATE in %)', '',
     'Aggregation: "single run" = the metric of that run; "mean over n runs" = arithmetic mean of the per-run metrics (so UDS is averaged per run);',
     'Tab. 3 = mean and sample standard deviation (ddof = 1) of the per-run UDS of three runs (seed shift 0, 10, 20).', '',
     '## Table cells (grouped by row)', '', '| table | row | target | printed AP / ATE / ASE / AOE / mDR / UDS | ok | aggregation | prediction files |', '|---|---|---|---|---|---|---|']
grp = collections.OrderedDict()
for c in M['cells']:
    if not c['id'].startswith('tab'): continue
    parts = c['id'].split('|')
    if parts[0] == 'tab1': key = ('Tab. 1', parts[1], parts[2])
    elif parts[0] == 'tab2': key = ('Tab. 2', parts[1], 'mav')
    else: key = ('Tab. 3', parts[1], parts[2])
    grp.setdefault(key, []).append(c)
for (t, row, tgt), cs in grp.items():
    pr = ' / '.join(c['printed'] for c in cs); ok = 'all' if all(c['ok'] for c in cs) else 'MISMATCH: ' + ', '.join('%s %s->%s' % (c['id'].split('|')[-1], c['printed'], c['recomputed']) for c in cs if not c['ok'])
    files = '<br>'.join('`%s`' % s(p) for p in cs[0]['files'])
    L.append('| %s | %s | %s | %s | %s | %s | %s |' % (t, row.replace('|', '/'), {'mav': 'MAV6D', 'anti': 'RealDrone3D'}[tgt], pr, ok, cs[0]['how'], files))
L += ['', '## Main-text numbers and claims', '', '| id | where | printed | recomputed | ok | how |', '|---|---|---|---|---|---|']
for c in M['cells']:
    if c['id'].startswith('tab'): continue
    L.append('| %s | %s | %s | %s | %s | %s |' % (c['id'][5:], c['where'], c['printed'], c['recomputed'], 'yes' if c['ok'] else '**NO**', c['how']))
L += ['', '## Prediction files (sha256, size, mtime)', '', '| file | sha256 | size | mtime |', '|---|---|---|---|']
for p, v in sorted(M['files'].items()):
    L.append('| `%s` | %s | %d | %s |' % (s(p), v['sha256'], v['size'], v['mtime']))
open(HERE + '/MANIFEST.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('MANIFEST.md rows:', len(L))
