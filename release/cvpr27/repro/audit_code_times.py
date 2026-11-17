# -*- coding: utf-8 -*-
"""For every reported run: which of its code files were modified after the run started (mtime > start), and which .bak_* copies exist.
Read-only. Output: code_time_audit.md / code_time_audit.json next to this file."""
import os, re, glob, json, time, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
G = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/'; X = G + 'srcx/'; R = G + 'review1113/'; CFG = 'E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/'
CODE_EXT = ('.py', '.sh', '.yaml', '.yml')
T = lambda s: dt.datetime.strptime('2026-' + s, '%Y-%m-%d %H:%M:%S')
mt = lambda p: dt.datetime.fromtimestamp(os.path.getmtime(p))


def starts(log):
    out = []
    for line in open(log, encoding='utf-8', errors='replace'):
        m = re.match(r'^(\d\d-\d\d \d\d:\d\d:\d\d)\s.*\bstart\b', line)
        if m and 'SKIPPED' not in line and 'waiting' not in line: out.append(T(m.group(1)))
    return out


def code_files(roots):
    fs = []
    for r in roots:
        if os.path.isfile(r): fs.append(r); continue
        for p in glob.glob(r + '/*') + glob.glob(r + '/*/*'):
            if os.path.isfile(p) and (p.endswith(CODE_EXT) or '.bak' in os.path.basename(p)) and '__pycache__' not in p: fs.append(p.replace('\\', '/'))
    return sorted(set(fs))


RUNS = [  # name, prediction file (end), run log(s) holding 'start' lines, code roots
    ('srcx/in2mav (Tab.1/2 full pipeline MAV6D, Tab.3 PST run 1)', X + 'in2mav/preds/pcmd_fused_test.pkl', [X + 'in2mav/logs/run_x.log'], [X + 'run_x.sh', X + 'lib', X + 'stage1_det.sh', CFG + 'srcx/real_in2mav.yaml', CFG + 'srcx/st_in2mav.yaml']),
    ('srcx/in2mav_none (Tab.3 RGB-only run 1)', X + 'in2mav_none/preds/pcmd_fused_test.pkl', [X + 'in2mav_none/logs/run_x.log'], [X + 'run_x.sh', X + 'lib', CFG + 'srcx/real_in2mav_none.yaml', CFG + 'srcx/st_in2mav_none.yaml']),
    ('srcx/out2antifs (Tab.3 PST run 1 RD; CenterNet* RD det0 reused by k249)', X + 'out2antifs/preds/pcmd_fused_test.pkl', [X + 'out2antifs/logs/run_x.log'], [X + 'run_x.sh', X + 'lib', X + 'stage1_det.sh', CFG + 'srcx/real_out2antifs.yaml', CFG + 'srcx/st_out2antifs.yaml']),
    ('srcx/out2antifs_none (Tab.3 RGB-only run 1 RD)', X + 'out2antifs_none/preds/pcmd_fused_test.pkl', [X + 'out2antifs_none/logs/run_x.log'], [X + 'run_x.sh', X + 'lib', CFG + 'srcx/real_out2antifs_none.yaml', CFG + 'srcx/st_out2antifs_none.yaml']),
    ('srcx/out2antifs_k249 (Tab.1 full pipeline RD)', X + 'out2antifs_k249/preds/pcmd_fused_test.pkl', [X + 'out2antifs_k249/logs/run_x.log'], [R + 'k249/run_k249.sh', R + 'k249/lib', R + 'k249/keys249.pkl']),
    ('review1113/sup3 (Tab.3 PST run 2 MAV6D)', R + 'sup3/run/preds/pcmd_fused_test.pkl', [R + 'sup3/run/logs/run_seed2.log'], [R + 'sup3/run_seed2.sh', R + 'sup3/make_seed_lib.py', R + 'sup3/lib', R + 'sup_patch.py']),
] + [('review1113/seeds/%s (Tab.3)' % n, R + 'seeds/%s/preds/pcmd_fused_test.pkl' % n, [R + 'seeds/%s/logs/run.log' % n],
      [R + 'seeds/%s/lib' % n, R + 'seeds/run_seed_chain.sh', R + 'seeds/run_all_seeds.sh', R + 'seeds/make_seeds_lib.py', R + 'sup_patch.py'])
     for n in ('mav_full_s20', 'mav_none_s10', 'mav_none_s20', 'antifs_full_s10', 'antifs_full_s20', 'antifs_none_s10', 'antifs_none_s20')] + \
    [('review1113/sensorfree/%s/%s (Tab.3)' % (a, n), R + 'sensorfree/%s/%s/preds/%s_fused_test.pkl' % (a, n, a.lower()), [R + 'sensorfree/%s/%s/logs/run.log' % (a, n), R + 'sensorfree/logs/sensorfree.log'],
      [R + 'sensorfree/%s/%s/lib' % (a, n), R + 'sensorfree/run_sf.sh', R + 'sensorfree/make_sf_lib.py', R + 'sensorfree/prep_hull.py', R + 'heads/heads_boot.py'])
     for a in ('E1', 'E4') for n in ('mav_s0', 'mav_s10', 'mav_s20', 'antifs_s0', 'antifs_s10', 'antifs_s20')] + [
    ('review1113/heads/H3/mav_s0 (Tab.2 row 5)', R + 'heads/H3/mav_s0/preds/h3_fused_test.pkl', [R + 'heads/H3/mav_s0/logs/run.log'], [R + 'heads/H3/mav_s0/lib', R + 'heads/run_arm.sh', R + 'heads/make_heads_lib.py']),
    ('review1113/clean2/buildup (Tab.2 rows 2-3)', R + 'clean2/buildup/b_det1_test.pkl', [], [R + 'clean2/buildup/buildup.py']),
] + [('review1113/ourdet_baseproto %s (Tab.1 CenterNet rows)' % n, R + 'ourdet_baseproto/preds/%s_test.pkl' % n, glob.glob(R + 'ourdet_baseproto/logs/*%s*.log' % n),
      [R + 'ourdet_baseproto/lib', R + 'ourdet_baseproto/ourdet_train.py', R + 'ourdet_baseproto/ourdet_dann_train.py', R + 'ourdet_baseproto/ub_zoom_train.py',
       R + 'ourdet_baseproto/train_eval.sh', R + 'ourdet_baseproto/common.sh', R + 'ourdet_baseproto/dann_eval.sh', R + 'ourdet_baseproto/ub_eval.sh', R + 'ourdet_baseproto/ubzoom_eval.sh', CFG + 'ourdet_baseproto'])
     for n in ('in2mav_sim_s0', 'in2mav_st', 'in2mav_pst', 'in2mav_dann', 'in2mav_ub', 'out2antifs_sim_s0', 'out2antifs_st', 'out2antifs_pst', 'out2antifs_dann', 'out2antifs_ubzoom')]
CM = 'E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/'
for n, m in (('monocd_s0', 'monocd'), ('monodgp_s0', 'monodgp'), ('gupnet_s0', 'gupnet'), ('gupnet_uda_dann_s0', 'gupnet')):
    RUNS.append(('D:/cmpfull_runs/%s (Tab.1 baseline, local run)' % n, glob.glob('D:/cmpfull_runs/%s/pred_real_test.pkl' % n)[0], [], [CM + m]))


def run_start(name, pred, logs):
    end = mt(pred); ss = []
    for l in logs:
        if os.path.isfile(l): ss += [s for s in starts(l) if s <= end]
    if ss: return max(ss), 'last "start" log line before the prediction file'
    # fallback: earliest log / run_info mtime in the run folder
    d = os.path.dirname(pred); cand = [p for p in glob.glob(d + '/*.log') + glob.glob(d + '/run_info.json') + glob.glob(d + '/train.log') + glob.glob(d + '/log.txt')]
    if cand: return min(mt(p) for p in cand) - dt.timedelta(seconds=0), 'earliest log/run_info mtime in the run folder (upper bound of the start)'
    return end, 'prediction mtime (no start time found)'


rows = []
for name, pred, logs, roots in RUNS:
    if not os.path.isfile(pred): rows.append(dict(run=name, error='missing ' + pred)); continue
    st, how = run_start(name, pred, logs); end = mt(pred)
    fs = [f for f in code_files(roots) if os.path.isfile(f)]
    changed = [(f, mt(f).strftime('%m-%d %H:%M:%S'), 'during run' if mt(f) <= end else 'after run') for f in fs if mt(f) > st + dt.timedelta(seconds=180) and '.bak' not in os.path.basename(f)]
    launch = [f for f in fs if st < mt(f) <= st + dt.timedelta(seconds=180)]
    baks = [(f, mt(f).strftime('%m-%d %H:%M:%S')) for f in fs if '.bak' in os.path.basename(f)]
    rows.append(dict(run=name, pred=pred, start=st.strftime('%m-%d %H:%M:%S'), start_how=how, end=end.strftime('%m-%d %H:%M:%S'), changed=changed, baks=baks, written_at_launch=len(launch)))

json.dump(rows, open(HERE + '/code_time_audit.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
L = ['# Code files modified after the start of a reported run', '', 'Files written within 3 min after the start are the per-run lib copies / generated configs made by the launcher itself and are not counted as changes.',
     'mtime of each code file vs the run window (start = last start line in the run log before the prediction file; end = prediction file mtime).',
     'A file listed here is the CURRENT version; the version that ran is the .bak_* copy only when a matching backup exists (see notes in README).', '']
for r in rows:
    if 'error' in r: L.append('- **%s**: %s' % (r['run'], r['error'])); continue
    L.append('## %s' % r['run']); L.append('window %s -> %s (%s)' % (r['start'], r['end'], r['start_how']))
    if r['changed']:
        for f, t, w in r['changed']: L.append('- CHANGED %s: `%s` (mtime %s)' % (w, f, t))
    else: L.append('- no code file modified after the start')
    if r.get('written_at_launch'): L.append('- %d file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change' % r['written_at_launch'])
    if r['baks']: L.append('- backups present: ' + ', '.join('`%s` (%s)' % (os.path.basename(f), t) for f, t in r['baks']))
    L.append('')
open(HERE + '/code_time_audit.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
