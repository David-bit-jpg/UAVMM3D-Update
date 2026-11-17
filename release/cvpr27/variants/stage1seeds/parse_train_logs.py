# -*- coding: utf-8 -*-
"""[stage1seeds] read-only: parse train.py-style logs of the stage-1 networks -> args, iters/epoch, minutes/epoch, wall time.
usage: python parse_train_logs.py <ckpt run dir> ...      (prints one block per run; uses every train_*.log in the dir)"""
import sys, re, glob, os, datetime as dt
KEYS = ('cfg_file', 'batch_size', 'epochs', 'workers', 'extra_tag', 'ckpt', 'pretrained_model', 'pretrained_skip', 'freeze', 'fix_random_seed', 'seed',
        'cudnn_benchmark', 'val_split', 'val_interval', 'val_every', 'val_match', 'skip_test_eval', 'max_ckpt_save_num', 'set_cfgs', 'use_amp', 'logger_iter_interval')
def ts(l):
    try: return dt.datetime.strptime(l[:19], '%Y-%m-%d %H:%M:%S')
    except Exception: return None
for d in sys.argv[1:]:
    logs = sorted(glob.glob(d + '/train_*.log'))
    print('==', d.replace(os.sep, '/').split('camnorm/')[-1], '(%d logs)' % len(logs))
    for f in logs:
        L = open(f, encoding='utf-8', errors='replace').read().split('\n')
        a = {}
        for l in L[:80]:
            m = re.match(r'.{19},\d+\s+INFO\s+(\w+)\s+(.*)$', l)
            if m and m.group(1) in KEYS and m.group(1) not in a: a[m.group(1)] = m.group(2).strip()
        ep = [(int(m.group(1)), float(m.group(2))) for l in L for m in [re.search(r'本轮平均损失（(\d+) iter，([\d.]+) 分钟）', l)] if m]
        teach = [l.split('INFO')[-1].strip()[:160] for l in L[:400] if re.search(r'TEACHER|teacher|DISTILL\.|PRETRAIN|LIDAR|TEMPLATE', l)][:6]
        t0 = next((ts(l) for l in L if ts(l)), None); t1 = next((ts(l) for l in reversed(L) if ts(l)), None)
        end = any('End training' in l for l in L[-5:])
        print('  log', os.path.basename(f), '| wall %.2f h' % ((t1 - t0).total_seconds() / 3600 if t0 and t1 else -1), '| end' if end else '| NOT ENDED')
        print('   args', ' '.join('%s=%s' % (k, a[k]) for k in KEYS if k in a))
        if ep:
            its = [e[0] for e in ep]; mins = [e[1] for e in ep]
            print('   epochs logged %d, iters/epoch %s, min/epoch median %.2f (min %.2f max %.2f), sum %.2f h' % (len(ep), sorted(set(its)), sorted(mins)[len(mins)//2], min(mins), max(mins), sum(mins) / 60))
        for t in teach: print('   cfg:', t)
