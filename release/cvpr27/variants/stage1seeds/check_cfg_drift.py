# -*- coding: utf-8 -*-
"""[stage1seeds] read-only, CPU: has any stage-1 training config changed since the reported networks were trained?
For every reference run (the checkpoint dir that produced a reported stage-1 network) the cfg dump that train.py wrote into its train_*.log
("cfg.X.Y: value" lines) is compared with a dump of the CURRENT yaml (same resolution: cfg_from_yaml_file + log_config_to_file).
Ignored keys: ROOT_DIR, LOCAL_RANK, TAG, EXP_GROUP_PATH (set from the path) and MODEL.DISTILL.TEACHER_CKPT (set on the command line).
usage: python check_cfg_drift.py            -> prints SAME / DIFF per network type, with the differing keys."""
import os, sys, glob, re, io, logging
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, 'E:/Open3DUAVDet'); os.chdir('E:/Open3DUAVDet/tools')
from easydict import EasyDict
from uavdet3d.config import cfg_from_yaml_file, log_config_to_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1paths as SP  # noqa

IGN = ('cfg.ROOT_DIR', 'cfg.LOCAL_RANK', 'cfg.TAG', 'cfg.EXP_GROUP_PATH', 'cfg.MODEL.DISTILL.TEACHER_CKPT')


def dump_current(cfg_file):
    c = EasyDict(); cfg_from_yaml_file(cfg_file, c)
    buf = io.StringIO(); lg = logging.getLogger('drift'); lg.handlers = []; h = logging.StreamHandler(buf); lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    log_config_to_file(c, pre='cfg', logger=lg)
    return parse(buf.getvalue().split('\n'))


def parse(lines):
    out = {}
    for l in lines:
        m = re.search(r'(cfg\.[A-Za-z0-9_.]+): (.*)$', l)
        if m and not m.group(1).startswith(IGN):
            out[m.group(1)] = m.group(2).strip()
    return out


def dump_log(run_dir):
    fs = sorted(glob.glob(run_dir + '/train_*.log'))   # newest log that reached 'End training' (an aborted first attempt is skipped)
    f = next((x for x in reversed(fs) if 'End training' in open(x, encoding='utf-8', errors='replace').read()[-3000:]), fs[-1])
    L = open(f, encoding='utf-8', errors='replace').read().split('\n')
    stop = next((i for i, l in enumerate(L) if 'Model ' in l and 'created' in l), 2000)
    return parse([l for l in L[:stop] if 'INFO' in l and 'cfg.' in l]), f


if __name__ == '__main__':
    bad = 0
    for src in ('in', 'out'):
        for item, ref in SP.REFERENCE_RUNS[src].items():
            cfg = SP.SPEC[src][item.split(':')[0]]['cfg']
            a, f = dump_log(ref); b = dump_current(cfg)
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            print('%-4s %-6s %s  (log %s, %d keys)' % (src, item, 'SAME' if not diff else 'DIFF %d keys' % len(diff), os.path.basename(f), len(a)))
            for k in diff[:12]:
                print('       %s | log: %s | now: %s' % (k, a.get(k), b.get(k)))
            bad += bool(diff)
    print('cfg drift check:', 'all SAME' if not bad else '%d network types DIFFER' % bad)
