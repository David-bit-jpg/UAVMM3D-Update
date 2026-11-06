# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 1, S1 (GPU, inference only): dump the 5 deployed KZ members on a MAV6D split (default: TRAIN, all
12,866 frames). Wrapper around tools/annot_audit/dump_real_preds.py -- the repo is NOT modified.

Exactly the deployed pipeline of e1sa (run_kz_0923.dump('kpKZavg_zoom')): 5 ramp position models + flip TTA + causal100
temporal background + label-free self-calibrated lwh (val) + KZ zoom 95 from the RAW undistorted images; the recording
hook is make_hook() of ./dump_kz_members.py (byte copy of e1sa/dump_kz_members.py, unchanged); the only differences are
  --split <split>   passed through to dump_real_preds (it redirects DATA_SPLIT['test'] -> <split>),
  members = the 5 RGB KZ only (release_20260923/kz_s0..s4.pth; the deployed value is their mean, bit-identical),
  --interval N      (default 1 = protocol; 2 = documented fallback) applied AFTER dump_real_preds line 61, main loader only.
Leak barrier for split != test: the ground-truth summary lines printed by dump_real_preds are suppressed, and after the
dump the pkl is rewritten with gt = matched = None and info.det_rate = None (nothing downstream can read train labels).

Writes ONLY into this folder:  real_ens5flip_kpKZavg_zoom_<tag>.pkl  kz_members_<tag>.npz  kz_members_<tag>.json

    cd @@ROOT@@\\tools
    python <scratchpad>\\kzst1\\dump_kz_members_kzst.py --split train --tag train5
"""
import sys

sys.dont_write_bytecode = True          # do not drop .pyc files into @@ROOT@@

import os           # noqa: E402

os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '@@ROOT@@'
TOOLS = ROOT + '/tools'
for _p in (TOOLS + '/annot_audit', TOOLS, ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:      # noqa: BLE001
    pass

import argparse     # noqa: E402
import builtins     # noqa: E402
import hashlib      # noqa: E402
import json         # noqa: E402
import pickle       # noqa: E402
import re           # noqa: E402
import time         # noqa: E402

import dump_kz_members as DKM     # noqa: E402  (byte copy of e1sa/dump_kz_members.py)

MEMBER_SETS = {'rgb': DKM.RGB}


def build_argv(pkl, members, split):
    return ['dump_real_preds.py', '--kp2d-from'] + [p for _, p in members] + ['--kp2d-cfg', DKM.KP_CFG, '--out', pkl,
                                                                              '--kp2d-zoom', DKM.ZOOM, '--split', split]


def scrub_gt(pkl, split, interval):
    D = pickle.load(open(pkl, 'rb'))
    for f in D['frames']:
        f['gt'] = None
        f['matched'] = None
    D['info']['det_rate'] = None
    D['info']['gt_scrubbed'] = 'kzst1/dump_kz_members_kzst.py: gt / matched / det_rate removed (split=%s, never read)' % split
    D['info']['sampled_interval'] = int(interval)
    with open(pkl + '.tmp', 'wb') as f:
        pickle.dump(D, f)
    os.replace(pkl + '.tmp', pkl)
    return len(D['frames'])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    ap.add_argument('--tag', default='train5')
    ap.add_argument('--members', default='rgb', choices=sorted(MEMBER_SETS))
    ap.add_argument('--interval', type=int, default=1, help='1 = protocol (all frames); 2 = fallback subsample')
    ap.add_argument('--out-dir', default=HERE)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    out_dir = os.path.abspath(a.out_dir)
    if not os.path.normcase(out_dir).startswith(os.path.normcase(HERE)):
        raise SystemExit('refusing to write outside %s' % HERE)
    if a.interval < 1:
        raise SystemExit('--interval must be >= 1')
    os.makedirs(out_dir, exist_ok=True)
    pkl = os.path.join(out_dir, 'real_ens5flip_kpKZavg_zoom_%s.pkl' % a.tag)
    if os.path.exists(pkl) and not a.force:
        raise SystemExit('%s exists (use --force to redo)' % pkl)
    members = MEMBER_SETS[a.members]
    missing = [p for _, p in members if not os.path.exists(p)]
    if missing:
        raise SystemExit('missing checkpoints: %s' % missing)
    names = [n for n, _ in members]

    sys.argv = build_argv(pkl, members, a.split)
    import kp_zoom                          # noqa: E402  (same module object dump_real_preds imports)
    import dump_real_preds as DRP           # noqa: E402  (chdir to tools at import, like the original runs)
    from uavdet3d.utils import pose_eval    # noqa: E402
    saved = {'hook': kp_zoom.install_zoom_kp2d_hook, 'bd': DRP.build_dataloader,
             'summ': pose_eval.summarize, 'fmt': pose_eval.format_summary}
    kp_zoom.install_zoom_kp2d_hook = DKM.make_hook(names, n_rgb=len(names))
    calls = {'n': 0}

    def bd(dataset_cfg, *args, **kw):          # main loader = first call (dump_real_preds line 66); KZ net loader untouched
        if calls['n'] == 0:
            assert str(dataset_cfg.DATA_SPLIT['test']) == a.split, dict(dataset_cfg.DATA_SPLIT)
            dataset_cfg.SAMPLED_INTERVAL['test'] = int(a.interval)
        calls['n'] += 1
        return saved['bd'](dataset_cfg, *args, **kw)
    DRP.build_dataloader = bd
    if a.split != 'test':
        pose_eval.summarize = lambda *x, **k: {}
        pose_eval.format_summary = lambda tag, s: '[kzst1] %s: ground-truth summary suppressed (split=%s)' % (tag, a.split)

        def masked_print(*args, **kw):
            s = ' '.join(str(x) for x in args)
            s = re.sub(r'检出 [0-9.]+%', '检出 <masked>', s)
            builtins.print(s, **kw)
        DRP.print = masked_print
    print('[kzst1] cwd %s; split %s; interval %d; argv %s' % (os.getcwd(), a.split, a.interval, ' '.join(sys.argv[1:])), flush=True)
    DKM.REC['t0'] = time.time()
    ok = False
    try:
        DRP.main()
        ok = True
    finally:
        kp_zoom.install_zoom_kp2d_hook = saved['hook']
        DRP.build_dataloader = saved['bd']
        pose_eval.summarize, pose_eval.format_summary = saved['summ'], saved['fmt']
        if 'print' in DRP.__dict__:
            del DRP.print
        n_scrub = None
        if ok and a.split != 'test' and os.path.exists(pkl):
            n_scrub = scrub_gt(pkl, a.split, a.interval)
        DKM.save_records(out_dir, a.tag, rec=DKM.REC, extra_info={
            'members': [{'name': n, 'path': p, 'role': 'rgb_deployed'} for n, p in members],
            'dump_pkl': os.path.basename(pkl), 'dump_argv': sys.argv[1:], 'dump_finished_ok': ok, 'split': a.split,
            'sampled_interval': int(a.interval), 'gt_scrubbed_frames': n_scrub,
            'dump_pkl_sha256_after_scrub': hashlib.sha256(open(pkl, 'rb').read()).hexdigest() if (ok and os.path.exists(pkl)) else None,
            'wrapper': os.path.basename(__file__), 'wrapper_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
            'prereg_sha256': (hashlib.sha256(open(os.path.join(HERE, 'prereg_stage1.json'), 'rb').read()).hexdigest()
                              if os.path.exists(os.path.join(HERE, 'prereg_stage1.json')) else None),
            'mirrors': 'e1sa/dump_kz_members.py (= run_kz_0923 dump kpKZavg_zoom) with --split %s, 5 RGB members only' % a.split})
    if ok:
        print('[kzst1] done: %s (gt scrubbed: %s)' % (pkl, n_scrub is not None), flush=True)
    return pkl


if __name__ == '__main__':
    main()
