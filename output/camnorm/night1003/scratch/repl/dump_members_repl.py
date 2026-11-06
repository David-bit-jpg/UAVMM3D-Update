# -*- coding: utf-8 -*-
"""Experiments R / C (GPU, inference only): full deployed pipeline on a MAV6D split with a chosen position ensemble and a
chosen list of KZ member checkpoints; records every member's 5x5 hm / kp2d window (e1sa hook) and SEALS the ground truth.

Wrapper around tools/annot_audit/dump_real_preds.py (repo not modified), same construction as kzst1/dump_kz_members_kzst.py:
  * position ensemble: --pos sim  = dump_real_preds defaults (release_20260920 ramp_s0..s4: pure-sim, 0.273 m deployment)
                       --pos st2  = --ckpt st2_s0 --ensemble st2_s1..s4 (release_20260924, the deployed ST2 recipe)
    + flip TTA + causal100 temporal background + label-free self-calibrated lwh on val (conf 0.7, interval 2) + KZ zoom 95;
  * --members kz (release_20260923/kz_s0..s4, deployed) | adabn (kzst1/adabn/kz_s*_adabn.pth) | --ckpts <list>: the hook
    forwards these members and hands the pipeline their mean (deployed statement of tools/kp_zoom.py);
  * recording hook = make_hook() of ./dump_kz_members.py (byte copy of e1sa/dump_kz_members.py, unchanged);
  * ground truth: after dump_real_preds.main() wrote its pkl, gt / matched / det_rate are moved into sealed_gt_<tag>.pkl
    and the pkl keeps label-free fields only. For split != test the ground-truth summary lines are also suppressed.

Writes ONLY into @@ROOT@@/output/camnorm/night1003/repl/<tag>/ :
    real_ens5flip_<tag>.pkl  sealed_gt_<tag>.pkl  kz_members_<tag>.npz  kz_members_<tag>.json

    cd @@ROOT@@/tools
    python <repl>/dump_members_repl.py --split val  --pos sim --members kz --tag val_kz          (experiment R, dump 1)
    python <repl>/dump_members_repl.py --split test --pos st2 --members kz --tag st2_test_kz     (experiment C, dump 1)
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
import re           # noqa: E402
import time         # noqa: E402

import repl_common as RC          # noqa: E402
import dump_kz_members as DKM     # noqa: E402  (byte copy of e1sa/dump_kz_members.py)

NAMES = ['KZs%d' % k for k in range(5)]          # member names in the npz (consumers index members by these names)
POS_SETS = {'sim': None, 'st2': RC.ST2}
MEMBER_SETS = {'kz': RC.RELEASE_KZ, 'adabn': RC.ADABN_CKPTS}


def build_argv(pkl, ckpts, split, pos):
    argv = ['dump_real_preds.py']
    if pos is not None:
        argv += ['--ckpt', pos[0], '--ensemble'] + list(pos[1:])
    return argv + ['--kp2d-from'] + list(ckpts) + ['--kp2d-cfg', DKM.KP_CFG, '--out', pkl, '--kp2d-zoom', DKM.ZOOM, '--split', split]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--pos', default='sim', choices=sorted(POS_SETS))
    ap.add_argument('--members', default='kz', choices=sorted(MEMBER_SETS))
    ap.add_argument('--ckpts', nargs=5, default=None, help='explicit 5 member checkpoints (overrides --members)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--interval', type=int, default=1, help='1 = protocol (all frames); >1 only for smoke tests')
    ap.add_argument('--out-root', default=RC.OUT_ROOT)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    if not RC.under_out_root(a.out_root):
        raise SystemExit('refusing to write outside %s' % RC.OUT_ROOT)
    if a.interval < 1:
        raise SystemExit('--interval must be >= 1')
    byte = RC.check_byte_copies()
    P = RC.tag_paths(a.tag, a.out_root)
    os.makedirs(P['dir'], exist_ok=True)
    if os.path.exists(P['npz']) and not a.force:
        raise SystemExit('%s exists = finished dump (use --force to redo)' % P['npz'])
    for p in (P['npz'], P['json'], P['sealed'], P['pkl']):    # unfinished leftovers of a crashed run are redone from scratch
        if os.path.exists(p):
            os.remove(p)
    ckpts = list(a.ckpts) if a.ckpts else list(MEMBER_SETS[a.members])
    pos = POS_SETS[a.pos]
    need = ckpts + (pos if pos is not None else RC.RAMP20)
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        raise SystemExit('missing checkpoints: %s' % missing)

    sys.argv = build_argv(P['pkl'], ckpts, a.split, pos)
    import kp_zoom                          # noqa: E402  (same module object dump_real_preds imports)
    import dump_real_preds as DRP           # noqa: E402  (chdir to tools at import, like the original runs)
    from uavdet3d.utils import pose_eval    # noqa: E402
    saved = {'hook': kp_zoom.install_zoom_kp2d_hook, 'bd': DRP.build_dataloader,
             'summ': pose_eval.summarize, 'fmt': pose_eval.format_summary}
    kp_zoom.install_zoom_kp2d_hook = DKM.make_hook(NAMES, n_rgb=len(NAMES))
    calls = {'n': 0}

    def bd(dataset_cfg, *args, **kw):          # main loader = first call (dump_real_preds line 66); calib / KZ loaders untouched
        if calls['n'] == 0:
            assert str(dataset_cfg.DATA_SPLIT['test']) == a.split, dict(dataset_cfg.DATA_SPLIT)
            dataset_cfg.SAMPLED_INTERVAL['test'] = int(a.interval)
        calls['n'] += 1
        return saved['bd'](dataset_cfg, *args, **kw)
    DRP.build_dataloader = bd
    suppress = a.split != 'test'
    if suppress:
        pose_eval.summarize = lambda *x, **k: {}
        pose_eval.format_summary = lambda tag, s: '[repl] %s: ground-truth summary suppressed (split=%s)' % (tag, a.split)

        def masked_print(*args, **kw):
            s = ' '.join(str(x) for x in args)
            s = re.sub(r'检出 [0-9.]+%', '检出 <masked>', s)
            builtins.print(s, **kw)
        DRP.print = masked_print
    print('[repl] cwd %s; split %s; interval %d; pos %s; members %s; argv %s' % (
        os.getcwd(), a.split, a.interval, a.pos, 'explicit' if a.ckpts else a.members, ' '.join(sys.argv[1:])), flush=True)
    DKM.REC['t0'] = time.time()
    ok = False
    n_sealed = None
    try:
        DRP.main()
        ok = True
    finally:
        kp_zoom.install_zoom_kp2d_hook = saved['hook']
        DRP.build_dataloader = saved['bd']
        pose_eval.summarize, pose_eval.format_summary = saved['summ'], saved['fmt']
        if 'print' in DRP.__dict__:
            del DRP.print
        if ok and os.path.exists(P['pkl']):
            n_sealed = RC.seal_gt(P['pkl'], P['sealed'], a.split, a.interval)
        if ok:          # members npz only for a finished dump (its existence is the queue's success marker)
            DKM.save_records(P['dir'], a.tag, rec=DKM.REC, extra_info={
                'members': [{'name': n, 'path': p, 'role': ('explicit' if a.ckpts else a.members)} for n, p in zip(NAMES, ckpts)],
                'arm': 'explicit' if a.ckpts else a.members, 'pos_set': a.pos,
                'pos_ckpts': pos if pos is not None else RC.RAMP20,
                'dump_pkl': os.path.basename(P['pkl']), 'dump_argv': sys.argv[1:], 'dump_finished_ok': ok, 'split': a.split,
                'sampled_interval': int(a.interval), 'gt_sealed_frames': n_sealed, 'gt_summary_suppressed': suppress,
                'sealed_gt': os.path.basename(P['sealed']),
                'sealed_gt_sha256': RC.sha256(P['sealed']) if os.path.exists(P['sealed']) else None,
                'dump_pkl_sha256_labelfree': RC.sha256(P['pkl']) if os.path.exists(P['pkl']) else None,
                'checkpoints_sha256': {os.path.basename(p): RC.sha256(p) for p in ckpts},
                'wrapper': os.path.basename(__file__), 'wrapper_sha256': RC.sha256(os.path.abspath(__file__)),
                'byte_copies_sha256': byte,
                'mirrors': 'e1sa / kzst1 member dumps (run_kz_0923 dump kpKZavg_zoom) with --split %s, position set %s' % (a.split, a.pos)})
    print('[repl] done: %s (gt sealed in %s: %d frames)' % (P['pkl'], os.path.basename(P['sealed']), n_sealed or 0), flush=True)
    return P


if __name__ == '__main__':
    main()
