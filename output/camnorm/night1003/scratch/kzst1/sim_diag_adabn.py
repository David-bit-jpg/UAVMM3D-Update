# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 1, S7 (GPU, ~46 s per model): simulation-forgetting guardrail for the AdaBN members.

Same measurement as the logged kz_s{k} baselines (output/camnorm/kz/kp_diag_KZs{k}.log, run_kz_0923.py line 115):
tools/annot_audit/kp_arm_diag.run(sim_full_r34_kp_w12_zoom.yaml, ckpt, parts='rot', interval=4) on the SIM test split,
heading fold-180 median over detected targets. Guardrail (prereg): AdaBN_k <= KZs_k + 1.5 deg for every k.
Baseline = the original kz_s{k} re-measured in the same run when --with-orig, else the logged 7.3/7.4/7.1/8.0/7.8.

    cd @@ROOT@@\\tools
    python <scratchpad>\\kzst1\\sim_diag_adabn.py [--with-orig]
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ('@@ROOT@@/tools/annot_audit', '@@ROOT@@/tools', '@@ROOT@@', HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:      # noqa: BLE001
    pass
import argparse      # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

import kzst_common as C   # noqa: E402

LOGGED = [7.3, 7.4, 7.1, 8.0, 7.8]


def measure(KD, ck, interval):
    o = KD.run(C.SIM_KP_CFG, ck, 'rot', interval)
    he = o['he']
    return {'n': int(len(he)), 'heading_fold180_median': float(np.median(np.minimum(he, 180 - he))),
            'heading_unfolded_median': float(np.median(he)), 'tilt_median': float(np.median(o['tilt'])),
            'gt150_frac': float((he > 150).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(HERE, 'adabn'))
    ap.add_argument('--tag', default='adabn')
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--interval', type=int, default=4)
    ap.add_argument('--with-orig', action='store_true', help='also re-measure the 5 original kz_s{k} (+4 min)')
    a = ap.parse_args(argv)
    pre, _ = C.check_prereg(a.prereg)
    import kp_arm_diag as KD          # chdir to tools at import
    t0 = time.time()
    res = {'prereg_sha256': pre, 'cfg': C.SIM_KP_CFG, 'parts': 'rot', 'interval': a.interval, 'logged_baseline': LOGGED,
           'adabn': {}, 'orig': {}}
    for k in range(5):
        ck = os.path.join(a.dir, 'kz_s%d_%s.pth' % (k, a.tag))
        res['adabn'][k] = dict(measure(KD, ck, a.interval), ckpt=ck, sha256=C.sha256(ck))
        print('[sim] AdaBN s%d: %s' % (k, res['adabn'][k]), flush=True)
        if a.with_orig:
            res['orig'][k] = dict(measure(KD, C.RELEASE_KZ[k], a.interval), ckpt=C.RELEASE_KZ[k])
            print('[sim] orig  s%d: %s' % (k, res['orig'][k]), flush=True)
    base = [res['orig'][k]['heading_fold180_median'] if k in res['orig'] else LOGGED[k] for k in range(5)]
    res['baseline_used'] = base
    res['baseline_source'] = 're-measured' if a.with_orig else 'logged kp_diag_KZs{k}.log'
    res['guardrail_sim_forgetting'] = {k: bool(res['adabn'][k]['heading_fold180_median'] <= base[k] + 1.5) for k in range(5)}
    res['pass'] = bool(all(res['guardrail_sim_forgetting'].values()))
    res['elapsed_s'] = time.time() - t0
    C.jdump(res, os.path.join(a.dir, 'sim_diag_%s.json' % a.tag))
    print('[sim] guardrail (AdaBN_k <= base_k + 1.5): %s -> %s  (%.0f s)' % (res['guardrail_sim_forgetting'], res['pass'], time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
