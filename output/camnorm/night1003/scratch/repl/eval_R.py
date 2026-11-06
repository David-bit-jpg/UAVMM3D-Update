# -*- coding: utf-8 -*-
"""Experiment R (CPU): pre-registered replication of "AdaBN + SA-TM beats KZ + SA-TM" on the MAV6D VAL split.

  freeze    LABEL-FREE. Validity checks (positions / crops identical between the two member files, replay check, AdaBN
            checkpoints == the stage-1 ones, deployed-decoder replication on the KZ dump), then the per-frame SA-TM / C0
            rotations of both arms -> frozen_R.npz / .json (sha256).
  evaluate  reads the sealed val ground truth (first and only time): primary criteria, guardrails, verdict, report-only
            numbers -> results_R.json, RESULTS_R.md
  all       freeze then evaluate (the queue command)

Everything that decides the verdict is R_SPEC below; make_prereg_R.py writes it (and this file's sha256) into
prereg_R.json before any val ground truth exists on disk; evaluate refuses to run if R_SPEC or this file changed.

    python eval_R.py all                       (real run: val_kz + val_adabn under the output root)
    python eval_R.py all --fixture test        (self-test on the stage-1 TEST files: must reproduce stage 1 exactly)
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import argparse      # noqa: E402
import json          # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import repl_common as RC   # noqa: E402

C, KFO = RC.C, RC.KFO
PREREG = os.path.join(HERE, 'prereg_R.json')

R_SPEC = {
    'version': 'R-1 (2026-10-03)',
    'question': 'Does AdaBN KZ (BN statistics re-estimated on real MAV6D TRAIN crops; the exact stage-1 checkpoints) + SA-TM '
                'beat the original KZ + SA-TM on the MAV6D VAL split?',
    'split': 'val', 'expected_frames': 2336, 'expected_sequences': 16, 'sampled_interval': 1,
    'positions': 'pure-sim release_20260920 ramp_s0..s4 5-ensemble + flip TTA + causal100 temporal background + label-free '
                 'self-calibrated lwh (val images, conf >= 0.7, interval 2) + KZ zoom 95 crops; one GPU dump (dump_members_repl.py '
                 '--split val --pos sim --members kz); the AdaBN members are forwarded on the SAME recorded crops '
                 '(replay_members_repl.py), so positions / detections / crops are identical by construction (asserted)',
    'arms': {'KZ': 'release_20260923/kz_s0..s4.pth, SA-TM of the 5 members at the recorded deployed read cell '
                   '(kz_fuse_offline.fuse_subset SATM, deployed decoder, fallback = dumped rotation)',
             'BN': 'kzst1/adabn/kz_s{0..4}_adabn.pth (sha256 = stage-1 json), read cell = argmax of the 5-member mean hm in the '
                   'recorded 5x5 window, same SA-TM code'},
    'frames': 'all val frames; metric frames = greedy 1 m matched frames of dump_real_preds (identical in both arms); the top '
              'detection carries the arm rotation when it is the hook cell, else the frame is unchanged (stage-1 rule)',
    'heading': 'fold-180 heading error min(a, 180 - a), a = |atan2(E[1,0], E[0,0])|, E = R_gt^T R_pred '
               '(annot_audit/summarize_dumps.stats); statistic = median over matched frames',
    'primary': {'delta_max_deg': -1.0,
                'delta': 'median heading(BN) - median heading(KZ) over all matched val frames',
                'ci_upper_lt': 0.0,
                'ci': 'sequence-cluster bootstrap (kz_fuse_offline.boot_draws / cluster_boot: resample the val sequences with '
                      'replacement, statistic = delta on the pooled frames), B = 10000, seed 20261003; upper bound = 97.5th percentile',
                'seq_better_frac_min': 0.60,
                'seq_rule': 'per val sequence with >= 1 matched frame: median(BN) - median(KZ) < 0 counts as improved (ties = not '
                            'improved); fraction = improved / sequences with >= 1 matched frame'},
    'guardrails': {'tilt_delta_max_deg': 0.3,
                   'tilt': 'median tilt (angle between body z axes, summarize_dumps.stats) over matched frames, BN - KZ <= +0.3 deg',
                   'ads_indoor_delta_min': 0.0,
                   'ads': 'repository LAA3D_ADS, profile indoor (tools/eval_camnorm.ads_eval, as KZ_SUMMARY), value parsed from the '
                          'report (2 decimals); BN - KZ >= 0.00 ("not worse")'},
    'B': 10000, 'seed': 20261003,
    'decision': {'replicated': 'all three primary criteria AND both guardrails pass',
                 'primary_only': 'all primary criteria pass but a guardrail fails -> not replicated as a deployable gain',
                 'not_replicated': 'any primary criterion fails'},
    'validity_checks_before_ground_truth': [
        'byte copies of kz_fuse_offline / kzst_common / dump_kz_members / adabn_kz unchanged (sha256)',
        'KZ dump: split val, interval 1, pos_set sim, members = release KZ, finished, 2336 frames = all val frames',
        'BN file: arm adabn, replay check passed, crops copied from the KZ npz (sha256), checkpoints sha256 == stage-1 AdaBN json',
        'positions / crops / geometry fields identical between the two member files; same top detection per frame',
        'deployed decoder replicated on the KZ dump: top conf == hook peak on >= 99 % of detected frames (others stay '
        'unchanged in both arms), PnP(recorded vector) == dumped rotation and offline C0 == dumped rotation on >= 99.9 % '
        'of hook frames'],
    'report_only': ['per class (phantom4 / mavic2) heading, <10 / <20 deg, tilt, 10deg10cm, ADD(0.5d)', 'C0 (plain mean) arms',
                    'ADS laa', 'one-sided bootstrap p', 'per-sequence deltas', 'unfolded flip rate (> 90 deg)',
                    'E5 strata of the KZ arm', 'comparison with the stage-1 TEST effect (-3.54 deg)'],
}

FIXTURE = {'test': {'kz_npz': RC.E1_NPZ, 'bn_npz': RC.S1_BN_NPZ, 'dump': RC.E1_DUMP, 'sealed': None, 'split': 'test',
                    'expected_frames': 4800,
                    'reproduce': {'KZ_SATM_head': 12.596766405655806, 'BN_SATM_head': 9.058097097458393,
                                  'KZ_SATM_tilt': 8.523623462590926, 'BN_SATM_tilt': 7.739088698857728,
                                  'ads_indoor': {'KZ_SATM': 50.88, 'BN_SATM': 52.9},
                                  'single_level_ci95': [-6.883224116152282, -1.3436293176645697]}}}


def paths(a):
    if a.fixture:
        F = FIXTURE[a.fixture]
        d = a.out_dir or os.path.join(RC.OUT_ROOT, 'selftest_R_%s' % a.fixture)
        kz_json = os.path.splitext(F['kz_npz'])[0] + '.json'
        return {'kz_npz': F['kz_npz'], 'kz_json': kz_json, 'bn_npz': F['bn_npz'], 'bn_json': os.path.splitext(F['bn_npz'])[0] + '.json',
                'dump': F['dump'], 'sealed': F['sealed'], 'dir': d}       # fixture inputs: scratchpad; outputs: output root
    kz, bn = RC.tag_paths(a.kz_tag), RC.tag_paths(a.bn_tag)
    return {'kz_npz': kz['npz'], 'kz_json': kz['json'], 'bn_npz': bn['npz'], 'bn_json': bn['json'], 'dump': kz['pkl'],
            'sealed': kz['sealed'], 'dir': a.out_dir or os.path.join(RC.OUT_ROOT, 'eval_R')}


def out_paths(d):
    return {'frozen': os.path.join(d, 'frozen_R.npz'), 'frozen_json': os.path.join(d, 'frozen_R.json'),
            'results': os.path.join(d, 'results_R.json'), 'report': os.path.join(d, 'RESULTS_R.md'), 'ads': os.path.join(d, 'ads')}


def prereg_or_none(a):
    if a.fixture or getattr(a, 'smoke', False):
        return (RC.sha256(PREREG) if os.path.exists(PREREG) else None), None
    pre, P = RC.check_prereg(a.prereg)
    if P.get('spec') != json.loads(json.dumps(R_SPEC)):
        raise SystemExit('R_SPEC in eval_R.py differs from the pre-registered spec')
    if P['code_sha256'].get('eval_R.py') != RC.sha256(os.path.abspath(__file__)):
        raise SystemExit('eval_R.py changed after the pre-registration')
    if P['code_sha256'].get('repl_common.py') != RC.sha256(os.path.join(HERE, 'repl_common.py')):
        raise SystemExit('repl_common.py changed after the pre-registration')
    return pre, P


# ======================================================================================= freeze (label-free)
def validity(a, Pt, Dk, ik, Db, ib, frames, log):
    v = {'byte_copies': RC.check_byte_copies()}
    if not a.smoke:
        exp_n = FIXTURE[a.fixture]['expected_frames'] if a.fixture else R_SPEC['expected_frames']
        assert len(Dk['seq']) == exp_n == len(frames), ('frame count', len(Dk['seq']), exp_n, len(frames))
    assert ik.get('dump_finished_ok') is True, 'KZ dump not finished'
    rel = [os.path.normcase(os.path.abspath(p)) for p in RC.RELEASE_KZ]
    kz_paths = {m['name']: os.path.normcase(os.path.abspath(m['path'])) for m in ik['members']}
    assert [kz_paths['KZs%d' % k] for k in range(5)] == rel, 'KZ dump members are not the release KZ'
    if not a.fixture:
        assert ik.get('split') == R_SPEC['split'], ('split', ik.get('split'))
        assert ik.get('pos_set') == 'sim' and ik.get('arm') == 'kz', ('pos_set / arm', ik.get('pos_set'), ik.get('arm'))
        assert ik.get('sealed_gt_sha256') == RC.sha256(Pt['sealed']), 'sealed ground truth file changed since the dump'
        assert ik.get('dump_pkl_sha256_labelfree') == RC.sha256(Pt['dump']), 'label-free dump pkl changed since the dump'
        assert os.path.normcase(os.path.abspath(ib['positions_crops_from'])) == os.path.normcase(os.path.abspath(Pt['kz_npz']))
        assert ib['positions_crops_from_sha256'] == RC.sha256(Pt['kz_npz']), 'BN crops were not replayed from this KZ npz'
        assert ib.get('replay_finished_ok') is True
        if not a.smoke:
            assert int(ik.get('sampled_interval', -1)) == R_SPEC['sampled_interval'], 'interval'
            assert len({str(s) for s in Dk['seq']}) == R_SPEC['expected_sequences'], 'val sequence count'
            assert ib.get('smoke') is False
    assert ib.get('arm') == 'adabn', ib.get('arm')
    rc = ib['replay_check']               # repl format {.., 'pass'}; stage-1 format {'train': {..}, 'test': {..}}
    assert (rc['pass'] is True) if 'pass' in rc else all(x['pass'] is True for x in rc.values()), 'replay check did not pass'
    s1 = json.load(open(RC.ADABN_STAGE1_JSON, encoding='utf-8'))['checkpoints_sha256']
    assert ib['checkpoints_sha256'] == s1, 'BN checkpoints are not the stage-1 AdaBN checkpoints'
    v['bn_checkpoints_sha256'] = s1
    v['bn_replay_check'] = ib['replay_check']
    RC.assert_same_positions(Dk, Db)
    v['positions_identical'] = True
    v['labelfree_kz'] = RC.labelfree_checks(Dk, ik, frames, log)
    if not v['labelfree_kz']['pass']:
        raise SystemExit('deployed decoder not replicated on the KZ dump -- stop before reading ground truth')
    return v


def stage_freeze(a):
    Pt = paths(a)
    O = out_paths(Pt['dir'])
    os.makedirs(Pt['dir'], exist_ok=True)
    if not RC.under_out_root(Pt['dir']):
        raise SystemExit('refusing to write outside %s' % RC.OUT_ROOT)
    pre, _ = prereg_or_none(a)
    t0 = time.time()
    log = []
    Dk, ik = C.load_npz_info(Pt['kz_npz'])
    Db, ib = C.load_npz_info(Pt['bn_npz'])
    frames, dinfo = RC.load_frames(Pt['dump'], with_gt=False)
    if not a.fixture:          # the dump itself: pure-sim position ensemble, val split, KZ zoom 95, label-free size calibration
        nc = lambda ps: [os.path.normcase(os.path.abspath(p)) for p in ps]       # noqa: E731
        assert nc(dinfo['ckpts']) == nc(RC.RAMP20), ('position ensemble', dinfo['ckpts'])
        assert dinfo['split'] == 'val' and float(dinfo['kp2d_zoom']) == 95.0 and dinfo['size_source'] == 'calib', dinfo
    v = validity(a, Pt, Dk, ik, Db, ib, frames, log)
    keys, A = RC.compute_arms(Dk, ik, frames)                       # KZ members (by name in the KZ npz)
    keys_b, B = RC.compute_arms(Db, ib, frames, members=list(range(5)))
    assert keys_b == keys and np.array_equal(A['top'], B['top']), 'top detections differ between arms'
    Rdep = RC.deployed_rot(frames, keys, A['top'])
    hook = A['top'] >= 0
    d_c0 = np.array([KFO.geodesic(A['C0'][i], Rdep[i]) for i in np.where(hook)[0]])
    v['kz_C0_vs_dumped_frac_lt_0p05'] = float((d_c0 < 0.05).mean())
    cls = np.array([frames[k]['cls'] for k in keys])
    seq = np.array([k[0] for k in keys])
    out = {'seq': seq, 'frame': np.array([k[1] for k in keys]), 'cls': cls, 'top': A['top'], 'Rdep': Rdep,
           'KZ_SATM': A['SATM'], 'KZ_C0': A['C0'], 'BN_SATM': B['SATM'], 'BN_C0': B['C0'],
           'KZ_stratum': A['stratum'], 'BN_stratum': B['stratum'], 'KZ_fb': A['fb_SATM'], 'BN_fb': B['fb_SATM']}
    with open(O['frozen'] + '.tmp', 'wb') as f:
        np.savez(f, **out)
    os.replace(O['frozen'] + '.tmp', O['frozen'])
    strata = {arm: {s: float((o['stratum'][hook] == s).mean()) for s in C.STRATA} for arm, o in (('KZ', A), ('BN', B))}
    changed = float(np.mean([KFO.fold(KFO.rel_yaw(A['SATM'][i], B['SATM'][i])) > 5 for i in np.where(hook)[0]]))
    J = {'prereg_sha256': pre, 'fixture': a.fixture, 'frozen_sha256': RC.sha256(O['frozen']), 'kz_npz_sha256': RC.sha256(Pt['kz_npz']),
         'bn_npz_sha256': RC.sha256(Pt['bn_npz']), 'dump_sha256': RC.sha256(Pt['dump']), 'n_frames': len(keys),
         'n_hook_frames': int(hook.sum()), 'validity': v, 'strata_hook_frames': strata,
         'labelfree_BN_vs_KZ_satm_fold_gt5_frac': changed, 'calib_lwh': dinfo.get('calib_lwh'),
         'script_sha256': RC.sha256(os.path.abspath(__file__)), 'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
         'elapsed_s': time.time() - t0, 'log': log}
    RC.jdump(J, O['frozen_json'])
    RC.say('[freeze] %d frames (%d hook), validity passed, KZ C0 == dumped rotation on %.4f; BN vs KZ SA-TM fold > 5 deg on %.3f '
           '(label-free); wrote %s (%.0f s)' % (len(keys), hook.sum(), v['kz_C0_vs_dumped_frac_lt_0p05'], changed, O['frozen'], time.time() - t0))
    return J


# ======================================================================================= evaluate (reads the sealed ground truth)
def stage_evaluate(a):
    Pt = paths(a)
    O = out_paths(Pt['dir'])
    pre, PR = prereg_or_none(a)
    FJ = json.load(open(O['frozen_json'], encoding='utf-8'))
    if FJ['frozen_sha256'] != RC.sha256(O['frozen']) or FJ['prereg_sha256'] != pre:
        raise SystemExit('frozen_R hash / prereg mismatch')
    if FJ['kz_npz_sha256'] != RC.sha256(Pt['kz_npz']) or FJ['bn_npz_sha256'] != RC.sha256(Pt['bn_npz']):
        raise SystemExit('member files changed after the freeze')
    Z = dict(np.load(O['frozen'], allow_pickle=False))
    keys = [(str(s), str(f)) for s, f in zip(Z['seq'], Z['frame'])]
    top, cls, seqs = Z['top'], Z['cls'], Z['seq']
    _, ik = C.load_npz_info(Pt['kz_npz'])
    if not a.fixture and ik.get('sealed_gt_sha256') != RC.sha256(Pt['sealed']):
        raise SystemExit('sealed ground truth file is not the one written by the val dump')
    t_read = time.strftime('%Y-%m-%d %H:%M:%S')
    frames, _ = RC.load_frames(Pt['dump'], Pt['sealed'], with_gt=True)            # ---- ground truth from here on ----
    log = []
    R = {'prereg_sha256': pre, 'fixture': a.fixture, 'frozen_sha256': FJ['frozen_sha256'], 'gt_first_read_at': t_read,
         'spec': R_SPEC, 'script_sha256': RC.sha256(os.path.abspath(__file__))}
    arms = ['KZ_SATM', 'BN_SATM', 'KZ_C0', 'BN_C0']
    Fs, Ms, T = {}, {}, {}
    for an in arms:
        Fs[an], Ms[an], T[an] = RC.arm_metrics(frames, keys, Z[an], top, cls)
    hit = np.isfinite(Ms['KZ_SATM']['he'])
    for an in arms:
        assert np.array_equal(np.isfinite(Ms[an]['he']), hit) and np.array_equal(Ms[an]['pos'], Ms['KZ_SATM']['pos']), \
            'positions / detections differ between arms (%s)' % an
    # the deployed rotation (dump as is) == KZ C0 arm (report: max heading difference over matched frames)
    Fdep, Mdep, Tdep = RC.arm_metrics(frames, keys, Z['Rdep'], top, cls)
    R['kz_C0_vs_dumped_heading_absdiff_max'] = float(np.nanmax(np.abs(Mdep['he'][hit] - Ms['KZ_C0']['he'][hit]))) if hit.any() else None
    R['table'] = T
    R['table_dumped_rotation'] = Tdep
    he_b, he_k = Ms['BN_SATM']['he'][hit], Ms['KZ_SATM']['he'][hit]
    sh = seqs[hit]
    bs = RC.seq_cluster_ci(he_b, he_k, sh, R_SPEC['B'], R_SPEC['seed'])
    frac = bs['n_seq_better'] / max(bs['n_seq'], 1)
    tilt_d = T['BN_SATM']['all']['tilt'] - T['KZ_SATM']['all']['tilt']
    ads = {}
    if not a.skip_ads:
        for an in ('KZ_SATM', 'BN_SATM', 'KZ_C0', 'BN_C0'):
            ads[an] = RC.ads(Fs[an], ik['W'], ik['H'], O['ads'], an, ('indoor', 'laa'))
    R['ads'] = ads
    ads_d = (ads['BN_SATM']['indoor'] - ads['KZ_SATM']['indoor']) if (ads.get('BN_SATM', {}).get('indoor') is not None and
                                                                     ads.get('KZ_SATM', {}).get('indoor') is not None) else None
    pr = R_SPEC['primary']
    gr = R_SPEC['guardrails']
    crit = {'delta_le_-1': bool(bs['delta'] <= pr['delta_max_deg']), 'ci_upper_lt_0': bool(bs['ci95'][1] < pr['ci_upper_lt']),
            'seq_better_ge_60pct': bool(frac >= pr['seq_better_frac_min'])}
    guard = {'tilt': {'BN': T['BN_SATM']['all']['tilt'], 'KZ': T['KZ_SATM']['all']['tilt'], 'delta': tilt_d,
                      'max': gr['tilt_delta_max_deg'], 'pass': bool(tilt_d <= gr['tilt_delta_max_deg'])},
             'ads_indoor': {'BN': ads.get('BN_SATM', {}).get('indoor'), 'KZ': ads.get('KZ_SATM', {}).get('indoor'), 'delta': ads_d,
                            'min': gr['ads_indoor_delta_min'], 'pass': bool(ads_d is not None and ads_d >= gr['ads_indoor_delta_min'] - 1e-9)}}
    if all(crit.values()) and all(g['pass'] for g in guard.values()):
        verdict = 'replicated'
    elif all(crit.values()):
        verdict = 'primary_only'
    else:
        verdict = 'not_replicated'
    R['primary'] = {'KZ_head': T['KZ_SATM']['all']['head'], 'BN_head': T['BN_SATM']['all']['head'], 'delta': bs['delta'],
                    'ci95': bs['ci95'], 'p_one_sided_ge0': bs['p_one_sided_ge0'], 'n_seq': bs['n_seq'], 'n_seq_better': bs['n_seq_better'],
                    'seq_better_frac': frac, 'per_seq_delta': bs['per_seq_delta'], 'per_seq_n_matched': bs['per_seq_n'],
                    'criteria': crit}
    R['guardrails'] = guard
    R['verdict'] = verdict
    # report-only
    rep = {'C0_arms': {'KZ_C0_head': T['KZ_C0']['all']['head'], 'BN_C0_head': T['BN_C0']['all']['head']},
           'per_class_delta_SATM': {c: T['BN_SATM'][c]['head'] - T['KZ_SATM'][c]['head'] for c in C.CLASSES if c in T['BN_SATM']},
           'kz_strata_matched': {s: int(((Z['KZ_stratum'] == s) & hit).sum()) for s in C.STRATA},
           'bn_strata_matched': {s: int(((Z['BN_stratum'] == s) & hit).sum()) for s in C.STRATA},
           'stage1_test_reference': {'KZ_SATM': 12.60, 'BN_SATM': 9.06, 'delta': -3.54, 'single_level_ci95': [-6.88, -1.34],
                                     'seq_better': '9/10', 'tilt': [8.52, 7.74], 'ads_indoor': [50.88, 52.9]}}
    strata_d = {}
    for s in ('agree', 'flip_only', 'trim_only', 'both'):
        m = hit & (Z['KZ_stratum'] == s)
        if m.any():
            strata_d[s] = {'n': int(m.sum()), 'KZ': float(np.median(Ms['KZ_SATM']['he'][m])), 'BN': float(np.median(Ms['BN_SATM']['he'][m]))}
    rep['by_kz_stratum'] = strata_d
    R['report'] = rep
    if a.fixture:
        ref = FIXTURE[a.fixture]['reproduce']
        assert abs(T['KZ_SATM']['all']['head'] - ref['KZ_SATM_head']) < 1e-9, ('fixture KZ head', T['KZ_SATM']['all']['head'])
        assert abs(T['BN_SATM']['all']['head'] - ref['BN_SATM_head']) < 1e-9, ('fixture BN head', T['BN_SATM']['all']['head'])
        assert abs(T['KZ_SATM']['all']['tilt'] - ref['KZ_SATM_tilt']) < 1e-9 and abs(T['BN_SATM']['all']['tilt'] - ref['BN_SATM_tilt']) < 1e-9
        assert np.allclose(bs['ci95'], ref['single_level_ci95'], atol=1e-9), ('fixture CI', bs['ci95'])
        if not a.skip_ads:
            for an, val in ref['ads_indoor'].items():
                assert ads[an]['indoor'] is not None and abs(ads[an]['indoor'] - val) < 0.01, ('fixture ADS', an, ads[an])
        R['fixture_reproduced_stage1'] = True
        RC.say('[evaluate] fixture %s: stage-1 numbers reproduced exactly (heads, tilts, single-level CI%s)' % (
            a.fixture, '' if a.skip_ads else ', ADS indoor'), log)
    RC.say('[evaluate] KZ %.2f | BN %.2f | delta %+.2f, seq-cluster 95%% [%+.2f, %+.2f], seq better %d/%d (%.0f%%) | tilt %+.2f | '
           'ADS indoor %s -> %s' % (T['KZ_SATM']['all']['head'], T['BN_SATM']['all']['head'], bs['delta'], bs['ci95'][0], bs['ci95'][1],
                                    bs['n_seq_better'], bs['n_seq'], 100 * frac, tilt_d,
                                    'n/a' if ads_d is None else '%+.2f' % ads_d, verdict), log)
    R['log'] = log
    RC.jdump(R, O['results'])
    write_report(R, O['report'])
    RC.say('[evaluate] wrote %s and %s' % (O['results'], O['report']))
    return R


def write_report(R, path):
    T, p, g = R['table'], R['primary'], R['guardrails']
    head = '实验 R：AdaBN + SA-TM 在 MAV6D val 上的预注册复现' + ('（自测夹具 %s）' % R['fixture'] if R['fixture'] else '')
    L = ['# ' + head, '', 'prereg sha256 `%s`；frozen `%s`；首次读真值 %s' % (R['prereg_sha256'], R['frozen_sha256'], R['gt_first_read_at']), '',
         '| 臂 | 航向折180 全部 | phantom4 | mavic2 | <10° | <20° | 倾斜 | 10°10cm | ADD(0.5d) | 检出 | 位置中位 |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for an, t in T.items():
        L.append('| %s | %.2f° | %s | %s | %.1f%% | %.1f%% | %.2f° | %.2f%% | %.2f%% | %.1f%% | %.3f m |' % (
            an, t['all']['head'], '%.2f°' % t['phantom4']['head'] if 'phantom4' in t else '-', '%.2f°' % t['mavic2']['head'] if 'mavic2' in t else '-',
            t['all']['head_lt10'], t['all']['head_lt20'], t['all']['tilt'], t['all']['acc_10deg10cm'], t['all']['add_0p5d'], t['all']['det'],
            t['all']['pos_median']))
    L += ['', '## 主判据（预注册）', '',
          'Δ = BN − KZ（都用 SA-TM）= %+.2f°（判据 ≤ −1.0 → %s）；序列聚类 bootstrap 95%% [%+.2f, %+.2f]（上界 < 0 → %s，单侧 p %.4f）；'
          '变好序列 %d/%d = %.0f%%（≥ 60%% → %s）' % (
              p['delta'], '过' if p['criteria']['delta_le_-1'] else '不过', p['ci95'][0], p['ci95'][1],
              '过' if p['criteria']['ci_upper_lt_0'] else '不过', p['p_one_sided_ge0'], p['n_seq_better'], p['n_seq'], 100 * p['seq_better_frac'],
              '过' if p['criteria']['seq_better_ge_60pct'] else '不过'),
          '', '护栏：倾斜 BN %.2f° vs KZ %.2f°（Δ %+.2f，上限 +0.3 → %s）；ADS indoor BN %s vs KZ %s（Δ %s，不许变差 → %s）' % (
              g['tilt']['BN'], g['tilt']['KZ'], g['tilt']['delta'], '过' if g['tilt']['pass'] else '不过',
              g['ads_indoor']['BN'], g['ads_indoor']['KZ'], 'n/a' if g['ads_indoor']['delta'] is None else '%+.2f' % g['ads_indoor']['delta'],
              '过' if g['ads_indoor']['pass'] else '不过'),
          '', '**结论：%s**' % {'replicated': '复现成立（主判据与护栏全过）', 'primary_only': '主判据过、护栏不过（不算可部署的复现）',
                               'not_replicated': '未复现'}[R['verdict']],
          '', '逐序列 Δ（匹配帧数）：' + '；'.join('%s %+.2f° (%d)' % (s, d, p['per_seq_n_matched'][s]) for s, d in p['per_seq_delta'].items()),
          '', 'ADS（indoor / laa）：' + json.dumps(R['ads'], ensure_ascii=False),
          '', '只报告：' + json.dumps(R['report'], ensure_ascii=False)]
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['freeze', 'evaluate', 'all'])
    ap.add_argument('--kz-tag', default='val_kz')
    ap.add_argument('--bn-tag', default='val_adabn')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--prereg', default=PREREG)
    ap.add_argument('--fixture', default=None, choices=sorted(FIXTURE), help='self-test on the stage-1 TEST files (no prereg needed)')
    ap.add_argument('--skip-ads', action='store_true', help='self-tests only')
    ap.add_argument('--smoke', action='store_true', help='CPU smoke dumps of a few val frames: FREEZE ONLY (label-free); '
                                                         'frame-count / interval / sequence-count checks relaxed')
    a = ap.parse_args(argv)
    if a.skip_ads and not a.fixture:
        raise SystemExit('--skip-ads is for self-tests only (ADS indoor is a pre-registered guardrail)')
    if a.smoke and (a.stage != 'freeze' or a.fixture or a.out_dir is None or os.path.basename(a.out_dir.rstrip('/\\')) == 'eval_R'):
        raise SystemExit('--smoke: stage freeze only, with its own --out-dir (never reads val ground truth)')
    if a.stage in ('freeze', 'all'):
        stage_freeze(a)
    if a.stage in ('evaluate', 'all'):
        return stage_evaluate(a)


if __name__ == '__main__':
    main()
