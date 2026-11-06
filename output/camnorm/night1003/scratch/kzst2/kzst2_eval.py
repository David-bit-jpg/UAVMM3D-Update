# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 2, S8 (CPU): the LOO arm on the real MAV6D TEST split (4800 frames), all criteria of prereg_stage2.json.

  freeze    LABEL-FREE. Per-frame rotations on test of
              A0      deployed 5-member KZ SA-TM (asserted bit-identical to e1sa/frozen_e1sa.npz)
              A0p     the same 5 members decoded with D_rho (rho_c frozen in stage 1 from TRAIN)
              BN      the 5 stage-1 AdaBN members, deployed decoder, SA-TM (asserted == stage-1 frozen_test BN)
              LOO     the 5 LOO students (kzst2/test/kz_members_loo_test.npz), D_rho decoder, SA-TM      <- the arm
              report-only: LOO_dep (students, deployed decoder), BN_rho (AdaBN, D_rho)
            self-peak singles of every member set, and SA-TM of all 126 seed multisets (5 draws with replacement) for
            kz / kz_rho / bn / loo_rho (two-level bootstrap, seeds paired by index k).  -> eval/frozen_test_stage2.npz/.json
  evaluate  reads the test ground truth: KZ_SUMMARY metrics; A0* (stage-1 rule); A0** = better of A0* and BN;
            P1 primary  LOO vs A0**  (delta <= -1.0, two-level bootstrap upper < 0, >= 7/10 sequences, paired seeds >= 4/5,
                        Holm with P1-BN of stage 1);  P1 secondary  LOO vs A0* (same statistics, report line);
            guardrails (tilt, ADS indoor, positions, mavic2, flip rate, GT view strata, GT-atypical, sim forgetting);
            decision; report-only items.  -> eval/results_stage2.json, eval/RESULTS_stage2.md
    python kzst2_eval.py freeze
    python kzst2_eval.py evaluate
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse      # noqa: E402
import concurrent.futures as cf   # noqa: E402
import json          # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import kzst2_common as S   # noqa: E402
import kzst_stage1_eval as E1   # noqa: E402  (kzst1: arms_for, heading_err_vec, roundtrip, seq_axial_resid, MULTISETS)

C1, KFO = S.C1, S.KFO
MULTISETS, IDENT = E1.MULTISETS, E1.IDENT
CONFIGS = {'kz': ('kz', False), 'kz_rho': ('kz', True), 'bn': ('bn', False), 'loo_rho': ('loo', True)}
E5_REF = dict(E1.E5_REF)
GUARD = dict(E1.GUARD)                     # tilt 8.82, ADS 50.58, mavic2 28.56, flip +3 pp, view +1.0, atypical +1.0 (n>=200)
P1 = dict(E1.P1)                           # delta -1.0, 7/10 sequences, paired 4/5, alpha 0.05, B 10000, seed 20261003
P1['seed'] = 20261004                      # stage 2 draws its own bootstrap stream (frozen here)
SIM_MARGIN = 1.5


def paths(a):
    d = os.path.abspath(a.out_dir)
    return {'frozen': os.path.join(d, 'frozen_test_stage2.npz'), 'frozen_json': os.path.join(d, 'frozen_test_stage2.json'),
            'work': os.path.join(d, '_ms_work_stage2.npz'), 'results': os.path.join(d, 'results_stage2.json'),
            'report': os.path.join(d, 'RESULTS_stage2.md'), 'ads': os.path.join(d, 'ads_stage2'),
            'loo_npz': a.loo_npz or os.path.join(S.TEST_DIR, 'kz_members_loo_test.npz'),
            'bn_npz': a.bn_npz or S.STAGE1['bn_npz'], 'sim_json': a.sim_json or os.path.join(S.SIM_DIR, 'sim_diag_loo.json'),
            'stage1_results': a.stage1_results or S.STAGE1['results'], 'stage1_frozen_test': a.stage1_frozen_test or S.STAGE1['frozen_test'],
            'frozen_train': a.frozen_train or S.STAGE1['frozen_train'], 'train_root': a.train_root or S.TRAIN_DIR,
            'pseudo_json': a.pseudo_json or S.PSEUDO_JSON}


# ======================================================================================= multiset workers
_W = {}


def _init(path):
    _W.clear()
    _W.update(dict(np.load(path, allow_pickle=False)))


def _ms_chunk(args):
    idx, cfgs = args
    W = _W
    search = int(W['search'])
    ks = float(W['kp_scale'])
    out = {c: np.full((len(idx), len(MULTISETS), 3, 3), np.nan) for c in cfgs}
    for n, i in enumerate(idx):
        if not W['has'][i]:
            continue
        for c in cfgs:
            src, use_rho = CONFIGS[c]
            g = KFO.Geo(W['K_in'][i], W['dist'][i], W['raw_size'][i], W['new_size'][i], W['stride_b'][i], W['pos_rc'][i],
                        W['center_res'][i], W['lwh'][i], ks)
            ctx = C1.RhoCtx(W[src + '_hm'][i], W[src + '_kp'][i], W['z'][i], g, W['rho'][i] if use_rho else None)
            rb, cb = W[src + '_rb_cb'][i]
            rc, cc = W['rc_cc'][i]
            rec_cell = (int(rb - (rc - search)), int(cb - (cc - search)))
            for j, ms in enumerate(MULTISETS):
                mem = list(ms)
                cell = rec_cell if j == IDENT else ctx.cell_for(mem)
                o, _, _, _ = KFO.fuse_subset(ctx, mem, cell, ['C0', 'SATM'], W['Rdep'][i])
                out[c][n, j] = o['SATM']
    return idx, out


def same_positions(Dx, D, n):
    for k in ('seq', 'frame', 'pos_rc', 'center_res', 'hm_peak', 'z', 'crop', 'rc_cc', 'win', 'K_in', 'dist', 'raw_size', 'new_size', 'stride_b'):
        if k not in D and k not in Dx:
            continue                      # synthetic fixtures carry no crop records
        assert np.array_equal(Dx[k], D[k][:n]), 'field %s differs from the e1sa test dump (positions / crops must be identical)' % k


# ======================================================================================= freeze (label-free)
def stage_freeze(a):
    P = paths(a)
    pre, _ = S.check_prereg(a.prereg)
    t0 = time.time()
    os.makedirs(os.path.dirname(P['frozen']), exist_ok=True)
    FT = json.load(open(os.path.splitext(P['frozen_train'])[0] + '.json', encoding='utf-8'))
    if FT['frozen_npz_sha256'] != S.sha256(P['frozen_train']):
        raise SystemExit('frozen_train_stage1 hash mismatch')
    if FT['split'] != 'train' and not a.allow_nontrain_rho:
        raise SystemExit('rho must come from the TRAIN split (got %r)' % FT['split'])
    ZT = np.load(P['frozen_train'])
    rho = {str(c): tuple(float(x) for x in r) for c, r in zip(ZT['rho_classes'], ZT['rho'])}
    D, info = C1.load_npz_info(a.test_npz)
    frames, _ = C1.load_dump_nogt(a.test_dump)
    T = C1.frame_table(D, frames)
    N = len(T['seq'])
    log = []
    KFO.say('[freeze2] %d test frames; rho (stage-1 TRAIN, rule-literal) %s' % (N, rho), log)
    A0 = E1.arms_for(D, info, frames, T, None)
    A0p = E1.arms_for(D, info, frames, T, rho)
    real = os.path.normcase(os.path.abspath(a.test_npz)) == os.path.normcase(os.path.abspath(S.TEST_NPZ))
    if os.path.exists(a.test_frozen):
        Z5 = dict(np.load(a.test_frozen, allow_pickle=False))
        arms5 = list(Z5['arm_names'])
        assert np.array_equal(A0['SATM'], Z5['R'][arms5.index('SATM')]), 'A0 != frozen_e1sa SA-TM'
        for k in range(5):
            assert np.array_equal(A0['single'][:, k], Z5['R'][arms5.index('single_KZs%d' % k)])
        e5_strata = Z5['stratum']
        a0_checked = True
    else:
        assert not real, 'frozen_e1sa.npz missing for the real test inputs'
        e5_strata = A0['stratum']
        a0_checked = False
    # ---- LOO students
    Dl, il = C1.load_npz_info(P['loo_npz'])
    if il.get('arm') != 'loo' or (il.get('smoke', False) and not a.allow_smoke):
        raise SystemExit('%s is not a (non-smoke) LOO test dump' % P['loo_npz'])
    if il.get('prereg_sha256') != pre:
        raise SystemExit('LOO test dump was produced under another prereg')
    assert len(Dl['seq']) == N, 'LOO npz does not cover all test frames'
    same_positions(Dl, D, N)
    LOO = E1.arms_for(Dl, il, frames, T, rho, members=list(range(5)))
    LOOd = E1.arms_for(Dl, il, frames, T, None, members=list(range(5)))
    assert np.array_equal(LOO['top'], A0['top'])
    out = {'seq': T['seq'], 'frame': T['frame'], 'cls': T['cls'], 'top': A0['top'], 'e5_stratum': e5_strata,
           'A0': A0['SATM'], 'A0_single': A0['single'], 'A0_member_R': A0['member_R'], 'A0_ratios': A0['ratios'],
           'A0p': A0p['SATM'], 'A0p_single': A0p['single'], 'A0p_member_R': A0p['member_R'],
           'LOO': LOO['SATM'], 'LOO_C0': LOO['C0'], 'LOO_single': LOO['single'], 'LOO_member_R': LOO['member_R'], 'LOO_stratum': LOO['stratum'],
           'LOO_dep': LOOd['SATM'], 'LOO_dep_single': LOOd['single'], 'LOO_dep_member_R': LOOd['member_R'], 'LOO_ratios': LOOd['ratios'],
           'rho': np.array([rho[c] for c in S.CLASSES])}
    J = {'prereg_sha256': pre, 'frozen_train_sha256': FT['frozen_npz_sha256'], 'test_npz_sha256': S.sha256(a.test_npz),
         'test_dump_sha256': S.sha256(a.test_dump), 'loo_npz_sha256': S.sha256(P['loo_npz']), 'loo_info': {k: il.get(k) for k in ('students', 'replay_check', 'smoke')},
         'rho': rho, 'real_test_inputs': real, 'a0_equals_frozen_e1sa_asserted': a0_checked,
         'strata_LOO': {c: {s: float((LOO['stratum'][T['cls'] == c] == s).mean()) for s in C1.STRATA} for c in S.CLASSES},
         'changed_LOO_vs_A0_heading_gt5_or_tilt_gt2': float(np.mean([
             (KFO.fold(KFO.rel_yaw(A0['SATM'][i], LOO['SATM'][i])) > 5 or KFO.tilt_between(A0['SATM'][i], LOO['SATM'][i]) > 2)
             for i in range(N) if A0['top'][i] >= 0]))}
    # ---- BN (stage 1 AdaBN members)
    have_bn = os.path.exists(P['bn_npz'])
    if have_bn:
        Db, ib = C1.load_npz_info(P['bn_npz'])
        if ib.get('arm') != 'adabn' or (ib.get('smoke', False) and not a.allow_smoke):
            raise SystemExit('%s is not a (non-smoke) AdaBN test dump' % P['bn_npz'])
        assert len(Db['seq']) == N
        same_positions(Db, D, N)
        BN = E1.arms_for(Db, ib, frames, T, None, members=list(range(5)))
        BNr = E1.arms_for(Db, ib, frames, T, rho, members=list(range(5)))
        if os.path.exists(P['stage1_frozen_test']):
            Z1 = np.load(P['stage1_frozen_test'])
            if 'BN' in Z1.files:
                assert np.array_equal(BN['SATM'], Z1['BN'], equal_nan=True), 'BN arm != stage-1 frozen_test BN'
                assert np.array_equal(A0p['SATM'], Z1['A0p'], equal_nan=True), "A0' != stage-1 frozen_test A0p"
                J['bn_equals_stage1_frozen'] = True
        out.update({'BN': BN['SATM'], 'BN_single': BN['single'], 'BN_member_R': BN['member_R'], 'BN_rho': BNr['SATM'],
                    'BN_rho_single': BNr['single']})
        J['bn_npz_sha256'] = S.sha256(P['bn_npz'])
    J['have_bn'] = have_bn
    # ---- multisets for the two-level bootstrap (seeds paired by index k across arms)
    rgb = C1.member_index(info)
    lwh = np.full((N, 3), np.nan)
    Rdep = np.full((N, 3, 3), np.nan)
    for i, k in enumerate(T['keys']):
        t = A0['top'][i]
        if t >= 0:
            lwh[i] = frames[k]['pred'][t, 3:6]
            Rdep[i] = KFO.rot_from_eul(frames[k]['pred'][t, 6:9])
    Wk = {'has': A0['top'] >= 0, 'lwh': lwh, 'Rdep': Rdep, 'rho': np.array([rho[c] for c in T['cls']]),
          'search': np.int64(info.get('search', 2)), 'kp_scale': np.float64(info.get('KP_SCALE') or 64.0),
          'kz_hm': D['hm_win'][:, rgb], 'kz_kp': D['kp_win'][:, rgb], 'kz_rb_cb': D['rb_cb'],
          'loo_hm': Dl['hm_win'], 'loo_kp': Dl['kp_win'], 'loo_rb_cb': Dl['rb_cb']}
    if have_bn:
        Wk.update({'bn_hm': Db['hm_win'], 'bn_kp': Db['kp_win'], 'bn_rb_cb': Db['rb_cb']})
    for k in ('K_in', 'dist', 'raw_size', 'new_size', 'stride_b', 'pos_rc', 'center_res', 'z', 'rc_cc'):
        Wk[k] = D[k]
    with open(P['work'], 'wb') as f:
        np.savez(f, **Wk)
    cfgs = [c for c in CONFIGS if c != 'bn' or have_bn]
    chunks = [list(range(j, min(N, j + 100))) for j in range(0, N, 100)]
    Rms = {c: np.full((len(MULTISETS), N, 3, 3), np.nan) for c in cfgs}
    t1 = time.time()
    try:
        with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(P['work'],)) as ex:
            for n_done, (idx, res) in enumerate(ex.map(_ms_chunk, [(ch, cfgs) for ch in chunks])):
                for c in cfgs:
                    Rms[c][:, idx] = np.transpose(res[c], (1, 0, 2, 3))
                if n_done % 8 == 7:
                    print('   multisets: %d / %d chunks, %.0f s' % (n_done + 1, len(chunks), time.time() - t1), flush=True)
    finally:
        if os.path.exists(P['work']):
            os.remove(P['work'])
    refs = {'kz': A0['SATM'], 'kz_rho': A0p['SATM'], 'loo_rho': LOO['SATM']}
    if have_bn:
        refs['bn'] = out['BN']
    for c in cfgs:
        assert np.array_equal(Rms[c][IDENT], refs[c], equal_nan=True), 'identity multiset != full arm for %s' % c
        out['ms_' + c] = Rms[c]
    out['multisets'] = np.array(MULTISETS)
    KFO.say('[freeze2] 126 multisets x %s done in %.0f s (identity multiset == full arms: asserted)' % (cfgs, time.time() - t1), log)
    with open(P['frozen'] + '.tmp', 'wb') as f:
        np.savez(f, **out)
    os.replace(P['frozen'] + '.tmp', P['frozen'])
    J.update({'frozen_sha256': S.sha256(P['frozen']), 'script_sha256': S.sha256(os.path.abspath(__file__)),
              'common_sha256': S.sha256(os.path.join(HERE, 'kzst2_common.py')), 'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
              'elapsed_s': time.time() - t0, 'log': log})
    S.jdump(J, P['frozen_json'])
    KFO.say('[freeze2] wrote %s (sha256 %s); A0 == frozen_e1sa %s; BN %s (%.0f s)' % (
        P['frozen'], J['frozen_sha256'], 'asserted' if a0_checked else 'not checked (no frozen_e1sa: fixture)', have_bn, time.time() - t0))
    return J


# ======================================================================================= evaluate (reads GT)
def two_level(HX, HY, sh, groups, B, seed):
    """HX, HY: (126, n_hit) per-multiset heading errors of the two arms (same multiset index = same seed draw).
    Each replicate: one seed multiset (5 draws with replacement of k, shared by both arms) then a sequence-cluster resample."""
    ms_index = {ms: j for j, ms in enumerate(MULTISETS)}
    rng = np.random.default_rng(seed)
    idx_by = [np.where(sh == s)[0] for s in groups]
    bs = np.empty(B)
    for b in range(B):
        j = ms_index[tuple(sorted(rng.integers(0, 5, 5).tolist()))]
        ii = np.concatenate([idx_by[g] for g in rng.integers(0, len(groups), len(groups))])
        bs[b] = np.median(HX[j][ii]) - np.median(HY[j][ii])
    return bs


def stage_evaluate(a):
    P = paths(a)
    pre, _ = S.check_prereg(a.prereg)
    FJ = json.load(open(P['frozen_json'], encoding='utf-8'))
    if FJ['frozen_sha256'] != S.sha256(P['frozen']) or FJ['prereg_sha256'] != pre:
        raise SystemExit('frozen_test_stage2 hash / prereg mismatch')
    for p in (S.ROOT, S.TOOLS, S.TOOLS + '/annot_audit'):
        if p not in sys.path:
            sys.path.insert(0, p)
    import summarize_dumps as SD
    Z = dict(np.load(P['frozen'], allow_pickle=False))
    _, info_m = C1.load_npz_info(a.test_npz)
    keys = [(str(s), str(f)) for s, f in zip(Z['seq'], Z['frame'])]
    top, cls = Z['top'], Z['cls']
    seqs = np.array([k[0] for k in keys])
    S1R = json.load(open(P['stage1_results'], encoding='utf-8')) if os.path.exists(P['stage1_results']) else None
    frames, _ = KFO.load_dump(a.test_dump, with_gt=True)                  # ---- ground truth read from here on ----
    R = {'prereg_sha256': pre, 'frozen_test_sha256': FJ['frozen_sha256'], 'gt_first_read_at': time.strftime('%Y-%m-%d %H:%M:%S'),
         'real_test_inputs': FJ['real_test_inputs']}
    log = []
    have_bn = bool(FJ['have_bn'])
    arms = ['A0', 'A0p', 'LOO', 'LOO_dep'] + (['BN', 'BN_rho'] if have_bn else [])
    M = {an: KFO.per_frame_metrics(KFO.build_arm_frames(frames, keys, Z[an], top)) for an in arms}
    hit = np.isfinite(M['A0']['he'])
    for an in arms:
        assert np.array_equal(np.isfinite(M[an]['he']), hit) and np.array_equal(M[an]['pos'], M['A0']['pos']), 'positions / detections changed (%s)' % an
    real = bool(FJ['real_test_inputs'])
    s_a0 = SD.stats(KFO.build_arm_frames(frames, keys, Z['A0'], top))
    if real:
        e5 = json.load(open(S.TEST_RESULTS_E5, encoding='utf-8'))
        assert abs(s_a0['head'] - e5['table']['SATM']['all']['head']) < 1e-9 and abs(s_a0['tilt'] - e5['table']['SATM']['all']['tilt']) < 1e-9
    assert float(np.median(M['A0']['he'][hit])) == s_a0['head']
    classes = {'all': np.ones(len(keys), bool), 'phantom4': cls == 'phantom4', 'mavic2': cls == 'mavic2'}
    R['table'] = {an: {c: KFO.summary_numbers(M[an], sel) for c, sel in classes.items()} for an in arms}
    singles = {'kz': Z['A0_single'], 'kz_rho': Z['A0p_single'], 'loo_rho': Z['LOO_single'], 'loo_dep': Z['LOO_dep_single']}
    if have_bn:
        singles.update({'bn': Z['BN_single'], 'bn_rho': Z['BN_rho_single']})
    Ms = {}
    for tag, arr in singles.items():
        for k in range(5):
            Ms['%s_single%d' % (tag, k)] = KFO.per_frame_metrics(KFO.build_arm_frames(frames, keys, arr[:, k], top))
    sp = {n: float(np.median(m['he'][hit])) for n, m in Ms.items()}
    R['singles_selfpeak_head'] = sp
    if real:
        for k in range(5):
            assert round(sp['kz_single%d' % k], 2) == E5_REF['single_kz_selfpeak'][k]
    h = {an: float(np.median(M[an]['he'][hit])) for an in arms}
    if S1R is not None and have_bn:
        assert abs(h['BN'] - S1R['table']['BN']['all']['head']) < 1e-9 and abs(h['A0p'] - S1R['table']['A0p']['all']['head']) < 1e-9, \
            'BN / A0p do not reproduce results_stage1.json'
    KFO.say('[evaluate2] reproduction: A0 %.2f / tilt %.2f == E5; kz self-peak singles == E5; BN %s / A0p %.2f == stage 1' % (
        s_a0['head'], s_a0['tilt'], ('%.2f' % h['BN']) if have_bn else 'n/a', h['A0p']), log)

    # ---------------- baselines: A0* (stage-1 rule) and A0** = better of A0* and BN (coordinator amendment, prereg)
    a0star = 'A0p' if h['A0p'] < h['A0'] else 'A0'
    if S1R is not None:
        assert S1R['A0_star']['arm'] == a0star, 'A0* differs from stage 1'
    a0ss = 'BN' if (have_bn and h['BN'] < h[a0star]) else a0star
    R['A0_star'] = {'arm': a0star, 'head': h[a0star]}
    R['A0_2star'] = {'arm': a0ss, 'head': h[a0ss], 'rule': 'lower all-frame fused heading median of A0* and BN (tie -> A0*)'}
    ms_of = {'A0': 'ms_kz', 'A0p': 'ms_kz_rho', 'BN': 'ms_bn', 'LOO': 'ms_loo_rho'}

    Rg = np.stack([KFO.rot_from_eul(frames[k]['gt'][6:9]) if hit[i] else np.eye(3) for i, k in enumerate(keys)])
    is_top = np.array([hit[i] and frames[k]['matched'] == top[i] for i, k in enumerate(keys)])

    def he_ms(arm):
        Rms_arr = Z[ms_of[arm]]
        o = np.empty((len(MULTISETS), int(hit.sum())))
        const = M[arm]['he'].copy()
        for j in range(len(MULTISETS)):
            v = const.copy()
            v[is_top] = E1.heading_err_vec(Rg[is_top], E1.roundtrip(Rms_arr[j][is_top]))[0]
            o[j] = v[hit]
        assert np.allclose(o[IDENT], M[arm]['he'][hit], atol=1e-9)
        return o
    HE = {arm: he_ms(arm) for arm in ['LOO', a0star, a0ss]}
    groups, draws = KFO.boot_draws(seqs[hit], P1['B'], P1['seed'])
    sh = seqs[hit]

    def dmed(x, y, key='he'):
        vx, vy = M[x][key][hit], M[y][key][hit]
        return lambda ii: float(np.median(vx[ii]) - np.median(vy[ii]))

    p_bn = float(S1R['P1_BN']['p_one_sided']) if (S1R is not None and 'P1_BN' in S1R) else 1.0

    def p1_block(base, seed_off, paired_base):
        # paired seeds: LOO_k vs base_k self-peak under the SAME decoder D_rho (protocol wording), whatever the arm decoder
        st = dmed('LOO', base)
        delta = st(np.arange(hit.sum()))
        bs2 = two_level(HE['LOO'], HE[base], sh, groups, P1['B'], P1['seed'] + seed_off)
        p = float((1 + np.sum(bs2 >= 0)) / (len(bs2) + 1))
        per_seq = {s: st(np.where(sh == s)[0]) for s in groups}
        n_better = int(sum(v < 0 for v in per_seq.values()))
        paired = {k: {'loo': sp['loo_rho_single%d' % k], 'base': sp['%s_single%d' % (paired_base, k)]} for k in range(5)}
        n_paired = int(sum(v['loo'] < v['base'] for v in paired.values()))
        bs1 = KFO.cluster_boot(st, sh, groups, draws)
        return {'base': base, 'base_head': h[base], 'LOO_head': h['LOO'], 'delta': delta, 'paired_base_singles': paired_base,
                'two_level_ci95': [float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))], 'p_one_sided': p,
                'single_level_ci95': [float(np.percentile(bs1, 2.5)), float(np.percentile(bs1, 97.5))],
                'per_seq_delta': per_seq, 'n_seq_better': n_better, 'paired_seeds': paired, 'n_paired_better': n_paired}

    prim = p1_block(a0ss, 1, 'bn_rho' if a0ss == 'BN' else 'kz_rho')
    rej = KFO.holm(np.array([prim['p_one_sided'], p_bn]), P1['alpha'])        # family {P1-LOO (primary), P1-BN (stage 1)}
    prim['holm_family'] = {'P1_LOO_primary': prim['p_one_sided'], 'P1_BN_stage1': p_bn}
    prim['holm_reject'] = bool(rej[0])
    prim['criteria'] = {k: bool(v) for k, v in {
        'delta_le_-1': prim['delta'] <= P1['delta_max'], 'ci_upper_lt_0': prim['two_level_ci95'][1] < 0,
        'seq_better_ge_7': prim['n_seq_better'] >= P1['seq_better_min'], 'paired_ge_4': prim['n_paired_better'] >= P1['paired_seed_min'],
        'holm': prim['holm_reject']}.items()}
    prim['pass'] = bool(all(prim['criteria'].values()))
    R['P1_primary_vs_A0_2star'] = prim
    sec = p1_block(a0star, 2, 'kz_rho')
    rej2 = KFO.holm(np.array([sec['p_one_sided'], p_bn]), P1['alpha'])
    sec['holm_reject_report'] = bool(rej2[0])
    sec['criteria_report'] = {k: bool(v) for k, v in {
        'delta_le_-1': sec['delta'] <= P1['delta_max'], 'ci_upper_lt_0': sec['two_level_ci95'][1] < 0,
        'seq_better_ge_7': sec['n_seq_better'] >= P1['seq_better_min'], 'paired_ge_4': sec['n_paired_better'] >= P1['paired_seed_min'],
        'holm': sec['holm_reject_report']}.items()}
    sec['would_pass_report_only'] = bool(all(sec['criteria_report'].values()))
    R['P1_secondary_vs_A0_star'] = sec
    for tag, blk in (('primary vs A0** = %s' % a0ss, prim), ('secondary vs A0* = %s' % a0star, sec)):
        KFO.say('[evaluate2] P1 %s: LOO %.2f vs %.2f: delta %+.2f, two-level CI [%+.2f, %+.2f], p %.4f, seq %d/10, paired %d/5' % (
            tag, blk['LOO_head'], blk['base_head'], blk['delta'], *blk['two_level_ci95'], blk['p_one_sided'], blk['n_seq_better'],
            blk['n_paired_better']), log)

    # ---------------- GT strata (evaluation only)
    gt_h = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        if hit[i]:
            gt_h[i] = C1.hd(KFO.rot_from_eul(frames[k]['gt'][6:9]))
    view = np.array([C1.side_regime(x) if np.isfinite(x) else '' for x in gt_h])
    atyp = np.zeros(len(keys), bool)
    for s in sorted(set(seqs[hit])):
        ii = np.where(hit & (seqs == s))[0]
        hh = gt_h[ii]
        dd = np.abs(((hh[:, None] - hh[None, :]) + 180.0) % 360.0 - 180.0)
        dd = np.minimum(dd, 180.0 - dd)
        med = hh[int(np.argmin(dd.sum(1)))]
        atyp[ii] = np.array([C1.f180(x - med) for x in hh]) > 20.0
    for an in arms:
        Fa = KFO.build_arm_frames(frames, keys, Z[an], top)
        unf = np.full(len(keys), np.nan)
        for i, f in enumerate(Fa):
            if f['matched'] is not None:
                E = KFO.rot_from_eul(f['gt'][6:9]).T @ KFO.rot_from_eul(f['pred'][f['matched']][6:9])
                unf[i] = abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
        M[an]['ang_unf'] = unf

    def flip_rate(an):
        return float(np.mean(M[an]['ang_unf'][hit] > 90.0))

    def strata_table(an, base):
        o = {}
        for lab, mask in [('view_' + v, view == v) for v in ('side', 'oblique', 'along')] + [('gt_atypical', atyp)] + \
                [('scene_' + sc, np.array([C1.scene_of(s) == sc for s in seqs])) for sc in sorted({C1.scene_of(s) for s in seqs})] + \
                [('class_' + c, cls == c) for c in S.CLASSES] + \
                [('e5_' + s, Z['e5_stratum'] == s) for s in ('agree', 'flip_only', 'trim_only', 'both')]:
            mm = mask & hit
            if mm.any():
                o[lab] = {'n': int(mm.sum()), 'head': float(np.median(M[an]['he'][mm])), 'base': float(np.median(M[base]['he'][mm])),
                          'delta': float(np.median(M[an]['he'][mm]) - np.median(M[base]['he'][mm]))}
        return o

    def demeaned_corr(an):
        Fa = KFO.build_arm_frames(frames, keys, Z[an], top)
        hp = np.array([C1.hd(KFO.rot_from_eul(f['pred'][f['matched']][6:9])) if f['matched'] is not None else np.nan for f in Fa])
        m = hit & np.isfinite(hp)
        return float(np.corrcoef(E1.seq_axial_resid(hp[m], seqs[m]), E1.seq_axial_resid(gt_h[m], seqs[m]))[0, 1])

    # ---------------- ADS indoor
    ads = {}
    if not a.skip_ads:
        for an in ['A0', 'LOO'] + (['BN'] if have_bn else []):
            ads[an] = KFO.ads_indoor(KFO.build_arm_frames(frames, keys, Z[an], top), info_m, P['ads'], an)
        if real and ads.get('A0') is not None:
            assert abs(ads['A0'] - E5_REF['ads_indoor']) < 0.01, ('ADS of A0 does not reproduce E5', ads['A0'])
    R['ads_indoor'] = ads

    # ---------------- guardrails (relative to A0*, as registered in stage 1 / the protocol)
    tl = R['table']['LOO']
    vt = strata_table('LOO', a0star)
    g = {'tilt': {'value': tl['all']['tilt'], 'max': GUARD['tilt_max'], 'pass': tl['all']['tilt'] <= GUARD['tilt_max']},
         'ads_indoor': {'value': ads.get('LOO'), 'min': GUARD['ads_min'], 'pass': (ads.get('LOO') is not None and ads['LOO'] >= GUARD['ads_min'])},
         'positions_identical': {'pass': True, 'note': 'asserted per frame (positions, detections, crops)'},
         'mavic2_head': {'value': tl['mavic2']['head'], 'max': GUARD['mavic2_max'], 'pass': tl['mavic2']['head'] <= GUARD['mavic2_max']},
         'flip_rate_unfolded': {'LOO': flip_rate('LOO'), 'A0_star': flip_rate(a0star),
                                'pass': 100.0 * (flip_rate('LOO') - flip_rate(a0star)) <= GUARD['flip_rate_pp_max']},
         'view_strata': {v: vt.get('view_' + v) for v in ('side', 'oblique', 'along')}}
    g['view_strata']['pass'] = all(vt['view_' + v]['delta'] <= GUARD['view_delta_max'] for v in ('side', 'oblique', 'along') if 'view_' + v in vt)
    at = vt.get('gt_atypical')
    g['gt_atypical'] = {'n': at['n'] if at else 0, 'delta': at['delta'] if at else None, 'is_guardrail': bool(at and at['n'] >= GUARD['atypical_min_n'])}
    g['gt_atypical']['pass'] = (not g['gt_atypical']['is_guardrail']) or at['delta'] <= GUARD['atypical_delta_max']
    if os.path.exists(P['sim_json']):
        SJ = json.load(open(P['sim_json'], encoding='utf-8'))
        if SJ.get('prereg_sha256') != pre or (SJ.get('smoke') and not a.allow_smoke):
            raise SystemExit('sim_diag_loo.json was produced under another prereg / is a smoke output')
        gs = SJ['guardrail_sim_forgetting_simBN']
        g['sim_forgetting'] = {'rule': gs['rule'], 'per_seed': {k: {'simBN_LOO': SJ['simbn']['LOO'][str(k)]['heading_fold180_median'],
                                                                     'simBN_KZ': SJ['simbn']['KZ'][str(k)]['heading_fold180_median']} for k in range(5)},
                               'pass': bool(gs['pass']), 'run': True,
                               'stage1_procedure_report_only': SJ.get('report_own_stage1_procedure')}
    else:
        g['sim_forgetting'] = {'run': False, 'pass': None}
    core = [g[k]['pass'] for k in ('tilt', 'ads_indoor', 'positions_identical', 'mavic2_head', 'flip_rate_unfolded', 'view_strata', 'gt_atypical')]
    g['all_pass_excluding_sim'] = bool(all(core))
    R['guardrails_LOO'] = g
    if prim['pass'] and g['all_pass_excluding_sim'] and g['sim_forgetting']['pass'] is True:
        dec = 'deploy_LOO'
    elif prim['pass'] and g['all_pass_excluding_sim'] and g['sim_forgetting']['pass'] is None:
        dec = 'pending_sim_guardrail'
    else:
        dec = 'do_not_deploy_LOO'
    R['decision_LOO'] = dec
    R['decision_BN_stage1'] = S1R.get('decision_BN') if S1R else None
    KFO.say('[evaluate2] guardrails LOO: %s ; sim %s -> decision %s (BN stage-1 decision %s, unchanged here)' % (
        {k: g[k]['pass'] for k in ('tilt', 'ads_indoor', 'mavic2_head', 'flip_rate_unfolded', 'view_strata', 'gt_atypical')},
        g['sim_forgetting']['pass'], dec, R['decision_BN_stage1']), log)

    # ---------------- report-only
    rep = {'LOO_strata_vs_A0star': vt, 'LOO_strata_vs_A0_2star': strata_table('LOO', a0ss) if a0ss != a0star else vt,
           'demeaned_corr': {an: demeaned_corr(an) for an in arms}, 'flip_rate_unfolded_gt90': {an: flip_rate(an) for an in arms},
           'LOO_dep_head_report_only': h['LOO_dep'], 'per_seed_selfpeak_head': sp}
    if have_bn:
        st = dmed('LOO', 'BN_rho')
        bsr = KFO.cluster_boot(st, sh, groups, draws)
        rep['LOO_minus_BN_rho_same_decoder'] = {'delta': st(np.arange(hit.sum())),
                                                 'single_level_ci95': [float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5))]}

    def pdis(MR):
        v = []
        for i in np.where(top >= 0)[0]:
            Ms_ = [MR[i, m] for m in range(5) if np.isfinite(MR[i, m]).all()]
            if len(Ms_) >= 2:
                v.append(np.median([KFO.fold(KFO.rel_yaw(Ms_[x], Ms_[y])) for x in range(len(Ms_)) for y in range(x + 1, len(Ms_))]))
        return float(np.median(v))
    dk, dkr, dl, dld = pdis(Z['A0_member_R']), pdis(Z['A0p_member_R']), pdis(Z['LOO_member_R']), pdis(Z['LOO_dep_member_R'])
    rep['member_pairwise_fold_median'] = {'kz_dep': dk, 'kz_rho': dkr, 'loo_rho': dl, 'loo_dep': dld, 'ratio_loo_over_kz_rho': dl / dkr,
                                          'collapse_flag_lt_0.5': bool(dl / dkr < 0.5)}
    if have_bn:
        rep['member_pairwise_fold_median']['bn_dep'] = pdis(Z['BN_member_R'])
    rho_c = {c: Z['rho'][j].tolist() for j, c in enumerate(S.CLASSES)}
    rep['span_ratio_median'] = {c: {'kz': np.nanmedian(Z['A0_ratios'][(cls == c)].reshape(-1, 2), 0).tolist(),
                                    'loo': np.nanmedian(Z['LOO_ratios'][(cls == c)].reshape(-1, 2), 0).tolist(), 'rho_target': rho_c[c]}
                                for c in S.CLASSES}
    for c in S.CLASSES:
        rep['span_ratio_median'][c]['loo_over_rho'] = (np.array(rep['span_ratio_median'][c]['loo']) / np.array(rho_c[c])).tolist()
    if os.path.exists(P['pseudo_json']):
        PJ = json.load(open(P['pseudo_json'], encoding='utf-8'))
        rep['pseudo_build'] = PJ.get('report_label_free')
    tr = {}
    for k in range(5):
        dj = os.path.join(P['train_root'], 'LOO_s%d' % k, 'DONE.json')
        if os.path.exists(dj):
            J = json.load(open(dj, encoding='utf-8'))
            tr[k] = {kk: J.get(kk) for kk in ('iterations', 'n_frames', 'n_quotient_frames', 'loss_terms_first_last50', 'train_seconds', 'smoke')}
    rep['training'] = tr
    FTJ = json.load(open(S.STAGE1['frozen_train_json'], encoding='utf-8')) if os.path.exists(S.STAGE1['frozen_train_json']) else {}
    rep['teacher_view_before_after_selection'] = {c: {kk: FTJ.get('report_label_free', {}).get(c, {}).get(kk) for kk in ('teacher_view_g0', 'teacher_view_selected')}
                                                  for c in S.CLASSES}
    R['report'] = rep
    R['log'] = log
    S.jdump(R, P['results'])
    write_report(R, P['report'])
    KFO.say('[evaluate2] wrote %s and %s' % (P['results'], P['report']))
    return R


def write_report(R, path):
    T = R['table']
    L = ['# KZ-ST v2 第二段结果（LOO 自训练；%s 首次读 test 真值）' % R['gt_first_read_at'], '',
         'prereg_stage2 sha256 `%s`；frozen_test_stage2 `%s`' % (R['prereg_sha256'], R['frozen_test_sha256']), '',
         'A0* = `%s`（%.2f°）；A0** = `%s`（%.2f°，主判据基线）' % (R['A0_star']['arm'], R['A0_star']['head'], R['A0_2star']['arm'], R['A0_2star']['head']), '',
         '| 臂 | 航向折180 全部 | phantom4 | mavic2 | <10° | <20° | 倾斜 | 10°10cm | ADD(0.5d) |', '|---|---|---|---|---|---|---|---|---|']
    for an, t in T.items():
        L.append('| %s | %.2f° | %.2f° | %.2f° | %.1f%% | %.1f%% | %.2f° | %.2f%% | %.2f%% |' % (
            an, t['all']['head'], t['phantom4']['head'], t['mavic2']['head'], t['all']['head_lt10'], t['all']['head_lt20'], t['all']['tilt'],
            t['all']['acc_10deg10cm'], t['all']['add_0p5d']))
    p, s = R['P1_primary_vs_A0_2star'], R['P1_secondary_vs_A0_star']
    L += ['', '## P1 主判据：LOO 对 A0**（%s）' % p['base'], '',
          'LOO %.2f° 对 %.2f°：Δ %+.2f°；两级 bootstrap 95%% [%+.2f, %+.2f]（单级 [%+.2f, %+.2f]），单侧 p %.4f，Holm（与 P1-BN p=%.4f 同族）%s；'
          '变好序列 %d/10；配对种子 %d/5 → **%s**' % (
              p['LOO_head'], p['base_head'], p['delta'], *p['two_level_ci95'], *p['single_level_ci95'], p['p_one_sided'],
              p['holm_family']['P1_BN_stage1'], '拒绝' if p['holm_reject'] else '不拒绝', p['n_seq_better'], p['n_paired_better'],
              '通过' if p['pass'] else '未通过'),
          '', '## P1 次要（只报告）：LOO 对 A0*（%s）' % s['base'], '',
          'Δ %+.2f°；两级 95%% [%+.2f, %+.2f]，p %.4f；变好序列 %d/10；配对种子 %d/5；按原判据%s' % (
              s['delta'], *s['two_level_ci95'], s['p_one_sided'], s['n_seq_better'], s['n_paired_better'], '会通过' if s['would_pass_report_only'] else '不会通过'),
          '', 'ADS indoor：%s' % json.dumps(R['ads_indoor']), '', '护栏：%s' % json.dumps(R['guardrails_LOO'], ensure_ascii=False), '',
          '**决定：%s**（BN 的第一段决定 `%s` 本段不改）' % (R['decision_LOO'], R['decision_BN_stage1']), '',
          '只报告：%s' % json.dumps(R['report'], ensure_ascii=False)]
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['freeze', 'evaluate'])
    ap.add_argument('--out-dir', default=S.EVAL_DIR)
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--test-npz', default=S.TEST_NPZ)
    ap.add_argument('--test-dump', default=S.TEST_DUMP)
    ap.add_argument('--test-frozen', default=S.TEST_FROZEN)
    ap.add_argument('--loo-npz', default=None)
    ap.add_argument('--bn-npz', default=None)
    ap.add_argument('--sim-json', default=None)
    ap.add_argument('--stage1-results', default=None)
    ap.add_argument('--stage1-frozen-test', default=None)
    ap.add_argument('--frozen-train', default=None)
    ap.add_argument('--train-root', default=None)
    ap.add_argument('--pseudo-json', default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--skip-ads', action='store_true')
    ap.add_argument('--allow-smoke', action='store_true', help='self-test only')
    ap.add_argument('--allow-nontrain-rho', action='store_true', help='self-test only')
    a = ap.parse_args(argv)
    S.ensure_under_out(os.path.abspath(a.out_dir))
    os.makedirs(a.out_dir, exist_ok=True)
    return {'freeze': stage_freeze, 'evaluate': stage_evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
