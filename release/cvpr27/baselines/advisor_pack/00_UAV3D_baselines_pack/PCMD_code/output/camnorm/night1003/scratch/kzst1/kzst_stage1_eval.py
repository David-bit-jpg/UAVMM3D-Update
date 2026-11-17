# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 1, S8 (CPU): A0' and the AdaBN arm on the real MAV6D TEST split (4800 frames).

  freeze    LABEL-FREE. Per-frame rotations on test of
              A0   = deployed 5-member KZ SA-TM (E5; asserted bit-identical to e1sa/frozen_e1sa.npz)
              A0'  = the same 5 members decoded with D_rho, rho_c frozen from TRAIN (frozen_train_stage1.npz)
              BN   = the 5 AdaBN members (adabn/kz_members_adabn_test.npz), deployed decoder, SA-TM
              self-peak singles of every member set; report-only variants (A0' with the agree-stratum rho, BN + D_rho)
            and the SA-TM of all 126 seed multisets (5 draws with replacement) for kz / kz+D_rho / BN, used by the
            two-level bootstrap. -> frozen_test_stage1.npz / .json (sha256)
  evaluate  reads the test ground truth (first time in this experiment): KZ_SUMMARY metrics, A0*, A0' vs A0, P1-BN
            (two-level bootstrap, sequences, paired seeds, Holm), guardrails, report-only items
            -> results_stage1.json, RESULTS_stage1.md
    python kzst_stage1_eval.py freeze
    python kzst_stage1_eval.py evaluate
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse      # noqa: E402
import concurrent.futures as cf   # noqa: E402
import itertools     # noqa: E402
import json          # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402
from scipy.spatial.transform import Rotation as Rot   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import kzst_common as C   # noqa: E402

KFO = C.KFO
MULTISETS = sorted({tuple(sorted(t)) for t in itertools.product(range(5), repeat=5)})
IDENT = MULTISETS.index((0, 1, 2, 3, 4))
assert len(MULTISETS) == 126
CONFIGS = ['kz', 'kz_rho', 'bn']
E5_REF = {'A0_head': 12.60, 'tilt': 8.52, 'ads_indoor': 50.88, 'mavic2_head': 27.56,
          'single_kz_selfpeak': [16.21, 18.11, 18.31, 19.29, 16.75]}
GUARD = {'tilt_max': 8.52 + 0.3, 'ads_min': 50.88 - 0.3, 'mavic2_max': 27.56 + 1.0, 'flip_rate_pp_max': 3.0,
         'view_delta_max': 1.0, 'atypical_delta_max': 1.0, 'atypical_min_n': 200, 'sim_margin': 1.5}
P1 = {'delta_max': -1.0, 'seq_better_min': 7, 'paired_seed_min': 4, 'alpha': 0.05, 'B': 10000, 'seed': 20261003}


def paths(a):
    d = a.dir
    return {'frozen_train': os.path.join(d, 'frozen_train_stage1.npz'), 'frozen_train_json': os.path.join(d, 'frozen_train_stage1.json'),
            'bn_npz': a.bn_npz or os.path.join(d, 'adabn', 'kz_members_adabn_test.npz'),
            'sim_json': a.sim_json or os.path.join(d, 'adabn', 'sim_diag_adabn.json'),
            'frozen': os.path.join(d, 'frozen_test_stage1.npz'), 'frozen_json': os.path.join(d, 'frozen_test_stage1.json'),
            'work': os.path.join(d, '_ms_work.npz'), 'results': os.path.join(d, 'results_stage1.json'),
            'report': os.path.join(d, 'RESULTS_stage1.md'), 'ads': os.path.join(d, 'ads_stage1')}


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
            src = 'bn' if c == 'bn' else 'kz'
            g = KFO.Geo(W['K_in'][i], W['dist'][i], W['raw_size'][i], W['new_size'][i], W['stride_b'][i], W['pos_rc'][i],
                        W['center_res'][i], W['lwh'][i], ks)
            ctx = C.RhoCtx(W[src + '_hm'][i], W[src + '_kp'][i], W['z'][i], g, W['rho'][i] if c == 'kz_rho' else None)
            rb, cb = W[src + '_rb_cb'][i]
            rc, cc = W['rc_cc'][i]
            rec_cell = (int(rb - (rc - search)), int(cb - (cc - search)))
            for j, ms in enumerate(MULTISETS):
                mem = list(ms)
                cell = rec_cell if j == IDENT else ctx.cell_for(mem)
                o, _, _, _ = KFO.fuse_subset(ctx, mem, cell, ['C0', 'SATM'], W['Rdep'][i])
                out[c][n, j] = o['SATM']
    return idx, out


# ======================================================================================= freeze (label-free)
def arms_for(D, info, frames, T, rho_by_cls, members=None):
    members = C.member_index(info) if members is None else members
    search = int(info.get('search', 2))
    ks = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    N = len(T['seq'])
    o = {'SATM': np.full((N, 3, 3), np.nan), 'C0': np.full((N, 3, 3), np.nan), 'single': np.full((N, 5, 3, 3), np.nan),
         'member_R': np.full((N, 5, 3, 3), np.nan), 'stratum': np.full(N, 'no_det', dtype='<U12'), 'top': np.full(N, -1, np.int64),
         'fb_SATM': np.zeros(N, bool), 'ratios': np.full((N, 5, 2), np.nan)}
    for i, k in enumerate(T['keys']):
        rho = None if rho_by_cls is None else rho_by_cls[T['cls'][i]]
        a = C.frame_arms(D, i, frames[k], members, search, ks, rho, selfpeak=True)
        if a is None:
            continue
        o['top'][i] = a['top']
        o['SATM'][i], o['C0'][i], o['stratum'][i], o['fb_SATM'][i] = a['SATM'], a['C0'], a['stratum'], a['fb_SATM']
        for m in range(5):
            o['single'][i, m] = a['selfpeak'][m][0]
            if a['member_R'][m] is not None:
                o['member_R'][i, m] = a['member_R'][m]
        if rho is None:
            for m, rr in enumerate(C.rho_ratios(a['ctx'], members, a['cell'], frames[k]['pred'][a['top'], :3])):
                if rr is not None:
                    o['ratios'][i, m] = rr
    return o


def stage_freeze(a):
    P = paths(a)
    pre, _ = C.check_prereg(a.prereg)
    t0 = time.time()
    FT = json.load(open(P['frozen_train_json'], encoding='utf-8'))
    if FT['frozen_npz_sha256'] != C.sha256(P['frozen_train']) or FT['prereg_sha256'] != pre:
        raise SystemExit('frozen_train_stage1 hash / prereg mismatch')
    if not FT['checks']['pass']:
        raise SystemExit('train dump replication checks did not pass -- rho is not usable')
    if FT['split'] != 'train' and not a.allow_nontrain_rho:
        raise SystemExit('rho must come from the TRAIN split (got %r)' % FT['split'])
    ZT = np.load(P['frozen_train'])
    rho = {str(c): tuple(float(x) for x in r) for c, r in zip(ZT['rho_classes'], ZT['rho'])}
    rho_ag = {str(c): tuple(float(x) for x in r) for c, r in zip(ZT['rho_classes'], ZT['rho_agree_report_only'])}
    D, info = C.load_npz_info(a.test_npz)
    frames, _ = C.load_dump_nogt(a.test_dump)
    T = C.frame_table(D, frames)
    N = len(T['seq'])
    log = []
    C.KFO.say('[freeze-test] %d test frames; rho (train) %s' % (N, rho), log)
    A0 = arms_for(D, info, frames, T, None)
    A0p = arms_for(D, info, frames, T, rho)
    A0pa = arms_for(D, info, frames, T, rho_ag) if all(np.isfinite(v).all() for v in rho_ag.values()) else None
    # A0 must be E5's SA-TM bit-for-bit
    if os.path.exists(a.test_frozen):
        Z5 = dict(np.load(a.test_frozen, allow_pickle=False))
        arms5 = list(Z5['arm_names'])
        assert np.array_equal(A0['SATM'], Z5['R'][arms5.index('SATM')]), 'A0 != frozen_e1sa SA-TM'
        for k in range(5):
            assert np.array_equal(A0['single'][:, k], Z5['R'][arms5.index('single_KZs%d' % k)])
        e5_strata = Z5['stratum']
    else:
        e5_strata = A0['stratum']
    out = {'seq': T['seq'], 'frame': T['frame'], 'cls': T['cls'], 'top': A0['top'], 'e5_stratum': e5_strata,
           'A0': A0['SATM'], 'A0_C0': A0['C0'], 'A0_single': A0['single'], 'A0_member_R': A0['member_R'], 'A0_ratios': A0['ratios'],
           'A0p': A0p['SATM'], 'A0p_C0': A0p['C0'], 'A0p_single': A0p['single'], 'A0p_stratum': A0p['stratum'],
           'rho': np.array([rho[c] for c in C.CLASSES]), 'rho_agree_report_only': np.array([rho_ag[c] for c in C.CLASSES])}
    if A0pa is not None:
        out['A0p_agree_report_only'] = A0pa['SATM']
    J = {'prereg_sha256': pre, 'frozen_train_sha256': FT['frozen_npz_sha256'], 'test_npz_sha256': C.sha256(a.test_npz),
         'test_dump_sha256': C.sha256(a.test_dump), 'rho': rho, 'rho_agree_report_only': rho_ag,
         'strata_A0p': {c: {s: float((A0p['stratum'][T['cls'] == c] == s).mean()) for s in C.STRATA} for c in C.CLASSES},
         'changed_A0p_vs_A0_heading_gt5_or_tilt_gt2': float(np.mean([
             (KFO.fold(KFO.rel_yaw(A0['SATM'][i], A0p['SATM'][i])) > 5 or KFO.tilt_between(A0['SATM'][i], A0p['SATM'][i]) > 2)
             for i in range(N) if A0['top'][i] >= 0]))}
    have_bn = os.path.exists(P['bn_npz'])
    if have_bn:
        Db, ib = C.load_npz_info(P['bn_npz'])
        assert ib.get('arm') == 'adabn' and not ib.get('smoke', False) or a.allow_smoke_bn, 'BN npz is a smoke output'
        assert ib.get('prereg_sha256') == pre, 'BN npz was produced under another prereg'
        for k in ('seq', 'frame', 'pos_rc', 'center_res', 'hm_peak', 'z', 'crop', 'rc_cc', 'win', 'K_in', 'dist', 'raw_size', 'new_size', 'stride_b'):
            if k not in D and k not in Db:
                continue                      # (synthetic fixtures carry no crop records)
            assert np.array_equal(Db[k], D[k][:len(Db[k])]), 'BN npz field %s differs from the e1sa test dump (positions / crops must be identical)' % k
        assert len(Db['seq']) == N, 'BN npz does not cover all test frames'
        BN = arms_for(Db, ib, frames, T, None, members=list(range(5)))
        assert np.array_equal(BN['top'], A0['top'])
        BNr = arms_for(Db, ib, frames, T, rho, members=list(range(5)))
        out.update({'BN': BN['SATM'], 'BN_C0': BN['C0'], 'BN_single': BN['single'], 'BN_member_R': BN['member_R'],
                    'BN_stratum': BN['stratum'], 'BN_ratios': BN['ratios'], 'BN_rho_report_only': BNr['SATM']})
        J['bn_npz_sha256'] = C.sha256(P['bn_npz'])
        J['bn_info'] = {k: ib.get(k) for k in ('replay_check', 'bn_stat_shift', 'reestimation', 'checkpoints_sha256', 'n_bn_layers')}
        J['strata_BN'] = {c: {s: float((BN['stratum'][T['cls'] == c] == s).mean()) for s in C.STRATA} for c in C.CLASSES}
        # multisets for the two-level bootstrap
        rgb = C.member_index(info)
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
              'bn_hm': Db['hm_win'], 'bn_kp': Db['kp_win'], 'bn_rb_cb': Db['rb_cb']}
        for k in ('K_in', 'dist', 'raw_size', 'new_size', 'stride_b', 'pos_rc', 'center_res', 'z', 'rc_cc'):
            Wk[k] = D[k]
        with open(P['work'], 'wb') as f:
            np.savez(f, **Wk)
        chunks = [list(range(j, min(N, j + 100))) for j in range(0, N, 100)]
        Rms = {c: np.full((len(MULTISETS), N, 3, 3), np.nan) for c in CONFIGS}
        t1 = time.time()
        with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(P['work'],)) as ex:
            for n_done, (idx, res) in enumerate(ex.map(_ms_chunk, [(ch, CONFIGS) for ch in chunks])):
                for c in CONFIGS:
                    Rms[c][:, idx] = np.transpose(res[c], (1, 0, 2, 3))
                if n_done % 8 == 7:
                    print('   multisets: %d / %d chunks, %.0f s' % (n_done + 1, len(chunks), time.time() - t1), flush=True)
        os.remove(P['work'])
        for c, ref in (('kz', A0['SATM']), ('kz_rho', A0p['SATM']), ('bn', BN['SATM'])):
            assert np.array_equal(Rms[c][IDENT], ref, equal_nan=True), 'identity multiset != full arm for %s' % c
        out['ms_kz'], out['ms_kz_rho'], out['ms_bn'] = Rms['kz'], Rms['kz_rho'], Rms['bn']
        out['multisets'] = np.array(MULTISETS)
        C.KFO.say('[freeze-test] 126 multisets x 3 configs done in %.0f s (identity multiset == full arms: asserted)' % (time.time() - t1), log)
    J['have_bn'] = have_bn
    with open(P['frozen'] + '.tmp', 'wb') as f:
        np.savez(f, **out)
    os.replace(P['frozen'] + '.tmp', P['frozen'])
    J.update({'frozen_sha256': C.sha256(P['frozen']), 'script_sha256': C.sha256(os.path.abspath(__file__)),
              'common_sha256': C.sha256(os.path.join(HERE, 'kzst_common.py')), 'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
              'elapsed_s': time.time() - t0, 'log': log})
    C.jdump(J, P['frozen_json'])
    C.KFO.say('[freeze-test] wrote %s (sha256 %s), A0 == frozen_e1sa asserted, BN %s (%.0f s)' % (P['frozen'], J['frozen_sha256'], have_bn, time.time() - t0))
    return J


# ======================================================================================= evaluate (reads GT)
def heading_err_vec(Rg, Rp):
    E = np.einsum('nji,njk->nik', Rg, Rp)
    a = np.abs(np.degrees(np.arctan2(E[:, 1, 0], E[:, 0, 0])))
    return np.minimum(a, 180.0 - a), a


def roundtrip(Rm):
    """the euler round trip of build_arm_frames + per_frame_metrics (rotation -> 'xyz' euler -> rotation)."""
    out = np.full_like(Rm, np.nan)
    ok = np.isfinite(Rm).all((1, 2))
    if ok.any():
        out[ok] = Rot.from_euler(KFO.SEQ, Rot.from_matrix(Rm[ok]).as_euler(KFO.SEQ)).as_matrix()
    return out


def seq_axial_resid(h, seqs):
    r = np.full(len(h), np.nan)
    for s in set(seqs):
        m = seqs == s
        mu = 0.5 * np.degrees(np.arctan2(np.sin(np.radians(2 * h[m])).sum(), np.cos(np.radians(2 * h[m])).sum()))
        r[m] = (h[m] - mu + 90.0) % 180.0 - 90.0
    return r


def stage_evaluate(a):
    P = paths(a)
    pre, _ = C.check_prereg(a.prereg)
    FJ = json.load(open(P['frozen_json'], encoding='utf-8'))
    if FJ['frozen_sha256'] != C.sha256(P['frozen']) or FJ['prereg_sha256'] != pre:
        raise SystemExit('frozen_test_stage1 hash / prereg mismatch')
    FT = json.load(open(P['frozen_train_json'], encoding='utf-8'))
    for p in (C.ROOT, C.TOOLS, C.TOOLS + '/annot_audit'):
        if p not in sys.path:
            sys.path.insert(0, p)
    import summarize_dumps as SD
    Z = dict(np.load(P['frozen'], allow_pickle=False))
    _, info_m = C.load_npz_info(a.test_npz)
    keys = [(str(s), str(f)) for s, f in zip(Z['seq'], Z['frame'])]
    top = Z['top']
    cls = Z['cls']
    seqs = np.array([k[0] for k in keys])
    frames, _ = KFO.load_dump(a.test_dump, with_gt=True)                  # ---- ground truth read from here on ----
    R = {'prereg_sha256': pre, 'frozen_test_sha256': FJ['frozen_sha256'], 'frozen_train_sha256': FJ['frozen_train_sha256'],
         'gt_first_read_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'rho_train': FJ['rho'], 'gate_final_disposition': FT['selection']['final_disposition'],
         'gate_verdict': {c: FT['gate']['verdict'][c]['pass'] for c in C.CLASSES}}
    log = []
    have_bn = bool(FJ['have_bn'])
    arms = ['A0', 'A0p'] + (['A0p_agree_report_only'] if 'A0p_agree_report_only' in Z else []) + (['BN', 'BN_rho_report_only'] if have_bn else [])
    singles = {'kz': Z['A0_single'], 'kz_rho': Z['A0p_single']}
    if have_bn:
        singles['bn'] = Z['BN_single']
    M = {an: KFO.per_frame_metrics(KFO.build_arm_frames(frames, keys, Z[an], top)) for an in arms}
    hit = np.isfinite(M['A0']['he'])
    for an in arms:
        assert np.array_equal(np.isfinite(M[an]['he']), hit) and np.array_equal(M[an]['pos'], M['A0']['pos']), 'positions / detections changed (%s)' % an
    # reproduction of E5 (A0 is E5's SA-TM) -- on the real test inputs only (self-tests use a synthetic fixture)
    real = os.path.normcase(os.path.abspath(a.test_npz)) == os.path.normcase(os.path.abspath(C.TEST_NPZ))
    R['real_test_inputs'] = real
    s_a0 = SD.stats(KFO.build_arm_frames(frames, keys, Z['A0'], top))
    if real:
        e5 = json.load(open(C.TEST_RESULTS_E5, encoding='utf-8'))
        assert abs(s_a0['head'] - e5['table']['SATM']['all']['head']) < 1e-9 and abs(s_a0['tilt'] - e5['table']['SATM']['all']['tilt']) < 1e-9
    assert float(np.median(M['A0']['he'][hit])) == s_a0['head']
    classes = {'all': np.ones(len(keys), bool), 'phantom4': cls == 'phantom4', 'mavic2': cls == 'mavic2'}
    R['table'] = {an: {c: KFO.summary_numbers(M[an], sel) for c, sel in classes.items()} for an in arms}
    Ms = {}
    for tag, arr in singles.items():
        for k in range(5):
            Ms['%s_single%d' % (tag, k)] = KFO.per_frame_metrics(KFO.build_arm_frames(frames, keys, arr[:, k], top))
    R['singles_selfpeak_head'] = {n: float(np.median(m['he'][hit])) for n, m in Ms.items()}
    if real:
        for k in range(5):
            assert round(R['singles_selfpeak_head']['kz_single%d' % k], 2) == E5_REF['single_kz_selfpeak'][k]
    C.KFO.say('[evaluate] reproduction: A0 = E5 SA-TM %.2f / tilt %.2f (asserted equal to results_e1sa.json); kz self-peak singles == E5' % (
        s_a0['head'], s_a0['tilt']), log)

    groups, draws = KFO.boot_draws(seqs[hit], P1['B'], P1['seed'])
    sh = seqs[hit]

    def dmed(x, y, key='he'):
        vx, vy = M[x][key][hit], M[y][key][hit]
        return lambda ii: float(np.median(vx[ii]) - np.median(vy[ii]))

    # ---------------- A0' vs A0 (no training: sequence-cluster bootstrap) and A0*
    h = {an: float(np.median(M[an]['he'][hit])) for an in arms}
    a0star = 'A0p' if h['A0p'] < h['A0'] else 'A0'
    st = dmed('A0p', 'A0')
    bs = KFO.cluster_boot(st, sh, groups, draws)
    R['A0p_vs_A0'] = {'A0_head': h['A0'], 'A0p_head': h['A0p'], 'delta': st(np.arange(hit.sum())),
                      'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      'p_one_sided': float((1 + np.sum(bs >= 0)) / (len(bs) + 1)),
                      'tilt_delta': dmed('A0p', 'A0', 'ti')(np.arange(hit.sum())),
                      'per_class_delta': {c: float(np.median(M['A0p']['he'][hit & classes[c]]) - np.median(M['A0']['he'][hit & classes[c]])) for c in C.CLASSES},
                      'per_seq_delta': {s: st(np.where(sh == s)[0]) for s in groups}}
    R['A0p_vs_A0']['decoding_improvement_flag'] = bool(R['A0p_vs_A0']['delta'] < 0 and R['A0p_vs_A0']['ci95'][1] < 0)
    R['A0_star'] = {'arm': a0star, 'head': h[a0star], 'rule': 'lower all-frame fused heading median of A0 and A0p (tie -> A0)'}
    C.KFO.say('[evaluate] A0 %.2f | A0\' %.2f (delta %+.2f, CI [%+.2f, %+.2f]; phantom4 %+.2f, mavic2 %+.2f) -> A0* = %s' % (
        h['A0'], h['A0p'], R['A0p_vs_A0']['delta'], *R['A0p_vs_A0']['ci95'], R['A0p_vs_A0']['per_class_delta']['phantom4'],
        R['A0p_vs_A0']['per_class_delta']['mavic2'], a0star), log)

    # ---------------- GT strata used by guardrails (evaluation only)
    gt_h = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        if hit[i]:
            gt_h[i] = C.hd(KFO.rot_from_eul(frames[k]['gt'][6:9]))
    view = np.array([C.side_regime(x) if np.isfinite(x) else '' for x in gt_h])
    atyp = np.zeros(len(keys), bool)
    for s in sorted(set(seqs[hit])):
        ii = np.where(hit & (seqs == s))[0]
        hh = gt_h[ii]
        dd = np.abs(((hh[:, None] - hh[None, :]) + 180.0) % 360.0 - 180.0)
        dd = np.minimum(dd, 180.0 - dd)
        med = hh[int(np.argmin(dd.sum(1)))]
        atyp[ii] = np.array([C.f180(x - med) for x in hh]) > 20.0

    def arm_pred_heading(an):
        Fa = KFO.build_arm_frames(frames, keys, Z[an], top)
        return np.array([C.hd(KFO.rot_from_eul(f['pred'][f['matched']][6:9])) if f['matched'] is not None else np.nan for f in Fa])

    def flip_rate(an):
        return float(np.mean(M[an]['ang_unf'][hit] > 90.0))

    for an in arms:
        Fa = KFO.build_arm_frames(frames, keys, Z[an], top)
        unf = np.full(len(keys), np.nan)
        for i, f in enumerate(Fa):
            if f['matched'] is not None:
                E = KFO.rot_from_eul(f['gt'][6:9]).T @ KFO.rot_from_eul(f['pred'][f['matched']][6:9])
                unf[i] = abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
        M[an]['ang_unf'] = unf

    def strata_table(an, base):
        o = {}
        for lab, mask in [('view_' + v, view == v) for v in ('side', 'oblique', 'along')] + [('gt_atypical', atyp)] + \
                [('scene_' + sc, np.array([C.scene_of(s) == sc for s in seqs])) for sc in sorted({C.scene_of(s) for s in seqs})] + \
                [('e5_' + s, Z['e5_stratum'] == s) for s in ('agree', 'flip_only', 'trim_only', 'both')]:
            mm = mask & hit
            if mm.any():
                o[lab] = {'n': int(mm.sum()), 'head': float(np.median(M[an]['he'][mm])), 'base': float(np.median(M[base]['he'][mm])),
                          'delta': float(np.median(M[an]['he'][mm]) - np.median(M[base]['he'][mm]))}
        return o

    def demeaned_corr(an):
        hp = arm_pred_heading(an)
        m = hit & np.isfinite(hp)
        rp = seq_axial_resid(hp[m], seqs[m])
        rg = seq_axial_resid(gt_h[m], seqs[m])
        return float(np.corrcoef(rp, rg)[0, 1])

    R['report'] = {'A0p_strata_vs_A0': strata_table('A0p', 'A0'), 'demeaned_corr': {}, 'flip_rate_unfolded_gt90': {}}
    for an in arms:
        R['report']['demeaned_corr'][an] = demeaned_corr(an)
        R['report']['flip_rate_unfolded_gt90'][an] = flip_rate(an)

    # ---------------- ADS indoor
    ads = {}
    if not a.skip_ads:
        for an in ['A0', 'A0p'] + (['BN'] if have_bn else []):
            ads[an] = KFO.ads_indoor(KFO.build_arm_frames(frames, keys, Z[an], top), info_m, P['ads'], an)
        if real and ads.get('A0') is not None:
            assert abs(ads['A0'] - E5_REF['ads_indoor']) < 0.01, ('ADS of A0 does not reproduce E5', ads['A0'])
    R['ads_indoor'] = ads

    # ---------------- P1-BN
    if have_bn:
        st = dmed('BN', a0star)
        delta = st(np.arange(hit.sum()))
        # two-level bootstrap: seeds (paired multiset) then sequences
        Rg = np.stack([KFO.rot_from_eul(frames[k]['gt'][6:9]) if hit[i] else np.eye(3) for i, k in enumerate(keys)])
        is_top = np.array([hit[i] and frames[k]['matched'] == top[i] for i, k in enumerate(keys)])
        base_ms = Z['ms_kz_rho'] if a0star == 'A0p' else Z['ms_kz']

        def he_ms(Rms_arr, an_ref):
            out = np.empty((len(MULTISETS), int(hit.sum())))
            const = M[an_ref]['he'].copy()                  # frames where the matched detection is not the top one: unchanged
            for j in range(len(MULTISETS)):
                v = const.copy()
                rp = roundtrip(Rms_arr[j][is_top])
                v[is_top] = heading_err_vec(Rg[is_top], rp)[0]
                out[j] = v[hit]
            return out
        HB = he_ms(Z['ms_bn'], 'BN')
        HA = he_ms(base_ms, a0star)
        assert np.allclose(HB[IDENT], M['BN']['he'][hit], atol=1e-9) and np.allclose(HA[IDENT], M[a0star]['he'][hit], atol=1e-9)
        ms_index = {ms: j for j, ms in enumerate(MULTISETS)}
        rng = np.random.default_rng(P1['seed'] + 1)
        idx_by = [np.where(sh == s)[0] for s in groups]
        bs2 = np.empty(P1['B'])
        for b in range(P1['B']):
            j = ms_index[tuple(sorted(rng.integers(0, 5, 5).tolist()))]
            ii = np.concatenate([idx_by[g] for g in rng.integers(0, len(groups), len(groups))])
            bs2[b] = np.median(HB[j][ii]) - np.median(HA[j][ii])
        p_bn = float((1 + np.sum(bs2 >= 0)) / (len(bs2) + 1))
        per_seq = {s: st(np.where(sh == s)[0]) for s in groups}
        n_better = int(sum(v < 0 for v in per_seq.values()))
        paired = {k: {'bn': R['singles_selfpeak_head']['bn_single%d' % k], 'kz': R['singles_selfpeak_head']['kz_single%d' % k]} for k in range(5)}
        n_paired = int(sum(v['bn'] < v['kz'] for v in paired.values()))
        rej = KFO.holm(np.array([1.0, p_bn]), P1['alpha'])           # [P1 (LOO, not run in stage 1 -> p = 1), P1-BN]
        p1 = {'BN_head': h['BN'], 'A0_star': a0star, 'A0_star_head': h[a0star], 'delta': delta,
              'two_level_ci95': [float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))], 'p_one_sided': p_bn,
              'holm_reject': bool(rej[1]), 'per_seq_delta': per_seq, 'n_seq_better': n_better, 'paired_seeds': paired,
              'n_paired_better': n_paired, 'single_level_ci95': None}
        bs1 = KFO.cluster_boot(st, sh, groups, draws)
        p1['single_level_ci95'] = [float(np.percentile(bs1, 2.5)), float(np.percentile(bs1, 97.5))]
        p1['criteria'] = {'delta_le_-1': delta <= P1['delta_max'], 'ci_upper_lt_0': p1['two_level_ci95'][1] < 0,
                          'seq_better_ge_7': n_better >= P1['seq_better_min'], 'paired_ge_4': n_paired >= P1['paired_seed_min'],
                          'holm': bool(rej[1])}
        p1['criteria'] = {k: bool(v) for k, v in p1['criteria'].items()}
        p1['pass'] = bool(all(p1['criteria'].values()))
        R['P1_BN'] = p1
        # guardrails
        tb = R['table']['BN']
        vt = strata_table('BN', a0star)
        g = {'tilt': {'value': tb['all']['tilt'], 'max': GUARD['tilt_max'], 'pass': tb['all']['tilt'] <= GUARD['tilt_max']},
             'ads_indoor': {'value': ads.get('BN'), 'min': GUARD['ads_min'], 'pass': (ads.get('BN') is not None and ads['BN'] >= GUARD['ads_min'])},
             'positions_identical': {'pass': True, 'note': 'asserted per frame (positions, detections, crops)'},
             'mavic2_head': {'value': tb['mavic2']['head'], 'max': GUARD['mavic2_max'], 'pass': tb['mavic2']['head'] <= GUARD['mavic2_max']},
             'flip_rate_unfolded': {'BN': flip_rate('BN'), 'A0_star': flip_rate(a0star),
                                    'pass': 100.0 * (flip_rate('BN') - flip_rate(a0star)) <= GUARD['flip_rate_pp_max']},
             'view_strata': {v: vt.get('view_' + v) for v in ('side', 'oblique', 'along')}}
        g['view_strata']['pass'] = all(vt['view_' + v]['delta'] <= GUARD['view_delta_max'] for v in ('side', 'oblique', 'along') if 'view_' + v in vt)
        at = vt.get('gt_atypical')
        g['gt_atypical'] = {'n': at['n'] if at else 0, 'delta': at['delta'] if at else None, 'is_guardrail': bool(at and at['n'] >= GUARD['atypical_min_n'])}
        g['gt_atypical']['pass'] = (not g['gt_atypical']['is_guardrail']) or at['delta'] <= GUARD['atypical_delta_max']
        if os.path.exists(P['sim_json']):
            S = json.load(open(P['sim_json'], encoding='utf-8'))
            g['sim_forgetting'] = {'per_seed': {k: {'adabn': S['adabn'][str(k)]['heading_fold180_median'], 'base': S['baseline_used'][k]} for k in range(5)},
                                   'baseline_source': S['baseline_source'], 'pass': bool(S['pass']), 'run': True}
        else:
            g['sim_forgetting'] = {'run': False, 'pass': None}
        core = [g[k]['pass'] for k in ('tilt', 'ads_indoor', 'positions_identical', 'mavic2_head', 'flip_rate_unfolded', 'view_strata', 'gt_atypical')]
        g['all_pass_excluding_sim'] = bool(all(core))
        R['guardrails_BN'] = g
        if p1['pass'] and g['all_pass_excluding_sim'] and g['sim_forgetting']['pass'] is True:
            dec = 'deploy_BN'
        elif p1['pass'] and g['all_pass_excluding_sim'] and g['sim_forgetting']['pass'] is None:
            dec = 'pending_sim_guardrail'
        else:
            dec = 'do_not_deploy_BN'
        R['decision_BN'] = dec
        R['report']['BN_strata_vs_A0star'] = vt
        R['report']['BN_rho_report_only_head'] = float(np.median(M['BN_rho_report_only']['he'][hit]))
        # label-free: member diversity (collapse monitor) and corner span ratio vs the KZ members
        def pdis(MR):
            v = []
            for i in np.where(top >= 0)[0]:
                Ms_ = [MR[i, m] for m in range(5) if np.isfinite(MR[i, m]).all()]
                if len(Ms_) >= 2:
                    v.append(np.median([KFO.fold(KFO.rel_yaw(Ms_[x], Ms_[y])) for x in range(len(Ms_)) for y in range(x + 1, len(Ms_))]))
            return float(np.median(v))
        dk, db = pdis(Z['A0_member_R']), pdis(Z['BN_member_R'])
        R['report']['member_pairwise_fold_median'] = {'kz': dk, 'bn': db, 'ratio_bn_over_kz': db / dk, 'collapse_flag_lt_0.5': bool(db / dk < 0.5)}
        R['report']['span_ratio_median'] = {c: {'kz': np.nanmedian(Z['A0_ratios'][(cls == c)].reshape(-1, 2), 0).tolist(),
                                                'bn': np.nanmedian(Z['BN_ratios'][(cls == c)].reshape(-1, 2), 0).tolist()} for c in C.CLASSES}
        C.KFO.say('[evaluate] P1-BN: BN %.2f vs A0* (%s) %.2f: delta %+.2f, two-level CI [%+.2f, %+.2f], p %.4f, holm %s, seq %d/10, paired %d/5 -> %s' % (
            h['BN'], a0star, h[a0star], delta, *p1['two_level_ci95'], p_bn, p1['holm_reject'], n_better, n_paired, 'PASS' if p1['pass'] else 'FAIL'), log)
        C.KFO.say('[evaluate] guardrails BN: %s ; sim %s -> decision %s' % (
            {k: g[k]['pass'] for k in ('tilt', 'ads_indoor', 'mavic2_head', 'flip_rate_unfolded', 'view_strata', 'gt_atypical')}, g['sim_forgetting']['pass'], dec), log)
    else:
        R['decision_BN'] = 'not_run'
    # A0' guardrail-style numbers (report only: A0' is a decoding reference, handed to idea #2)
    R['report']['A0p_guardrail_style'] = {'tilt': R['table']['A0p']['all']['tilt'], 'mavic2_head': R['table']['A0p']['mavic2']['head'],
                                          'ads_indoor': ads.get('A0p'), 'flip_rate_delta_pp_vs_A0': 100.0 * (flip_rate('A0p') - flip_rate('A0'))}
    if 'A0p_agree_report_only' in arms:
        R['report']['A0p_agree_report_only_head'] = float(np.median(M['A0p_agree_report_only']['he'][hit]))
    R['log'] = log
    C.jdump(R, P['results'])
    write_report(R, P['report'], have_bn)
    C.KFO.say('[evaluate] wrote %s and %s' % (P['results'], P['report']))
    return R


def write_report(R, path, have_bn):
    T = R['table']
    L = ['# KZ-ST v2 第一段结果（%s 首次读 test 真值）' % R['gt_first_read_at'], '',
         'prereg sha256 `%s`；frozen_test `%s`；frozen_train `%s`' % (R['prereg_sha256'], R['frozen_test_sha256'], R['frozen_train_sha256']), '',
         '闸门：phantom4 %s，mavic2 %s；处置 `%s`；train 上冻结的 rho：%s' % (
             '过' if R['gate_verdict']['phantom4'] else '不过', '过' if R['gate_verdict']['mavic2'] else '不过', R['gate_final_disposition'],
             json.dumps(R['rho_train'])), '',
         '| 臂 | 航向折180 全部 | phantom4 | mavic2 | <10° | <20° | 倾斜 | 10°10cm | ADD(0.5d) |', '|---|---|---|---|---|---|---|---|---|']
    for an, t in T.items():
        L.append('| %s | %.2f° | %.2f° | %.2f° | %.1f%% | %.1f%% | %.2f° | %.2f%% | %.2f%% |' % (
            an, t['all']['head'], t['phantom4']['head'], t['mavic2']['head'], t['all']['head_lt10'], t['all']['head_lt20'], t['all']['tilt'],
            t['all']['acc_10deg10cm'], t['all']['add_0p5d']))
    d = R['A0p_vs_A0']
    L += ['', "A0' 对 A0：Δ %+.2f°（序列聚类 95%% [%+.2f, %+.2f]，单侧 p %.4f）；phantom4 %+.2f°，mavic2 %+.2f°；倾斜 Δ %+.2f°；"
          "解码改进标记 %s（只记录，交给第 2 条形状自洽解码）" % (d['delta'], d['ci95'][0], d['ci95'][1], d['p_one_sided'],
                                                     d['per_class_delta']['phantom4'], d['per_class_delta']['mavic2'], d['tilt_delta'],
                                                     d['decoding_improvement_flag']),
          'A0* = `%s`（%.2f°）' % (R['A0_star']['arm'], R['A0_star']['head']), '', 'ADS indoor：%s' % json.dumps(R['ads_indoor'])]
    if have_bn:
        p = R['P1_BN']
        L += ['', '## P1-BN（AdaBN 臂，部署解码器，SA-TM）', '',
              'BN %.2f° 对 A0* %.2f°：Δ %+.2f°；两级 bootstrap 95%% [%+.2f, %+.2f]（单级 [%+.2f, %+.2f]），单侧 p %.4f，Holm %s；变好序列 %d/10；配对种子 %d/5 → **%s**' % (
                  p['BN_head'], p['A0_star_head'], p['delta'], p['two_level_ci95'][0], p['two_level_ci95'][1], p['single_level_ci95'][0],
                  p['single_level_ci95'][1], p['p_one_sided'], '拒绝' if p['holm_reject'] else '不拒绝', p['n_seq_better'], p['n_paired_better'],
                  '通过' if p['pass'] else '未通过'),
              '', '护栏：%s' % json.dumps(R['guardrails_BN'], ensure_ascii=False), '', '**决定：%s**' % R['decision_BN']]
    L += ['', '只报告：%s' % json.dumps(R['report'], ensure_ascii=False)]
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['freeze', 'evaluate'])
    ap.add_argument('--dir', default=HERE)
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--test-npz', default=C.TEST_NPZ)
    ap.add_argument('--test-dump', default=C.TEST_DUMP)
    ap.add_argument('--test-frozen', default=C.TEST_FROZEN)
    ap.add_argument('--bn-npz', default=None)
    ap.add_argument('--sim-json', default=None)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--skip-ads', action='store_true')
    ap.add_argument('--allow-smoke-bn', action='store_true', help='self-test only')
    ap.add_argument('--allow-nontrain-rho', action='store_true', help='self-test only')
    a = ap.parse_args(argv)
    return {'freeze': stage_freeze, 'evaluate': stage_evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
