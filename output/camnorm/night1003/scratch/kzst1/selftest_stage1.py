# -*- coding: utf-8 -*-
"""CPU self-tests for KZ-ST stage 1 (no GPU, no ground truth of any real split is read).

  t1  label-free REAL replication: the stage-1 per-frame pipeline run on the E5 test member dump reproduces
      frozen_e1sa.npz bit-for-bit (strata, SA-TM, member PnPs, self-peak singles) and the gate statistics reproduce
      author_kzst/test_ref_gated.json + test_ref_labelfree.json (the frozen test reference of the gate)
  t2  g0 replica on the ST1 train dump == the ST1 accepted frame set (index_pseudo_st1.pkl keys; label-free)
  t3  synthetic D_rho: planted anisotropic corner shrink (rho_u, rho_v) is recovered by the rho estimator, D_rho PnP
      recovers the rigid rotation (< 0.01 deg) while the deployed decoder does not; rho=None path is bit-identical
  t4  cap_uniform / select_per_seq / sample_size_check / gate_verdict logic on toy inputs
  t5  end-to-end kzst_stage1_freeze.py on a fake 'train' made from the e1sa TEST member dump (label-free), with a
      temporary prereg in ./smoke/ -- exercises checks, gate, rho, D_rho pass, selection, LOO, AdaBN list, outputs
    python selftest_stage1.py [t1 t2 t3 t4 t5]
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json          # noqa: E402
import pickle        # noqa: E402
import shutil        # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402
from scipy.spatial.transform import Rotation as Rot   # noqa: E402

import kzst_common as C   # noqa: E402

KFO = C.KFO
SMOKE = os.path.join(C.HERE, 'smoke')


def t1():
    import kzst_stage1_freeze as F
    t0 = time.time()
    D, info = C.load_npz_info(C.TEST_NPZ)
    frames, _ = C.load_dump_nogt(C.TEST_DUMP)
    T = C.frame_table(D, frames)
    A = F.compute_pass(D, info, frames, T, None, selfpeak=True, rho_ratios=True, log_every=0)
    Z = dict(np.load(C.TEST_FROZEN, allow_pickle=False))
    arms = list(Z['arm_names'])
    assert list(Z['seq']) == list(T['seq']) and list(Z['frame']) == list(T['frame'])
    assert np.array_equal(A['top'], Z['top'])
    assert np.array_equal(A['stratum'], Z['stratum']), 'strata differ from frozen_e1sa'
    assert np.array_equal(A['R_satm'], Z['R'][arms.index('SATM')]), 'SA-TM differs from frozen_e1sa'
    assert np.array_equal(A['R_c0'], Z['R'][arms.index('C0')]), 'C0 differs from frozen_e1sa'
    assert np.array_equal(A['member_R'], Z['member_R'], equal_nan=True), 'member PnPs differ from frozen_e1sa'
    for k in range(5):
        assert np.array_equal(A['R_single'][:, k], Z['R'][arms.index('single_KZs%d' % k)]), 'single %d differs' % k
    st = C.gate_stats(T['cls'], T['seq'], T['fk'], A['stratum'], A['R_satm'], A['member_R'], gap_ranks=2)
    ref_g = json.load(open(C.TEST_REF_GATED, encoding='utf-8'))
    ref_l = json.load(open(C.TEST_REF_LABELFREE, encoding='utf-8'))
    for c in C.CLASSES:
        g = st[c]['gated']
        r = ref_g[c]['gated']
        assert (g['n'], g['n_pairs']) == (r['n'], r['n_pairs']), (c, g, r)
        assert round(g['jump_median'], 2) == r['jump_gap2_median'] and round(g['pairwise_fold_median'], 2) == r['pairwise_fold_median'], (c, g, r)
        ad = st[c]['all_det']
        assert round(ad['jump_median'], 2) == ref_g[c]['all_det']['jump_gap2_median'] and round(ad['pairwise_fold_median'], 2) == ref_g[c]['all_det']['pairwise_fold_median']
        assert round(st[c]['gated_share'], 3) == ref_l[c]['agree+flip_only'], (c, st[c]['gated_share'], ref_l[c]['agree+flip_only'])
    bounds = C.gate_bounds(ref_g, ref_l)
    v, disp = C.gate_verdict(st, bounds)
    assert v['phantom4']['pass'] and v['mavic2']['pass'] and disp == 'train_both'      # test vs itself passes trivially
    # rho on test, label-free (smoke only; the real rho comes from TRAIN): compare with r3_alt/diag_scale.py numbers
    out = {}
    for c in C.CLASSES:
        for s in ('agree', 'all'):
            m = (T['cls'] == c) & ((A['stratum'] == s) if s != 'all' else (A['top'] >= 0))
            r = A['ratios'][m].reshape(-1, 2)
            r = r[np.isfinite(r).all(1)]
            out['%s_%s' % (c, s)] = (round(float(np.median(r[:, 0])), 3), round(float(np.median(r[:, 1])), 3), len(r))
    print('[t1] OK: stage-1 pipeline on the e1sa test dump == frozen_e1sa.npz bit-for-bit (strata, SA-TM, C0, member PnPs, '
          'self-peak singles); gate stats == test_ref_gated/labelfree.json %s; (%.0f s)' % (
              {c: (round(st[c]['gated_share'], 3), round(st[c]['gated']['jump_median'], 2), round(st[c]['gated']['pairwise_fold_median'], 2)) for c in C.CLASSES},
              time.time() - t0))
    print('[t1]     label-free span ratios on TEST (smoke only, not used anywhere): %s' % out)


def t2():
    t0 = time.time()
    fr, info = C.load_dump_nogt(C.ST1_TRAIN_DUMP)
    keys = sorted(fr)
    seq = np.array([k[0] for k in keys])
    fk = np.array([C.fk_digits(k[1]) for k in keys], np.int64)
    conf = np.full(len(keys), np.nan)
    pos = np.full((len(keys), 3), np.nan)
    for i, k in enumerate(keys):
        t = KFO.top_index(fr[k])
        if t >= 0:
            conf[i], pos[i] = fr[k]['conf'][t], fr[k]['pred'][t, :3]
    g0, reasons = C.g0_mask(seq, fk, conf, pos)
    idx = pickle.load(open(C.ST1_PSEUDO_INDEX, 'rb'))
    acc = {(m['seq'], m['frame']) for m in idx['metas'] if m}
    del idx
    mine = {keys[i] for i in np.where(g0)[0]}
    assert mine == acc, (len(mine), len(acc), len(mine ^ acc))
    print('[t2] OK: g0 replica on the ST1 train dump == ST1 accepted set (%d frames; reasons %s) (%.0f s)' % (len(acc), reasons, time.time() - t0))


def t3():
    rng = np.random.default_rng(5)
    lwh = np.array([0.3489, 0.3487, 0.1996])
    worst_rho, worst_dep, n = 0.0, 0.0, 0
    ratios = []
    rho_true = (0.73, 0.52)
    for (K, Rb, t) in KFO.synth_geometry(200, rng):
        q = K @ t
        u, v = q[0] / q[2], q[1] / q[2]
        g = KFO.Geo(K, np.zeros(5), (512, 288), (512, 288), 4.0, (int(v // 4), int(u // 4)), (u / 4 - int(u // 4), v / 4 - int(v // 4)), lwh)
        x = KFO.synth_member_vec(K, Rb, t, g, lwh)
        xs = x * C.rho16(rho_true)                                 # anisotropic shrink of the offsets around the centre
        # build a FrameCtx holding the shrunk vector as member 0 (z = 1, one cell)
        hm = np.ones((1, 1, 1), np.float32)
        kp = xs.astype(np.float32).reshape(1, 16, 1, 1)
        c_dep = C.RhoCtx(hm, kp, 1.0, g, None)
        c_rho = C.RhoCtx(hm, kp, 1.0, g, rho_true)
        Rd, Rr = c_dep.R(0, (0, 0)), c_rho.R(0, (0, 0))
        if Rd is None or Rr is None:
            continue
        n += 1
        worst_rho = max(worst_rho, KFO.geodesic(Rr, Rb))
        worst_dep = max(worst_dep, KFO.geodesic(Rd, Rb))
        rr = C.rho_ratios(c_dep, [0], (0, 0), t)[0]
        if rr is not None:
            ratios.append(rr)
        # rho=None path is bit-identical to the E5 FrameCtx
        e5 = KFO.FrameCtx(hm, kp, 1.0, g)
        assert np.array_equal(e5.x(0, (0, 0)), c_dep.x(0, (0, 0))) and np.array_equal(e5.R(0, (0, 0)), Rd)
    ratios = np.array(ratios)
    assert worst_rho < 0.01, worst_rho
    print('[t3] OK: D_rho recovers the rigid rotation from anisotropically shrunk corners (max %.1e deg over %d frames); '
          'deployed decoder off by up to %.1f deg; rho=None == E5 bit-for-bit' % (worst_rho, n, worst_dep))
    print('[t3]     rho estimator on the shrunk corners (truth %s): median (%.3f, %.3f) -- biased by the deployed PnP '
          'fitting the distorted box (label-free estimator, reported as is)' % (rho_true, np.median(ratios[:, 0]), np.median(ratios[:, 1])))
    # isotropic shrink: rho estimator exact (PnP rotation unchanged under similarity in weak perspective, t fixed)
    iso = []
    for (K, Rb, t) in KFO.synth_geometry(100, rng):
        q = K @ t
        u, v = q[0] / q[2], q[1] / q[2]
        g = KFO.Geo(K, np.zeros(5), (512, 288), (512, 288), 4.0, (int(v // 4), int(u // 4)), (u / 4 - int(u // 4), v / 4 - int(v // 4)), lwh)
        x = KFO.synth_member_vec(K, Rb, t, g, lwh) * 0.72
        c = C.RhoCtx(np.ones((1, 1, 1), np.float32), x.astype(np.float32).reshape(1, 16, 1, 1), 1.0, g, None)
        rr = C.rho_ratios(c, [0], (0, 0), t)[0]
        if rr is not None:
            iso.append(rr)
    iso = np.array(iso)
    print('[t3]     isotropic 0.72 shrink -> estimator median (%.3f, %.3f)' % tuple(np.median(iso, 0)))
    assert abs(np.median(iso[:, 0]) - 0.72) < 0.03 and abs(np.median(iso[:, 1]) - 0.72) < 0.03


def t4():
    assert C.cap_uniform(list(range(5)), 80) == list(range(5))
    p = C.cap_uniform(list(range(125)), 80)
    assert len(p) == 80 and p[0] == 0 and p[-1] == 124 and len(set(p)) == 80
    p = C.cap_uniform(list(range(81)), 80)
    assert len(p) == 80 and len(set(p)) == 80
    seq = np.array(['a'] * 100 + ['b'] * 30 + ['c'] * 10)
    fk = np.concatenate([np.arange(100)[::-1], np.arange(30), np.arange(10)])
    mask = np.ones(140, bool)
    mask[100:105] = False
    sel = C.select_per_seq(mask, seq, fk, 80)
    assert (seq[sel] == 'a').sum() == 80 and (seq[sel] == 'b').sum() == 25 and (seq[sel] == 'c').sum() == 10
    cls = np.array(['phantom4'] * 130 + ['mavic2'] * 10)
    ss = C.sample_size_check(sel, cls, seq, ['phantom4'])
    assert not ss['pass'] and not ss['checks']['total'] and not ss['checks']['n_seq'] and not ss['checks']['per_class'] and ss['per_class']['phantom4'] == 105
    # gate verdict logic
    b = {c: {'test_gated_share': 0.6, 'test_jump_gap2_median': 5.0, 'test_pairwise_fold_median': 10.0,
             'share_min': 0.5, 'jump_max': 6.5, 'pairwise_max': 13.0} for c in C.CLASSES}
    mk = lambda sh, j, pw: {'gated_share': sh, 'gated': {'jump_median': j, 'pairwise_fold_median': pw, 'n': 1, 'n_pairs': 1}}  # noqa: E731
    v, d = C.gate_verdict({'phantom4': mk(0.5, 6.5, 13.0), 'mavic2': mk(0.49, 1, 1)}, b)
    assert v['phantom4']['pass'] and not v['mavic2']['pass'] and d == 'train_phantom4_only'      # inclusive bounds
    v, d = C.gate_verdict({'phantom4': mk(0.6, 6.51, 1), 'mavic2': mk(0.9, 1, 1)}, b)
    assert not v['phantom4']['pass'] and d == 'no_LOO'
    # g0 on a toy sequence: one 1 m jump frame and one low-conf frame
    seq = np.array(['s'] * 10)
    fk = np.arange(10)
    conf = np.full(10, 0.9)
    conf[3] = 0.4
    pos = np.zeros((10, 3))
    pos[6] = [1.0, 0, 0]
    g0, r = C.g0_mask(seq, fk, conf, pos)
    assert not g0[3] and not g0[6] and g0[[0, 1, 2, 4, 5, 7, 8, 9]].all(), (g0, r)
    print('[t4] OK: cap_uniform / select_per_seq / sample sizes / gate verdict (inclusive bounds, dispositions) / g0 toy')


def write_temp_prereg(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'id': 'SMOKE-ONLY', 'note': 'temporary prereg for CPU self-tests; not the real pre-registration'}, f)
    import hashlib
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    open(path + '.sha256', 'w', encoding='utf-8').write('%s  %s\n' % (h, os.path.basename(path)))
    return path


def t5():
    import kzst_stage1_freeze as F
    d = os.path.join(SMOKE, 'fake_train')
    os.makedirs(d, exist_ok=True)
    pre = write_temp_prereg(os.path.join(SMOKE, 'prereg_smoke.json'))
    D, info = C.load_npz_info(C.TEST_NPZ)
    rgb = C.member_index(info)
    Z = dict(D)
    Z['hm_win'] = D['hm_win'][:, rgb]
    Z['kp_win'] = D['kp_win'][:, rgb]
    with open(os.path.join(d, 'kz_members_fake.npz'), 'wb') as f:
        np.savez(f, **Z)
    info2 = dict(info)
    info2.update({'member_names': list(C.RGB), 'split': 'train', 'sampled_interval': 1})
    C.jdump(info2, os.path.join(d, 'kz_members_fake.json'))
    P = pickle.load(open(C.TEST_DUMP, 'rb'))
    for f in P['frames']:
        f['gt'] = None
        f['matched'] = None
    P['info']['det_rate'] = None
    P['info']['split'] = 'train'
    pickle.dump(P, open(os.path.join(d, 'real_ens5flip_kpKZavg_zoom_fake.pkl'), 'wb'))
    del P
    for fn in ('adabn_frames_fake.json',):
        if os.path.exists(os.path.join(d, fn)):
            os.remove(os.path.join(d, fn))
    R, code = F.main(['--dir', d, '--tag', 'fake', '--prereg', pre, '--st1', os.path.join(d, 'none.pkl')])
    assert code == 0 and R['checks']['pass'], R['checks']
    V = json.load(open(os.path.join(d, 'gate_verdict_stage1.json'), encoding='utf-8'))
    assert V['gate']['phantom4']['pass'] and V['gate']['mavic2']['pass']
    assert V['final_disposition'] == 'no_LOO_sample_size', V['final_disposition']        # only 10 test sequences < 20
    Zf = np.load(os.path.join(d, 'frozen_train_stage1.npz'))
    assert len(Zf['sel_idx']) > 0 and Zf['loo_RT'].shape == (len(Zf['sel_idx']), 5, 3, 3)
    print('[t5] OK: freeze end-to-end on a fake train (= e1sa test, label-free): checks pass, gate pass on itself, '
          'disposition %s (10 sequences < 20), rho %s, selected %d, AdaBN list %s' % (
              V['final_disposition'], V['rho'], len(Zf['sel_idx']), V['adabn_frames']))


def fake_bn_npz(D, info, members, out_npz, prereg_path, smoke=True):
    """A stand-in for adabn_kz.py's output: given members of a member npz re-packed in the AdaBN schema, read cell and
    deployed vector recomputed with the hook's rule (float32 mean hm argmax in the window, mean kp / z)."""
    import hashlib
    S = D['hm_win'].shape[-1]
    search = (S - 1) // 2
    Z = {k: D[k] for k in D if k not in ('hm_win', 'kp_win', 'rb_cb', 'kp_deployed')}
    Z['hm_win'] = D['hm_win'][:, members].copy()
    Z['kp_win'] = D['kp_win'][:, members].copy()
    N = len(D['seq'])
    rb = np.zeros((N, 2), np.int64)
    kd = np.zeros((N, 16), np.float32)
    for i in range(N):
        mh = np.mean(Z['hm_win'][i], 0, dtype=np.float32)
        j = int(np.argmax(np.where(np.isfinite(mh), mh, -np.inf).reshape(-1)))
        rc, cc = D['rc_cc'][i]
        rb[i] = (rc - search + j // S, cc - search + j % S)
        kd[i] = (np.mean(Z['kp_win'][i][:, :, j // S, j % S], 0, dtype=np.float32) / np.float32(D['z'][i])).astype(np.float32)
    Z['rb_cb'], Z['kp_deployed'] = rb, kd
    with open(out_npz, 'wb') as f:
        np.savez(f, **Z)
    inf = {k: v for k, v in info.items() if k not in ('member_names', 'members')}
    inf.update({'member_names': list(C.RGB), 'arm': 'adabn', 'smoke': smoke,
                'prereg_sha256': hashlib.sha256(open(prereg_path, 'rb').read()).hexdigest()})
    C.jdump(inf, os.path.splitext(out_npz)[0] + '.json')
    return out_npz


def fake_frozen_train(d, rho, prereg_path, split='train'):
    import hashlib
    rho_arr = np.array([rho[c] for c in C.CLASSES])
    p = os.path.join(d, 'frozen_train_stage1.npz')
    with open(p, 'wb') as f:
        np.savez(f, rho=rho_arr, rho_classes=np.array(C.CLASSES), rho_agree_report_only=rho_arr * 1.05)
    C.jdump({'frozen_npz_sha256': C.sha256(p), 'prereg_sha256': hashlib.sha256(open(prereg_path, 'rb').read()).hexdigest(),
             'checks': {'pass': True}, 'split': split, 'selection': {'final_disposition': 'SMOKE'},
             'gate': {'verdict': {c: {'pass': False} for c in C.CLASSES}}}, os.path.join(d, 'frozen_train_stage1.json'))


def t6():
    """synthetic fixture (fabricated ground truth): kzst_stage1_eval freeze + evaluate end-to-end."""
    import kzst_stage1_eval as E
    t0 = time.time()
    pre = write_temp_prereg(os.path.join(SMOKE, 'prereg_smoke.json'))
    root = os.path.join(SMOKE, 'fixture_eval')
    if os.path.exists(root):
        shutil.rmtree(root)
    KFO.make_fixture(root, N=400, seed=11)
    D, info = C.load_npz_info(os.path.join(root, 'kz_members_synth.npz'))
    os.makedirs(os.path.join(root, 'adabn'), exist_ok=True)
    fake_bn_npz(D, info, [5, 6, 7, 8, 9], os.path.join(root, 'adabn', 'kz_members_adabn_test.npz'), pre)
    fake_frozen_train(root, {'phantom4': (0.95, 1.0), 'mavic2': (1.0, 0.9)}, pre)
    args = ['--dir', root, '--prereg', pre, '--test-npz', os.path.join(root, 'kz_members_synth.npz'),
            '--test-dump', os.path.join(root, 'real_ens5flip_kpKZavg_zoom_synth.pkl'), '--test-frozen', os.path.join(root, 'none.npz'),
            '--allow-smoke-bn', '--workers', '8']
    J = E.main(['freeze'] + args)
    assert J['have_bn']
    R = E.main(['evaluate'] + args)
    assert R['decision_BN'] in ('deploy_BN', 'pending_sim_guardrail', 'do_not_deploy_BN')
    assert set(R['P1_BN']['criteria']) == {'delta_le_-1', 'ci_upper_lt_0', 'seq_better_ge_7', 'paired_ge_4', 'holm'}
    assert os.path.exists(os.path.join(root, 'RESULTS_stage1.md'))
    print('[t6] OK: eval freeze + evaluate on a synthetic fixture (fabricated GT): A0 %.2f A0p %.2f BN %.2f; P1-BN %s; decision %s (%.0f s)' % (
        R['table']['A0']['all']['head'], R['table']['A0p']['all']['head'], R['table']['BN']['all']['head'], R['P1_BN']['criteria'],
        R['decision_BN'], time.time() - t0))


def t7():
    """LABEL-FREE on the real test inputs: eval 'freeze' only (never 'evaluate'): A0 == frozen_e1sa asserted inside,
    A0' with a smoke rho, a stand-in BN = the 5 multimodal KZ students (members 5..9), 126 multisets x 3 configs timed."""
    import kzst_stage1_eval as E
    t0 = time.time()
    pre = write_temp_prereg(os.path.join(SMOKE, 'prereg_smoke.json'))
    d = os.path.join(SMOKE, 'real_freeze')
    os.makedirs(os.path.join(d, 'adabn'), exist_ok=True)
    D, info = C.load_npz_info(C.TEST_NPZ)
    fake_bn_npz(D, info, [5, 6, 7, 8, 9], os.path.join(d, 'adabn', 'kz_members_adabn_test.npz'), pre)
    fake_frozen_train(d, {'phantom4': (0.6342, 0.7020), 'mavic2': (0.6178, 0.5146)}, pre, split='smoke-from-test')
    J = E.main(['freeze', '--dir', d, '--prereg', pre, '--allow-smoke-bn', '--allow-nontrain-rho', '--workers', '12'])
    print('[t7] OK: label-free eval freeze on the real test dump: A0 == frozen_e1sa (asserted), multisets identity == full arms '
          '(asserted), A0p changes heading>5/tilt>2 on %.3f of frames; %.0f s total (no ground truth read)' % (
              J['changed_A0p_vs_A0_heading_gt5_or_tilt_gt2'], time.time() - t0))


if __name__ == '__main__':
    todo = sys.argv[1:] or ['t1', 't2', 't3', 't4', 't5', 't6', 't7']
    for t in todo:
        globals()[t]()
