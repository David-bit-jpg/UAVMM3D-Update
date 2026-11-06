# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 1, S2 (CPU, LABEL-FREE -- never reads gt / matched): the pre-registered train-side gate.

Input : kz_members_<tag>.npz/.json + real_ens5flip_kpKZavg_zoom_<tag>.pkl from dump_kz_members_kzst.py (train split).
Steps : (1) replication checks of the dump (E5 style) + positions/conf identical to the ST1 train dump + g0 replica
        (2) pass A, deployed decoder: member PnPs, E5 strata, SA-TM -> gate statistics vs the frozen test reference,
            per-class verdict, disposition
        (3) rho_c per class from g0 frames (label-free span ratios)        -> frozen for A0' and for stage 2
        (4) pass B, D_rho decoder: strata (g1), SA-TM; selection g0 & g1 & trained class, <= 80 / sequence; sample sizes
        (5) LOO teachers (4-member SA-TM under D_rho) + quotient flags on the selected frames (stage-2 input, no training)
        (6) AdaBN frame list (g0, <= 80 / sequence, both classes) -- identical rule to adabn_kz.py (asserted)
Output: frozen_train_stage1.npz / .json (sha256), gate_verdict_stage1.json

    python kzst_stage1_freeze.py            (defaults: --tag train5, this folder)
Exit code 0 = gate decided (pass or not); 1 = replication check failed (stop and fix, nothing is decided).
"""
import sys

sys.dont_write_bytecode = True
import argparse      # noqa: E402
import collections   # noqa: E402
import os            # noqa: E402
import json          # noqa: E402
import pickle        # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

import kzst_common as C   # noqa: E402

KFO = C.KFO


def compute_pass(D, info, frames, T, rho_by_cls=None, idx=None, loo=False, rho_ratios=False, selfpeak=False, log_every=2000):
    """Per-frame arms for the frames in idx (default all). rho_by_cls: {cls: (rho_u, rho_v)} or None (deployed decoder)."""
    members = C.member_index(info)
    search = int(info.get('search', 2))
    ks = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    N = len(T['seq'])
    M = len(members)
    idx = np.arange(N) if idx is None else np.asarray(idx)
    out = {'top': np.full(N, -1, np.int64), 'stratum': np.full(N, 'no_det', dtype='<U12'),
           'R_satm': np.full((N, 3, 3), np.nan), 'R_c0': np.full((N, 3, 3), np.nan), 'member_R': np.full((N, M, 3, 3), np.nan),
           'fb_satm': np.zeros(N, bool), 'fb_c0': np.zeros(N, bool), 'cell_ok': np.zeros(N, bool),
           'pnp_dep_deg': np.full(N, np.nan), 'c0_vs_dep_deg': np.full(N, np.nan)}
    if rho_ratios:
        out['ratios'] = np.full((N, M, 2), np.nan)
    if selfpeak:
        out['R_single'] = np.full((N, M, 3, 3), np.nan)
    if loo:
        out['loo_RT'] = np.full((N, M, 3, 3), np.nan)
        out['loo_quotient'] = np.zeros((N, M), bool)
        out['loo_stratum'] = np.full((N, M), '', dtype='<U12')
        out['rigid_rms_rel'] = np.full(N, np.nan)
    t0 = time.time()
    for n_done, i in enumerate(idx):
        k = T['keys'][i]
        fr = frames[k]
        rho = None if rho_by_cls is None else rho_by_cls[T['cls'][i]]
        a = C.frame_arms(D, i, fr, members, search, ks, rho, selfpeak=selfpeak, loo=loo)
        if a is None:
            continue
        out['top'][i] = a['top']
        out['stratum'][i] = a['stratum']
        out['R_satm'][i], out['R_c0'][i] = a['SATM'], a['C0']
        out['fb_satm'][i], out['fb_c0'][i] = a['fb_SATM'], a['fb_C0']
        for m, Rm in enumerate(a['member_R']):
            if Rm is not None:
                out['member_R'][i, m] = Rm
        ctx = a['ctx']
        out['cell_ok'][i] = ctx.cell_for(members) == a['cell']
        if rho is None:
            Rp = KFO.pnp(D['kp_deployed'][i], ctx.g)
            if Rp is not None:
                out['pnp_dep_deg'][i] = KFO.geodesic(Rp, a['Rdep'])
            if not a['fb_C0']:
                out['c0_vs_dep_deg'][i] = KFO.geodesic(a['C0'], a['Rdep'])
        if rho_ratios:
            for m, rr in enumerate(C.rho_ratios(ctx, members, a['cell'], fr['pred'][a['top'], :3])):
                if rr is not None:
                    out['ratios'][i, m] = rr
        if selfpeak:
            for m, (R_, _) in enumerate(a['selfpeak']):
                out['R_single'][i, m] = R_
        if loo:
            for m in range(M):
                out['loo_RT'][i, m] = a['loo_RT'][m]
                out['loo_quotient'][i, m] = a['loo_quotient'][m]
                out['loo_stratum'][i, m] = a['loo_stratum'][m]
            if a['plan'] is not None:          # label-free rigid residual of the 5-member SA-TM vector (report only)
                xs = [ctx.x(m, a['cell']) for m in members]
                out['rigid_rms_rel'][i] = C.rigid_rms_rel(ctx, KFO.satm_vec(xs, a['plan']))
        if log_every and n_done % log_every == log_every - 1:
            print('   %d / %d frames, %.0f s' % (n_done + 1, len(idx), time.time() - t0), flush=True)
    return out


def st1_position_check(frames, keys, st1):
    """(iii) pred[:, :6] and conf identical to the ST1 train dump frame by frame; deployed rotation difference reported."""
    bad, rd = 0, []
    for k in keys:
        fo, fn = st1.get(k), frames[k]
        if fo is None or fn['pred'].shape != fo['pred'].shape or not np.array_equal(fn['pred'][:, :6], fo['pred'][:, :6]) \
                or not np.array_equal(fn['conf'], fo['conf']):
            bad += 1
            continue
        t = KFO.top_index(fn)
        if t >= 0:
            rd.append(KFO.geodesic(KFO.rot_from_eul(fn['pred'][t, 6:9]), KFO.rot_from_eul(fo['pred'][t, 6:9])))
    rd = np.asarray(rd)
    return {'iii_position_conf_mismatch_vs_ST1': bad, 'iii_deployed_rot_vs_ST1_max_deg': float(rd.max()) if len(rd) else None,
            'iii_deployed_rot_vs_ST1_frac_lt_0p05': float((rd < 0.05).mean()) if len(rd) else None}


def stage(a):
    pre, P = C.check_prereg(a.prereg)
    t0 = time.time()
    npz = os.path.join(a.dir, 'kz_members_%s.npz' % a.tag)
    pkl = os.path.join(a.dir, 'real_ens5flip_kpKZavg_zoom_%s.pkl' % a.tag)
    D, info = C.load_npz_info(npz)
    frames, dinfo = C.load_dump_nogt(pkl)
    if info.get('dump_finished_ok') is not True:
        raise SystemExit('the member dump did not finish (dump_finished_ok=%r) -- rerun S1' % info.get('dump_finished_ok'))
    split = info.get('split', dinfo.get('split'))
    interval = int(info.get('sampled_interval', 1))
    if split != a.expect_split:
        raise SystemExit('dump split %r != expected %r' % (split, a.expect_split))
    if C.GATE['gap_raw_frames'] % interval:
        raise SystemExit('interval %d does not divide the 2-frame gap' % interval)
    gap_ranks = C.GATE['gap_raw_frames'] // interval
    T = C.frame_table(D, frames)
    N = len(T['seq'])
    log = []
    R = {'prereg_sha256': pre, 'npz_sha256': C.sha256(npz), 'dump_sha256': C.sha256(pkl), 'split': split, 'sampled_interval': interval,
         'gap_ranks': gap_ranks, 'n_frames': N, 'script_sha256': C.sha256(os.path.abspath(__file__)),
         'common_sha256': C.sha256(os.path.join(C.HERE, 'kzst_common.py')), 'kfo_sha256': C.sha256(os.path.join(C.HERE, 'kz_fuse_offline.py')),
         'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    C.KFO.say('[freeze] %s: %d frames, %d sequences (%s), members %s, interval %d -> jump gap %d ranks' % (
        split, N, len(set(T['seq'])), dict(collections.Counter(T['cls'])), info.get('member_names'), interval, gap_ranks), log)

    # ---------------- pass A (deployed decoder) + replication checks
    A = compute_pass(D, info, frames, T, None, rho_ratios=True)
    has = A['top'] >= 0
    n_top = int(np.sum([KFO.top_index(frames[k]) >= 0 for k in T['keys']]))
    # E5 rule: frames where the deployed PnP failed (rot-head rotation kept) are counted, not compared
    pnp_ok = has & np.isfinite(A['pnp_dep_deg'])
    c0_ok = pnp_ok & np.isfinite(A['c0_vs_dep_deg'])
    chk = {'n_with_detection': n_top, 'n_top_at_hook_cell': int(has.sum()), 'n_top_not_hook_cell': int(n_top - has.sum()),
           'n_deployed_pnp_failed': int(has.sum() - pnp_ok.sum()), 'n_offline_C0_pnp_failed': int((has & A['fb_c0']).sum()),
           'i_pnp_on_deployed_vector_frac_lt_1e-3': float((A['pnp_dep_deg'][pnp_ok] < 1e-3).mean()),
           'i_pnp_on_deployed_vector_max_deg': float(A['pnp_dep_deg'][pnp_ok].max()),
           'i_offline_C0_frac_lt_0p05': float((A['c0_vs_dep_deg'][c0_ok] < 0.05).mean()),
           'v_cell_reselect_frac': float(A['cell_ok'][has].mean()), 'fallback_SATM': int(A['fb_satm'][has].sum()),
           'fallback_C0': int(A['fb_c0'][has].sum())}
    # (iii) positions / conf identical to the ST1 train dump of 09-24 (same deployed pipeline), when it covers the frames
    if os.path.exists(a.st1) and split == 'train' and interval == 1:      # interval 2 changes the temporal background
        chk.update(st1_position_check(frames, T['keys'], C.load_dump_nogt(a.st1)[0]))
    fails = []
    if chk['n_top_not_hook_cell'] > 0.005 * max(n_top, 1):
        fails.append('top detection != hook cell on %d frames (>0.5%%)' % chk['n_top_not_hook_cell'])
    if chk['n_deployed_pnp_failed'] > 0.005 * max(n_top, 1) or chk['n_offline_C0_pnp_failed'] > chk['n_deployed_pnp_failed'] + 0.001 * max(n_top, 1):
        fails.append('PnP failures: deployed %d, offline C0 %d' % (chk['n_deployed_pnp_failed'], chk['n_offline_C0_pnp_failed']))
    if chk['i_pnp_on_deployed_vector_frac_lt_1e-3'] < 0.999:
        fails.append('PnP replica on the deployed vector %.4f < 0.999' % chk['i_pnp_on_deployed_vector_frac_lt_1e-3'])
    if chk['i_offline_C0_frac_lt_0p05'] < 0.995:
        fails.append('offline C0 %.4f < 0.995' % chk['i_offline_C0_frac_lt_0p05'])
    if chk['v_cell_reselect_frac'] < 0.999:
        fails.append('cell re-selection %.4f < 0.999' % chk['v_cell_reselect_frac'])
    if chk.get('iii_position_conf_mismatch_vs_ST1'):
        fails.append('positions/conf differ from the ST1 train dump on %d frames' % chk['iii_position_conf_mismatch_vs_ST1'])
    chk['fails'] = fails
    chk['pass'] = not fails
    R['checks'] = chk
    C.KFO.say('[freeze] checks: %s' % {k: v for k, v in chk.items() if k != 'fails'}, log)
    if fails and not a.allow_check_fail:
        R['log'] = log
        C.jdump(R, os.path.join(a.dir, 'frozen_train_stage1_FAILED_checks.json'))
        C.KFO.say('[freeze] replication check FAILED: %s -- stop and fix (nothing decided)' % '; '.join(fails))
        return R, 1

    # ---------------- g0 (ST1 rule) and the gate
    g0, g0_reasons = C.g0_mask(T['seq'], T['fk'], T['conf_top'], T['pos_top'])
    g0h = g0 & has
    R['g0'] = {'reasons': g0_reasons, 'n_g0': int(g0.sum()), 'n_g0_at_hook_cell': int(g0h.sum()),
               'per_class': {c: int((g0h & (T['cls'] == c)).sum()) for c in C.CLASSES}}
    if split == 'train' and interval == 1 and os.path.exists(C.ST1_PSEUDO_INDEX):
        idx = pickle.load(open(C.ST1_PSEUDO_INDEX, 'rb'))
        acc = {(m['seq'], m['frame']) for m in idx['metas'] if m}          # keys only (selection was label-free)
        del idx
        mine = {T['keys'][i] for i in np.where(g0)[0]}
        R['g0']['equals_ST1_accepted_set'] = bool(mine == acc)
        R['g0']['ST1_accepted'] = len(acc)
    ref_g = json.load(open(C.TEST_REF_GATED, encoding='utf-8'))
    ref_l = json.load(open(C.TEST_REF_LABELFREE, encoding='utf-8'))
    bounds = C.gate_bounds(ref_g, ref_l)
    stats = C.gate_stats(T['cls'], T['seq'], T['fk'], A['stratum'], A['R_satm'], A['member_R'], gap_ranks=gap_ranks)
    verdict, disp = C.gate_verdict(stats, bounds)
    R['gate'] = {'bounds': bounds, 'stats_deployed_decoder': stats, 'verdict': verdict, 'disposition_by_gate': disp,
                 'reference_files': {'test_ref_gated.json': C.sha256(C.TEST_REF_GATED), 'test_ref_labelfree.json': C.sha256(C.TEST_REF_LABELFREE)}}
    for c in C.CLASSES:
        v = verdict[c]
        C.KFO.say('[gate] %-9s share %.3f (>= %.3f: %s) | SA-TM gap-2 jump %.2f (<= %.2f: %s) | pairwise %.2f (<= %.2f: %s) -> %s   [n gated %d, pairs %d]' % (
            c, v['values']['gated_share'], bounds[c]['share_min'], v['share'], v['values']['jump_median'], bounds[c]['jump_max'], v['jump'],
            v['values']['pairwise_fold_median'], bounds[c]['pairwise_max'], v['pairwise'], 'PASS' if v['pass'] else 'FAIL',
            v['values']['n_gated'], v['values']['n_jump_pairs']), log)

    # ---------------- rho_c (label-free) on g0 frames
    rho, rho_rep, rho_agree = {}, {}, {}
    for c in C.CLASSES:
        m = g0h & (T['cls'] == c)
        r = A['ratios'][m].reshape(-1, 2)
        r = r[np.isfinite(r).all(1)]
        if len(r) < 50:
            raise SystemExit('rho: only %d (frame, member) samples for %s' % (len(r), c))
        rho[c] = (float(np.median(r[:, 0])), float(np.median(r[:, 1])))
        rho_rep[c] = {'rho_u': rho[c][0], 'rho_v': rho[c][1], 'n_samples': int(len(r)), 'n_frames': int(m.sum()),
                      'iqr_u': [float(np.percentile(r[:, 0], 25)), float(np.percentile(r[:, 0], 75))],
                      'iqr_v': [float(np.percentile(r[:, 1], 25)), float(np.percentile(r[:, 1], 75))],
                      'by_stratum': {s: [float(np.nanmedian(A['ratios'][m & (A['stratum'] == s)][..., 0])),
                                         float(np.nanmedian(A['ratios'][m & (A['stratum'] == s)][..., 1]))]
                                     for s in ('agree', 'flip_only', 'trim_only', 'both') if (m & (A['stratum'] == s)).any()}}
        ra = A['ratios'][m & (A['stratum'] == 'agree')].reshape(-1, 2)
        ra = ra[np.isfinite(ra).all(1)]
        rho_agree[c] = (float(np.median(ra[:, 0])), float(np.median(ra[:, 1]))) if len(ra) >= 50 else (float('nan'), float('nan'))
        C.KFO.say('[rho] %-9s rho = (%.4f, %.4f) from %d samples / %d g0 frames | report-only agree-stratum rho (%.4f, %.4f) '
                  '(protocol expectation, taken from the test agree stratum: ~%s)' % (
                      c, rho[c][0], rho[c][1], len(r), m.sum(), rho_agree[c][0], rho_agree[c][1],
                      {'phantom4': '(0.72, 0.74)', 'mavic2': '(0.73, 0.52)'}[c]), log)
    R['rho'] = rho_rep
    R['rho_agree_report_only'] = rho_agree

    # ---------------- pass B (D_rho): g1 strata and selection
    B = compute_pass(D, info, frames, T, rho)
    stats_rho = C.gate_stats(T['cls'], T['seq'], T['fk'], B['stratum'], B['R_satm'], B['member_R'], gap_ranks=gap_ranks)
    R['report_gate_stats_under_D_rho'] = stats_rho
    trained = {'train_both': ['phantom4', 'mavic2'], 'train_phantom4_only': ['phantom4'], 'no_LOO': []}[disp]
    planned = trained if trained else ['phantom4']           # report-only "what would have been trained"
    elig = g0h & np.isin(B['stratum'], C.GATED) & np.isin(T['cls'], planned)
    sel = C.select_per_seq(elig, T['seq'], T['fk'], C.CAP_PER_SEQ)
    ss = C.sample_size_check(sel, T['cls'], T['seq'], planned)
    loo_go = bool(trained) and ss['pass']
    final = disp if loo_go else ('no_LOO' if not trained else 'no_LOO_sample_size')
    R['selection'] = {'trained_classes': trained, 'planned_classes_for_report': planned, 'n_eligible': int(elig.sum()),
                      'sample_size': ss, 'loo_go': loo_go, 'final_disposition': final,
                      'per_seq': {s: int((T['seq'][sel] == s).sum()) for s in sorted(set(T['seq'][sel]))}}
    C.KFO.say('[select] disposition by gate: %s | eligible %d -> selected %d (%s, %d seqs) | sample sizes %s -> LOO %s' % (
        disp, elig.sum(), len(sel), ss['per_class'], ss['n_seq'], ss['checks'], 'GO' if loo_go else 'NO'), log)
    # label-free reports: strata under both decoders, teacher view distribution before / after selection
    rep = {}
    for c in C.CLASSES:
        mc = T['cls'] == c
        rep[c] = {'n_dumped': int(mc.sum()), 'n_g0': int((g0h & mc).sum()), 'n_g1_rho_and_g0': int((elig & mc).sum()) if c in planned else None,
                  'n_selected': int((T['cls'][sel] == c).sum()),
                  'strata_deployed': {s: float((A['stratum'][mc] == s).mean()) for s in C.STRATA},
                  'strata_D_rho': {s: float((B['stratum'][mc] == s).mean()) for s in C.STRATA}}
        for tag, mm in (('teacher_view_g0', g0h & mc), ('teacher_view_selected', np.isin(np.arange(N), sel) & mc)):
            regs = [C.side_regime(C.hd(B['R_satm'][i])) for i in np.where(mm)[0] if np.isfinite(B['R_satm'][i]).all()]
            cnt = collections.Counter(regs)
            rep[c][tag] = {r_: (cnt[r_] / max(len(regs), 1)) for r_ in ('side', 'oblique', 'along')}
            rep[c][tag]['n'] = len(regs)
    R['report_label_free'] = rep

    # ---------------- LOO teachers on the selected frames (stage-2 input only)
    L = compute_pass(D, info, frames, T, rho, idx=sel, loo=True, log_every=0) if len(sel) else None
    if L is not None:
        rr = L['rigid_rms_rel'][sel]
        R['report_label_free']['rigid_residual_selected'] = {
            c: {'median_rms_over_width': float(np.nanmedian(rr[T['cls'][sel] == c])) if (T['cls'][sel] == c).any() else None}
            for c in C.CLASSES}
        R['report_label_free']['loo_quotient_frac'] = float(L['loo_quotient'][sel].mean())
        R['report_label_free']['loo_teacher_strata'] = dict(collections.Counter(L['loo_stratum'][sel].ravel().tolist()))
    # ---------------- AdaBN frame list (same rule as adabn_kz.py)
    ad_sel, _, g0_chk, _ = C.adabn_selection(D, frames)
    assert np.array_equal(g0_chk, g0)
    R['adabn_frames'] = {'n': int(len(ad_sel)), 'per_class': {c: int((T['cls'][ad_sel] == c).sum()) for c in C.CLASSES},
                         'n_seq': int(len(set(T['seq'][ad_sel])))}
    ad_json = os.path.join(a.dir, 'adabn_frames_%s.json' % a.tag)
    if os.path.exists(ad_json):
        prev = json.load(open(ad_json, encoding='utf-8'))
        R['adabn_frames']['equals_adabn_kz_list'] = bool(prev['keys'] == [list(T['keys'][i]) for i in ad_sel])
        assert R['adabn_frames']['equals_adabn_kz_list'], 'AdaBN frame list differs from the one adabn_kz.py used'

    # ---------------- write
    Z = {'seq': T['seq'], 'frame': T['frame'], 'cls': T['cls'], 'fk': T['fk'], 'top': A['top'], 'g0': g0, 'g0_hook': g0h,
         'conf_top': T['conf_top'], 'pos_top': T['pos_top'],
         'stratum': A['stratum'], 'R_satm': A['R_satm'], 'R_c0': A['R_c0'], 'member_R': A['member_R'], 'ratios': A['ratios'],
         'stratum_rho': B['stratum'], 'R_satm_rho': B['R_satm'], 'member_R_rho': B['member_R'],
         'sel_idx': sel, 'adabn_idx': ad_sel,
         'rho': np.array([rho[c] for c in C.CLASSES]), 'rho_classes': np.array(C.CLASSES),
         'rho_agree_report_only': np.array([rho_agree[c] for c in C.CLASSES])}
    if L is not None:
        Z.update({'loo_RT': L['loo_RT'][sel], 'loo_quotient': L['loo_quotient'][sel], 'loo_stratum': L['loo_stratum'][sel],
                  'loo_R_satm5_rho': L['R_satm'][sel]})
    fz = os.path.join(a.dir, 'frozen_train_stage1.npz')
    with open(fz + '.tmp', 'wb') as f:
        np.savez(f, **Z)
    os.replace(fz + '.tmp', fz)
    R['frozen_npz_sha256'] = C.sha256(fz)
    R['elapsed_s'] = time.time() - t0
    R['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    R['log'] = log
    C.jdump(R, os.path.join(a.dir, 'frozen_train_stage1.json'))
    V = {'prereg_sha256': pre, 'frozen_npz_sha256': R['frozen_npz_sha256'], 'gate': {c: {k: verdict[c][k] for k in ('share', 'jump', 'pairwise', 'pass', 'values', 'ratios_to_test')} for c in C.CLASSES},
         'bounds': bounds, 'disposition_by_gate': disp, 'sample_size': ss, 'final_disposition': final,
         'rho': {c: list(rho[c]) for c in C.CLASSES}, 'rho_agree_report_only': {c: list(rho_agree[c]) for c in C.CLASSES},
         'adabn_frames': R['adabn_frames'], 'checks_pass': chk['pass']}
    C.jdump(V, os.path.join(a.dir, 'gate_verdict_stage1.json'))
    C.KFO.say('[freeze] FINAL: %s ; rho %s ; wrote frozen_train_stage1.{npz,json}, gate_verdict_stage1.json (%.0f s)' % (
        final, {c: tuple(round(x, 4) for x in rho[c]) for c in C.CLASSES}, time.time() - t0))
    return R, 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=C.HERE)
    ap.add_argument('--tag', default=C.TRAIN_TAG)
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--st1', default=C.ST1_TRAIN_DUMP)
    ap.add_argument('--expect-split', default='train')
    ap.add_argument('--allow-check-fail', action='store_true', help='NOT for the real run')
    a = ap.parse_args(argv)
    R, code = stage(a)
    return R, code


if __name__ == '__main__':
    sys.exit(main()[1])
