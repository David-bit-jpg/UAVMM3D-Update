# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 2, S4 (CPU, LABEL-FREE -- never reads gt / matched): pseudo labels + deployed training crops.

Inputs (stage 1, sha256-checked): frozen_train_stage1.npz (rho_c, selected gated frames sel_idx, LOO teachers loo_RT /
loo_quotient / loo_stratum), kz_members_train5.npz/.json + the scrubbed train dump (positions, crop windows).

Per selected frame i and student k (= kz_s{k}):
  * the LOO teacher is RE-COMPUTED with the stage-1 rule (other 4 members, D_rho, their own read cell, SA-TM) and
    asserted bit-identical to the frozen loo_RT / loo_quotient / loo_stratum; the fallback level is recorded;
  * target  g'(k) = diag(rho_c) . g(R_T(k))  in the deployed crop's input pixels / 64, with g(R) = projected corners of
    the rigid box (R, t = position-network centre, lwh = deployed PnP box) relative to the projected centre;
    in full-input units this is exactly z . diag(rho_c) . x16 with D_rho(x16) -> PnP -> R_T (asserted < 0.01 deg both
    with the crop intrinsics and with the deployed decoder geometry KFO.Geo);
  * valid(i, k) = LOO SA-TM did not fall back (the teacher exists) AND the box centre projects inside the crop AND both
    round trips < 0.01 deg.  Invalid pairs are dropped from student k's training list (counts in the json).
Crops: the deployed zoom-95 window recorded in the train dump, re-cut from the RAW 1920x1080 image with the hook's
statements (kzst1/adabn_kz.recut = kp_zoom.RawSource remap + cv2.warpAffine INTER_AREA / INTER_LINEAR), stored as
uint8 BGR (N, 288, 512, 3) -> data/crops_train_u8.npy (lossless; exactly the pixels the deployed KZ saw).

Writes ONLY under @@ROOT@@/output/camnorm/night1003/kzst2/data (or --out-dir under kzst2/ for smoke tests):
  pseudo_stage2.npz, crops_train_u8.npy, crops_train.json, pseudo_stage2.json (written LAST = success marker).
Idempotent: if the marker exists and every recorded sha256 matches, prints 'already built' and exits 0.

    python kzst2_build.py                      (real run, ~3-5 min CPU)
    python kzst2_build.py --max-frames 40 --out-dir <kzst2>/smoke/data --prereg <smoke prereg>    (smoke)
"""
import sys

sys.dont_write_bytecode = True
import argparse      # noqa: E402
import collections   # noqa: E402
import concurrent.futures as cf   # noqa: E402
import json          # noqa: E402
import os            # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

import kzst2_common as S   # noqa: E402

C1, KFO = S.C1, S.KFO


def build_pseudo(a, log):
    Z = np.load(S.STAGE1['frozen_train'])
    rho_by = {str(c): (float(r[0]), float(r[1])) for c, r in zip(Z['rho_classes'], Z['rho'])}
    sel = Z['sel_idx']
    if a.max_frames:
        sel = sel[np.linspace(0, len(sel) - 1, a.max_frames).astype(int)]
    pos_in_frozen = {int(v): j for j, v in enumerate(Z['sel_idx'])}
    D, info = C1.load_npz_info(S.STAGE1['train_npz'])
    if info.get('split') != 'train' or info.get('dump_finished_ok') is not True:
        raise SystemExit('the stage-1 member dump is not a finished TRAIN dump')
    frames, _ = C1.load_dump_nogt(S.STAGE1['train_dump'])           # gt / matched dropped at load (and scrubbed in the file)
    members = C1.member_index(info)
    search = int(info.get('search', 2))
    ks = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    assert ks == S.KP_SCALE and tuple(info['in_wh']) == S.IN_WH
    N, M = len(sel), len(members)
    out = {'src_idx': sel.astype(np.int64), 'seq': np.array([str(D['seq'][i]) for i in sel]),
           'frame': np.array([str(D['frame'][i]) for i in sel]), 'cls': np.empty(N, dtype='<U8'),
           'crop': D['crop'][sel].astype(np.float64), 'z': D['z'][sel].astype(np.float64),
           'K_raw': D['K_raw_src'][sel].astype(np.float64), 'raw_wh': D['raw_img_wh'][sel].astype(np.int64),
           'K_in': D['K_in'][sel].astype(np.float64),
           'K_crop': np.zeros((N, 3, 3)), 't': np.zeros((N, 3)), 'lwh': np.zeros((N, 3)), 'uv_full': np.zeros((N, 2)),
           'uv_crop': np.zeros((N, 2)), 'center_in_crop': np.zeros(N, bool), 'rho': np.zeros((N, 2)),
           'R_T': np.full((N, M, 3, 3), np.nan), 'quotient': np.zeros((N, M), bool), 'loo_stratum': np.full((N, M), '', dtype='<U12'),
           'fb_satm': np.zeros((N, M), bool), 'fb_level': np.full((N, M), '', dtype='<U5'),
           'target16': np.full((N, M, 16), np.nan), 'rt_crop_deg': np.full((N, M), np.nan), 'rt_dec_deg': np.full((N, M), np.nan),
           'valid': np.zeros((N, M), bool)}
    t0 = time.time()
    n_mismatch = 0
    for n, i in enumerate(sel):
        key = (str(D['seq'][i]), str(D['frame'][i]))
        fr = frames[key]
        c = str(fr['cls'])
        out['cls'][n] = c
        rho = rho_by[c]
        L = S.loo_teacher(D, i, fr, members, search, ks, rho)
        assert L is not None, 'a selected frame has no hook detection'
        # bit-identical to the frozen stage-1 LOO teacher
        j = pos_in_frozen[int(i)]
        for k in range(M):
            same = (np.array_equal(L['RT'][k], Z['loo_RT'][j, k]) and bool(L['quotient'][k]) == bool(Z['loo_quotient'][j, k])
                    and L['stratum'][k] == str(Z['loo_stratum'][j, k]))
            n_mismatch += int(not same)
        t = fr['pred'][L['top'], :3].astype(np.float64)
        lwh = fr['pred'][L['top'], 3:6].astype(np.float64)
        g = L['ctx'].g
        x0, y0, w, h, s = [float(v) for v in D['crop'][i]]
        assert abs(s - S.IN_WH[0] / w) < 1e-9 and abs(h - w * S.IN_WH[1] / S.IN_WH[0]) < 1e-6, 'crop not isotropic (sx != sy)'
        Kc = S.crop_K(D['K_raw_src'][i], x0, y0, s)
        # the crop camera and the decoder camera are the same affine family: offsets scale by exactly z
        assert np.allclose(S.proj(D['K_in'][i], t)[0], g.uv, atol=1e-6), 'proj(t) != decoder centre uv'
        uvc = S.proj(Kc, t)[0]
        out['K_crop'][n], out['t'][n], out['lwh'][n], out['uv_full'][n], out['uv_crop'][n] = Kc, t, lwh, g.uv, uvc
        out['rho'][n] = rho
        inside = bool(0 <= uvc[0] < S.IN_WH[0] and 0 <= uvc[1] < S.IN_WH[1])
        out['center_in_crop'][n] = inside
        r16 = C1.rho16(rho)
        for k in range(M):
            RT = L['RT'][k]
            out['R_T'][n, k] = RT
            out['quotient'][n, k] = L['quotient'][k]
            out['loo_stratum'][n, k] = L['stratum'][k]
            out['fb_satm'][n, k] = L['fb_satm'][k]
            out['fb_level'][n, k] = L['level'][k]
            tc = S.g16(Kc, t, lwh, RT) * r16                                  # g' in crop-input px / 64
            tf = S.g16(D['K_in'][i], t, lwh, RT)                              # g(R_T) in full-input px / 64
            assert np.allclose(tc, float(D['z'][i]) * r16 * tf, rtol=1e-7, atol=1e-9), 'crop offsets != z x full offsets'
            out['target16'][n, k] = tc
            Rc = S.decode_crop(tc, rho, Kc, t, lwh)
            out['rt_crop_deg'][n, k] = KFO.geodesic(Rc, RT) if Rc is not None else np.inf
            # deployed decoder path: the raw crop-map value tc (float32, as the network would emit it) -> x = kp / z ->
            # D_rho (/ rho16) -> KFO.pnp on the decoder geometry (centre uv, K_in) = what the deployed pipeline would do
            ctx = C1.RhoCtx(np.ones((1, 1, 1), np.float32), tc.astype(np.float32).reshape(1, 16, 1, 1), float(D['z'][i]), g, rho)
            Rd = ctx.R(0, (0, 0))
            out['rt_dec_deg'][n, k] = KFO.geodesic(Rd, RT) if Rd is not None else np.inf
            out['valid'][n, k] = bool((not L['fb_satm'][k]) and inside and out['rt_crop_deg'][n, k] < S.MAX_ROUNDTRIP_DEG
                                      and out['rt_dec_deg'][n, k] < S.MAX_ROUNDTRIP_DEG)
        if a.log_every and n % a.log_every == a.log_every - 1:
            print('   %d / %d frames, %.0f s' % (n + 1, N, time.time() - t0), flush=True)
    if n_mismatch:
        raise SystemExit('re-computed LOO teacher differs from frozen_train_stage1.npz on %d (frame, student) pairs' % n_mismatch)
    rt_c = out['rt_crop_deg'][np.isfinite(out['rt_crop_deg'])]
    rt_d = out['rt_dec_deg'][np.isfinite(out['rt_dec_deg'])]
    rep = {'n_frames': N, 'n_seq': int(len(set(out['seq']))), 'per_class': dict(collections.Counter(out['cls'].tolist())),
           'loo_teacher_bit_identical_to_stage1': True, 'n_pairs': N * M,
           'fallback_levels': dict(collections.Counter(out['fb_level'].ravel().tolist())),
           'center_outside_crop': int((~out['center_in_crop']).sum()),
           'roundtrip_crop_deg_max': float(rt_c.max()) if len(rt_c) else None, 'roundtrip_dec_deg_max': float(rt_d.max()) if len(rt_d) else None,
           'roundtrip_fail_pairs': int(((out['rt_crop_deg'] >= S.MAX_ROUNDTRIP_DEG) | (out['rt_dec_deg'] >= S.MAX_ROUNDTRIP_DEG)).sum()),
           'valid_per_student': [int(out['valid'][:, k].sum()) for k in range(M)],
           'quotient_frac_per_student': [float(out['quotient'][out['valid'][:, k], k].mean()) for k in range(M)],
           'loo_strata': dict(collections.Counter(out['loo_stratum'].ravel().tolist())),
           'iters_per_student_3_epochs_batch8': [int(3 * np.ceil(out['valid'][:, k].sum() / 8.0)) for k in range(M)],
           'z_quantiles': [float(q) for q in np.percentile(out['z'], [0, 25, 50, 75, 100])],
           'target_span_crop_px_median': float(np.median(np.ptp(out['target16'][..., 0::2], axis=-1)[out['valid']] * S.KP_SCALE)),
           'rho': rho_by}
    rep['max_iters'] = int(max(rep['iters_per_student_3_epochs_batch8']))
    assert rep['max_iters'] <= S.TRAIN['iter_cap'], 'iteration cap exceeded'
    KFO.say('[build] pseudo labels: %d frames x %d students; LOO teacher == stage 1 (bit-identical); fallback %s; centre outside crop %d; '
            'round trip max crop %.1e / decoder %.1e deg; valid per student %s; quotient frac %s; iters %s (%.0f s)' % (
                N, M, rep['fallback_levels'], rep['center_outside_crop'], rep['roundtrip_crop_deg_max'] or -1, rep['roundtrip_dec_deg_max'] or -1,
                rep['valid_per_student'], [round(x, 3) for x in rep['quotient_frac_per_student']], rep['iters_per_student_3_epochs_batch8'],
                time.time() - t0), log)
    return out, rep, D


def build_crops(a, out, D, crops_path, log):
    import adabn_kz as AB            # kzst1: recut() = the deployed hook's crop statements, NpzSource = kp_zoom.RawSource
    src = AB.NpzSource(D)
    N = len(out['src_idx'])
    arr = np.lib.format.open_memmap(crops_path + '.tmp.npy', mode='w+', dtype=np.uint8, shape=(N, S.IN_WH[1], S.IN_WH[0], 3))
    t0 = time.time()

    def one(n):
        i = int(out['src_idx'][n])
        arr[n] = AB.recut(D, i, src, S.IN_WH)
        return n
    with cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        for j, _ in enumerate(ex.map(one, range(N))):
            if a.log_every and j % (4 * a.log_every) == 4 * a.log_every - 1:
                print('   crops %d / %d, %.0f s' % (j + 1, N, time.time() - t0), flush=True)
    arr.flush()
    del arr
    os.replace(crops_path + '.tmp.npy', crops_path)
    KFO.say('[build] %d deployed crops re-cut from the raw images -> %s (%.0f s)' % (N, crops_path, time.time() - t0), log)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=S.DATA)
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--max-frames', type=int, default=0, help='smoke tests only')
    ap.add_argument('--threads', type=int, default=6, help='JPEG decode threads')
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    out_dir = S.ensure_under_out(os.path.abspath(a.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    pre, _ = S.check_prereg(a.prereg)
    npz_p, crops_p = os.path.join(out_dir, 'pseudo_stage2.npz'), os.path.join(out_dir, 'crops_train_u8.npy')
    crops_j, marker = os.path.join(out_dir, 'crops_train.json'), os.path.join(out_dir, 'pseudo_stage2.json')
    if os.path.exists(marker) and not a.force:
        J = json.load(open(marker, encoding='utf-8'))
        ok = (J.get('prereg_sha256') == pre and os.path.exists(npz_p) and S.sha256(npz_p) == J['npz_sha256']
              and os.path.exists(crops_p) and S.sha256(crops_p) == J['crops_sha256'])
        if ok:
            print('[build] already built and verified (%s)' % marker, flush=True)
            return J
        raise SystemExit('%s exists but does not verify (prereg / sha256) -- rerun with --force' % marker)
    for p in (marker, crops_j):
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()
    log = []
    s1 = S.check_stage1_inputs(('prereg_stage1.json', 'frozen_train_stage1.npz', 'kz_members_train5.npz', 'real_ens5flip_kpKZavg_zoom_train5.pkl'))
    out, rep, D = build_pseudo(a, log)
    with open(npz_p + '.tmp', 'wb') as f:
        np.savez(f, **out)
    os.replace(npz_p + '.tmp', npz_p)
    build_crops(a, out, D, crops_p, log)
    cj = {'prereg_sha256': pre, 'shape': [int(len(out['src_idx'])), S.IN_WH[1], S.IN_WH[0], 3], 'dtype': 'uint8', 'channel_order': 'BGR',
          'rows': 'same order as pseudo_stage2.npz', 'recut': 'kzst1/adabn_kz.recut (kp_zoom hook statements) on kp_zoom.RawSource',
          'adabn_kz_sha256': S.sha256(os.path.join(S.KZST1, 'adabn_kz.py')), 'crops_sha256': S.sha256(crops_p)}
    S.jdump(cj, crops_j)
    J = {'prereg_sha256': pre, 'stage1_inputs_sha256': s1, 'npz': os.path.basename(npz_p), 'npz_sha256': S.sha256(npz_p),
         'crops': os.path.basename(crops_p), 'crops_sha256': cj['crops_sha256'], 'smoke': bool(a.max_frames),
         'script_sha256': S.sha256(os.path.abspath(__file__)), 'common_sha256': S.sha256(os.path.join(S.HERE, 'kzst2_common.py')),
         'stage1_common_sha256': S.sha256(os.path.join(S.KZST1, 'kzst_common.py')), 'report_label_free': rep,
         'elapsed_s': time.time() - t0, 'written_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'log': log}
    S.jdump(J, marker)
    print('[build] done: %s (%.0f s)' % (marker, time.time() - t0), flush=True)
    return J


if __name__ == '__main__':
    main()
