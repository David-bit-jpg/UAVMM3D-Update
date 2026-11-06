# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 2, S6 (GPU, inference only, no labels): the 5 LOO students on the real MAV6D TEST split.

Same deployment pipeline as E5's dump with only the KZ weights replaced: the position network, detections, temporal
background, z and crop windows of every test frame are those recorded in e1sa/kz_members_e1sa.npz (the deployed dump);
the KZ input crop depends only on them, so it is re-cut from the RAW image with the hook's statements
(kzst1/adabn_kz.recut) and the 5 students are forwarded (batch 1, as in the hook). Recorded per frame, e1sa schema:
5x5 hm / kp2d windows of every student, read cell = argmax of the 5-student mean hm in the window, deployed vector =
mean kp2d / z (hook statements). Positions / detections / crops are copied from e1sa and asserted identical.
Replay check first (hard stop): the ORIGINAL kz_s{k} on the re-cut crops reproduce the recorded e1sa member windows
(GPU tolerance kp 2e-3, hm 2e-4; read cell may differ on <= max(1, n//50) of 24 frames) -- identical to stage 1.

Output (only under kzst2/test/): kz_members_loo_test.npz, kz_members_loo_test.json (written last = success marker).

    cd @@ROOT@@\\tools
    python <scratchpad>\\kzst2\\kzst2_test_infer.py
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import argparse      # noqa: E402
import concurrent.futures as cf   # noqa: E402
import json          # noqa: E402
import time          # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:      # noqa: BLE001
    pass
import numpy as np   # noqa: E402

import kzst2_common as S   # noqa: E402

C1, KFO = S.C1, S.KFO
STUDENT_NAMES = ['LOOs%d' % k for k in range(5)]


def student_ckpts(train_root, allow_smoke=False):
    out, info = [], []
    for k in range(5):
        dj = os.path.join(train_root, 'LOO_s%d' % k, 'DONE.json')
        J = json.load(open(dj, encoding='utf-8'))
        if J.get('smoke') and not allow_smoke:
            raise SystemExit('%s is a SMOKE training' % dj)
        if S.sha256(J['ckpt']) != J['ckpt_sha256']:
            raise SystemExit('student %d checkpoint changed after training' % k)
        out.append(J['ckpt'])
        info.append({'student': k, 'ckpt': J['ckpt'], 'ckpt_sha256': J['ckpt_sha256'], 'iterations': J['iterations'],
                     'n_frames': J['n_frames'], 'smoke': J.get('smoke', False)})
    return out, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--train-root', default=S.TRAIN_DIR)
    ap.add_argument('--out-dir', default=S.TEST_DIR)
    ap.add_argument('--test-npz', default=S.TEST_NPZ)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--verify-n', type=int, default=24)
    ap.add_argument('--workers', type=int, default=6, help='JPEG decode threads')
    ap.add_argument('--max-test-frames', type=int, default=0, help='smoke tests only')
    ap.add_argument('--allow-smoke-students', action='store_true', help='smoke tests only')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    pre, _ = S.check_prereg(a.prereg)
    out = S.ensure_under_out(os.path.abspath(a.out_dir))
    os.makedirs(out, exist_ok=True)
    out_npz = os.path.join(out, 'kz_members_loo_test.npz')
    out_json = os.path.splitext(out_npz)[0] + '.json'
    if os.path.exists(out_json) and not a.force:
        print('[test] %s exists (use --force)' % out_json, flush=True)
        return out_npz
    if os.path.exists(out_json):
        os.remove(out_json)
    t00 = time.time()
    import torch
    for p in (S.ROOT, S.TOOLS):
        if p not in sys.path:
            sys.path.insert(0, p)
    import adabn_kz as AB                                  # kzst1: recut / forward / window / verify_replay / build_models
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        assert torch.cuda.is_available()
    smoke = bool(a.max_test_frames or a.allow_smoke_students)
    ckpts, cinfo = student_ckpts(a.train_root, a.allow_smoke_students)
    Dte, ite = C1.load_npz_info(a.test_npz)
    nm = np.asarray(ite['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(ite['norm_std'], np.float32).reshape(3, 1, 1)
    assert list(ite['norm_mean']) == S.TRAIN['norm_mean'] and list(ite['norm_std']) == S.TRAIN['norm_std']
    search = int(ite.get('search', 2))
    src_te = AB.NpzSource(Dte)
    te_members = C1.member_index(ite)
    tol = (2e-3, 2e-4) if dev.type == 'cuda' else (2e-2, 2e-3)
    nte = len(Dte['seq']) if not a.max_test_frames else a.max_test_frames
    # ---- replay check with the ORIGINAL release KZ (crops reproduce the deployed member windows)
    orig, _, _, in_wh = AB.build_models(torch, dev, S.RELEASE_KZ)
    assert tuple(in_wh) == S.IN_WH == tuple(ite['in_wh'])
    vi = np.linspace(0, nte - 1, max(1, min(a.verify_n, nte))).astype(int)
    vr = AB.verify_replay(Dte, vi, te_members, orig, src_te, in_wh, nm, ns, search, torch, dev, tol[0], tol[1], 'test', a.workers)
    del orig
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    if not vr['pass']:
        raise SystemExit('replay check failed -- the re-cut crops do not reproduce the recorded member outputs; stop')
    # ---- students
    models, _, _, _ = AB.build_models(torch, dev, ckpts)
    S_ = 2 * search + 1
    HM = np.full((nte, 5, S_, S_), np.nan, np.float32)
    KP = np.full((nte, 5, 16, S_, S_), np.nan, np.float32)
    RB = np.zeros((nte, 2), np.int64)
    KD = np.zeros((nte, 16), np.float32)
    t0 = time.time()
    chunk = 64
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for j0 in range(0, nte, chunk):
            idx = list(range(j0, min(nte, j0 + chunk)))
            crops = list(ex.map(lambda i: AB.recut(Dte, i, src_te, in_wh), idx))
            for i, crop in zip(idx, crops):
                outs = AB.forward(models, AB.to_input(crop, nm, ns)[None], torch, dev)
                HM[i], KP[i], RB[i], KD[i] = AB.window(Dte, i, [o[0][0] for o in outs], [o[1][0] for o in outs], search, torch)
            if (j0 // chunk) % 10 == 9:
                print('[test]   %d / %d frames, %.0f s' % (j0 + len(idx), nte, time.time() - t0), flush=True)
    keep = ('seq', 'frame', 'pos_rc', 'center_res', 'hm_peak', 'u', 'v', 'w2d', 'z', 'crop', 'uc_vc', 'rc_cc', 'win', 'raw_img_wh',
            'K_raw_src', 'K_in', 'dist', 'raw_size', 'new_size', 'stride_b', 'known_size')
    Z = {k: Dte[k][:nte] for k in keep}
    Z.update({'hm_win': HM, 'kp_win': KP, 'rb_cb': RB, 'kp_deployed': KD})
    with open(out_npz + '.tmp', 'wb') as f:
        np.savez(f, **Z)
    os.replace(out_npz + '.tmp', out_npz)
    info = {k: ite[k] for k in ('n_rgb', 'search', 'target_w2d', 'z_range', 'stride', 'in_wh', 'KP_SCALE', 'W', 'H', 'dec_kwargs',
                               'KNOWN_SIZE_at_install', 'SIZE_SOURCE_at_install', 'norm_mean', 'norm_std') if k in ite}
    info.update({'member_names': STUDENT_NAMES, 'arm': 'loo', 'members': [{'name': n, 'path': p, 'role': 'loo_student'} for n, p in zip(STUDENT_NAMES, ckpts)],
                 'students': cinfo, 'n_frames': int(nte), 'npz': os.path.basename(out_npz), 'npz_sha256': S.sha256(out_npz),
                 'kp2d_dtype': ['torch.float32'], 'device': str(dev), 'prereg_sha256': pre, 'smoke': smoke,
                 'positions_crops_from': os.path.basename(a.test_npz), 'positions_crops_from_sha256': S.sha256(a.test_npz),
                 'replay_check': vr, 'script_sha256': S.sha256(os.path.abspath(__file__)),
                 'elapsed_s': time.time() - t00, 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    S.jdump(info, out_json)
    print('[test] wrote %s (%d test frames, 5 students) in %.0f s' % (out_npz, nte, time.time() - t00), flush=True)
    return out_npz


if __name__ == '__main__':
    main()
