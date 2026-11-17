# -*- coding: utf-8 -*-
"""Experiments R / C (GPU, inference only): forward a list of KZ checkpoints (default: the 5 AdaBN members of stage 1) on
EXACTLY the crops recorded by a member dump of dump_members_repl.py -- the stage-1 method (kzst1/adabn_kz.py step 4) that
produced the 9.06 deg test result, generalised to any source dump and any checkpoint list.

  * crops: the recorded zoom-95 window (x0, y0, w, h, s) re-cut from the undistorted raw image with the hook's statements
    (./adabn_kz.py recut: kp_zoom.RawSource remap, cv2.warpAffine INTER_AREA / INTER_LINEAR, x/255, (x - mean) / std);
  * replay check BEFORE anything else: the ORIGINAL release KZ members on re-cut crops must reproduce the windows recorded
    in the source npz (tolerances of adabn_kz.py: GPU 2e-3 kp / 2e-4 hm; CPU smoke 2e-2 / 2e-3) -- hard stop otherwise;
  * output read cell = argmax of the mean hm of the NEW members in the recorded 5x5 window (hook statement, torch),
    deployed vector = mean kp2d / z; positions, detections, crops and geometry are COPIED from the source npz, so they
    are identical by construction (asserted again by the evaluation scripts).

Writes ONLY into @@ROOT@@/output/camnorm/night1003/repl/<tag>/ :  kz_members_<tag>.npz / .json

    cd @@ROOT@@/tools
    python <repl>/replay_members_repl.py --src-tag val_kz --tag val_adabn            (experiment R, dump 2)
    python <repl>/replay_members_repl.py --src-tag st2_test_kz --tag st2_test_adabn  (experiment C, dump 2)
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '@@ROOT@@'
TOOLS = ROOT + '/tools'
for _p in (TOOLS, ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:      # noqa: BLE001
    pass

import argparse      # noqa: E402
import concurrent.futures as cf   # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

import repl_common as RC   # noqa: E402
import adabn_kz as AK      # noqa: E402  (byte copy of kzst1/adabn_kz.py; only its functions are used)

C = RC.C
NAMES = ['KZs%d' % k for k in range(5)]
KEEP = ('seq', 'frame', 'pos_rc', 'center_res', 'hm_peak', 'u', 'v', 'w2d', 'z', 'crop', 'uc_vc', 'rc_cc', 'win', 'raw_img_wh',
        'K_raw_src', 'K_in', 'dist', 'raw_size', 'new_size', 'stride_b', 'known_size')
INFO_KEEP = ('n_rgb', 'search', 'target_w2d', 'z_range', 'stride', 'in_wh', 'KP_SCALE', 'W', 'H', 'dec_kwargs',
             'KNOWN_SIZE_at_install', 'SIZE_SOURCE_at_install', 'norm_mean', 'norm_std', 'split', 'sampled_interval', 'pos_set',
             'pos_ckpts', 'dump_pkl', 'sealed_gt', 'sealed_gt_sha256', 'dump_pkl_sha256_labelfree')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-tag', default=None, help='source dump tag under the output root (dump_members_repl.py --tag)')
    ap.add_argument('--src-npz', default=None, help='explicit source member npz (self-tests)')
    ap.add_argument('--ckpts', nargs=5, default=None, help='default: the 5 stage-1 AdaBN members')
    ap.add_argument('--arm', default='adabn')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-root', default=RC.OUT_ROOT)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--verify-n', type=int, default=24)
    ap.add_argument('--workers', type=int, default=6, help='JPEG decode threads')
    ap.add_argument('--max-frames', type=int, default=0, help='smoke tests only')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    t00 = time.time()
    if not RC.under_out_root(a.out_root):
        raise SystemExit('refusing to write outside %s' % RC.OUT_ROOT)
    byte = RC.check_byte_copies()
    src_npz = a.src_npz or RC.tag_paths(a.src_tag, a.out_root)['npz']
    P = RC.tag_paths(a.tag, a.out_root)
    os.makedirs(P['dir'], exist_ok=True)
    if os.path.exists(P['npz']) and not a.force:
        raise SystemExit('%s exists (use --force)' % P['npz'])
    ckpts = list(a.ckpts) if a.ckpts else list(RC.ADABN_CKPTS)
    missing = [p for p in ckpts + RC.RELEASE_KZ + [src_npz] if not os.path.exists(p)]
    if missing:
        raise SystemExit('missing inputs: %s' % missing)
    import torch
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        assert torch.cuda.is_available()
    smoke = bool(a.max_frames)

    D, info = C.load_npz_info(src_npz)
    if info.get('dump_finished_ok') is not True:
        raise SystemExit('source dump %s did not finish' % src_npz)
    N = len(D['seq']) if not a.max_frames else min(a.max_frames, len(D['seq']))
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    search = int(info.get('search', 2))
    src = AK.NpzSource(D)

    # ---------------- replay check with the ORIGINAL release KZ members (hard stop)
    ref, _, _, in_wh = AK.build_models(torch, dev, RC.RELEASE_KZ)
    assert list(in_wh) == list(info['in_wh']), (in_wh, info['in_wh'])
    src_members = C.member_index(info)
    src_paths = {m['name']: m['path'] for m in info.get('members', [])}
    if src_paths:
        assert [os.path.normcase(os.path.abspath(src_paths[n])) for n in NAMES] == [os.path.normcase(os.path.abspath(p)) for p in RC.RELEASE_KZ], \
            'the source npz was not recorded with the release KZ members -- the replay check would be meaningless'
    tol = (2e-3, 2e-4) if dev.type == 'cuda' else (2e-2, 2e-3)
    vi = np.linspace(0, N - 1, max(1, min(a.verify_n, N))).astype(int)
    vr = AK.verify_replay(D, vi, src_members, ref, src, in_wh, nm, ns, search, torch, dev, tol[0], tol[1], 'source', a.workers)
    if not vr['pass']:
        raise SystemExit('replay check failed -- the re-cut crops do not reproduce the recorded member outputs; stop')
    del ref
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    # ---------------- forward the new members on the recorded crops
    models, _, _, in_wh2 = AK.build_models(torch, dev, ckpts)
    assert list(in_wh2) == list(in_wh)
    S = 2 * search + 1
    HM = np.full((N, 5, S, S), np.nan, np.float32)
    KP = np.full((N, 5, 16, S, S), np.nan, np.float32)
    RB = np.zeros((N, 2), np.int64)
    KD = np.zeros((N, 16), np.float32)
    t0 = time.time()
    chunk = 64
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for j0 in range(0, N, chunk):
            idx = list(range(j0, min(N, j0 + chunk)))
            crops = list(ex.map(lambda i: AK.recut(D, i, src, in_wh), idx))
            for i, crop in zip(idx, crops):
                outs = AK.forward(models, AK.to_input(crop, nm, ns)[None], torch, dev)
                HM[i], KP[i], RB[i], KD[i] = AK.window(D, i, [o[0][0] for o in outs], [o[1][0] for o in outs], search, torch)
            if (j0 // chunk) % 10 == 9:
                print('[replay]   %d / %d frames, %.0f s' % (j0 + len(idx), N, time.time() - t0), flush=True)
    Z = {k: D[k][:N] for k in KEEP}
    Z.update({'hm_win': HM, 'kp_win': KP, 'rb_cb': RB, 'kp_deployed': KD})
    with open(P['npz'] + '.tmp', 'wb') as f:
        np.savez(f, **Z)
    os.replace(P['npz'] + '.tmp', P['npz'])
    out = {k: info[k] for k in INFO_KEEP if k in info}
    out.update({'member_names': list(NAMES), 'arm': a.arm, 'members': [{'name': n, 'path': p, 'role': a.arm} for n, p in zip(NAMES, ckpts)],
                'n_frames': int(N), 'npz': os.path.basename(P['npz']), 'npz_sha256': RC.sha256(P['npz']),
                'kp2d_dtype': ['torch.float32'], 'device': str(dev), 'smoke': smoke, 'dump_finished_ok': True, 'replay_finished_ok': True,
                'positions_crops_from': os.path.abspath(src_npz), 'positions_crops_from_sha256': RC.sha256(src_npz),
                'source_member_names': info.get('member_names'), 'replay_check': vr,
                'checkpoints_sha256': {os.path.basename(p): RC.sha256(p) for p in ckpts}, 'byte_copies_sha256': byte,
                'script': os.path.basename(__file__), 'script_sha256': RC.sha256(os.path.abspath(__file__)),
                'elapsed_s': time.time() - t00, 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    RC.jdump(out, P['json'])
    print('[replay] wrote %s (%d frames, %s members) in %.0f s' % (P['npz'], N, a.arm, time.time() - t00), flush=True)
    return P


if __name__ == '__main__':
    main()
