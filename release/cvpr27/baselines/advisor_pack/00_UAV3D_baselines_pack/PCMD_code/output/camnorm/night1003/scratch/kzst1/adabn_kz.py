# -*- coding: utf-8 -*-
"""KZ-ST v2 stage 1, S3 (GPU, no labels, no pseudo labels, no gradient): the AdaBN arm.

1. Re-estimation frames = TRAIN only: g0 AND top detection at the KZ hook cell, <= 80 per sequence uniform in time
   (kzst_common.adabn_selection; identical list is re-derived and asserted by kzst_stage1_freeze.py). Asserted: no test
   sequence among them.
2. Crops = the exact deploy crops: for every frame the train member dump recorded the zoom-95 window (x0, y0, w, h, s)
   that tools/kp_zoom.py cut from the undistorted raw image; this script re-cuts it with the same statements
   (kp_zoom.RawSource remap, cv2.warpAffine INTER_AREA / INTER_LINEAR, x/255, (x - NORM_MEAN) / NORM_STD).
   Replay check before anything else: the ORIGINAL kz_s{k} on replayed crops must reproduce the recorded member windows
   (train and test) -- hard stop otherwise.
3. AdaBN (Li et al.; same recipe as tools/eval_camnorm.adapt_bn): every BatchNorm2d reset_running_stats(), momentum=None
   (cumulative average over batches), BN layers in train mode, everything else eval, torch.no_grad(); batches of 16
   crops in a fixed random order (numpy default_rng(0) permutation); all 5 members see the same batches. No weight update.
   Saved as adabn/kz_s{k}_adabn.pth (same checkpoint format as release_20260923/kz_s{k}.pth).
4. Test evaluation input: re-cut the 4800 test crops recorded in e1sa/kz_members_e1sa.npz (positions / crops are those of
   the deployed pipeline, unchanged), forward the 5 AdaBN members (batch 1, as in the hook), record hm / kp2d windows,
   read-out cell = argmax of the 5-member mean hm in the 5x5 window, deployed vector = mean kp2d / z (hook statements).
   Writes kz_members_adabn_test.npz/.json in the e1sa schema (member names KZs0..KZs4, arm = adabn).

    cd @@ROOT@@\\tools
    python <scratchpad>\\kzst1\\adabn_kz.py
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
import json          # noqa: E402
import time          # noqa: E402

import cv2           # noqa: E402
import numpy as np   # noqa: E402

import kzst_common as C   # noqa: E402

KFO = C.KFO


class NpzSource:
    """kp_zoom.RawSource fed with the (seq, frame, K_raw, raw_wh) recorded in a member npz (no dataset index needed)."""

    def __init__(self, D, root='@@MAV6D@@'):
        import kp_zoom
        metas = [{'seq': str(s), 'frame': str(f), 'K_raw': np.asarray(K, np.float64), 'raw_wh': tuple(int(v) for v in wh)}
                 for s, f, K, wh in zip(D['seq'], D['frame'], D['K_raw_src'], D['raw_img_wh'])]
        self.src = kp_zoom.RawSource(type('DS', (), {'metas': metas})(), root)

    def get(self, seq, frame):
        return self.src.get(seq, frame)


def recut(D, i, src, in_wh):
    """The hook's crop statements with the recorded window (x0, y0, w, h, s)."""
    img, K_raw = src.get(str(D['seq'][i]), str(D['frame'][i]))
    raw_h, raw_w = img.shape[:2]
    assert (raw_w, raw_h) == tuple(int(v) for v in D['raw_img_wh'][i]), (raw_w, raw_h, D['raw_img_wh'][i])
    assert np.allclose(K_raw, D['K_raw_src'][i])
    x0, y0, w, h, s = [float(v) for v in D['crop'][i]]
    in_w, in_h = in_wh
    assert abs(s - in_w / w) < 1e-9
    M = np.array([[s, 0, (0.5 - x0) * s - 0.5], [0, s, (0.5 - y0) * s - 0.5]], np.float64)
    return cv2.warpAffine(img, M, (in_w, in_h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def to_input(crop, nm, ns):
    x = crop.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (x - nm) / ns


def forward(models, x_b, torch, dev):
    """x_b: (B,3,H,W) float32 numpy -> list over models of (kp2d (B,16,h,w) float32, sigmoid hm (B,h,w)) torch tensors."""
    t = torch.from_numpy(np.ascontiguousarray(x_b[:, None])).to(dev)
    outs = []
    with torch.no_grad():
        for km in models:
            tb = {'image': t, 'batch_size': int(t.shape[0])}
            for mod in km.module_list:
                tb = mod(tb)
            outs.append((tb['pred_center_dict']['kp2d'].float(), torch.sigmoid(tb['pred_center_dict']['hm'].float()[:, 0])))
    return outs


def window(D, i, maps, hms, search, torch):
    """hook statements: 5x5 window around the recorded (rc, cc), NaN-padded; read cell from the mean of the members."""
    M = len(maps)
    S = 2 * search + 1
    rc, cc = (int(v) for v in D['rc_cc'][i])
    hm_b = torch.stack(hms).mean(0)
    kp_map = torch.stack(maps).mean(0)
    r0, r1 = max(0, rc - search), min(hm_b.shape[0], rc + search + 1)
    c0, c1 = max(0, cc - search), min(hm_b.shape[1], cc + search + 1)
    assert (r0, r1, c0, c1) == tuple(int(v) for v in D['win'][i]), ((r0, r1, c0, c1), D['win'][i])
    sub = hm_b[r0:r1, c0:c1]
    j = int(torch.argmax(sub.reshape(-1)))
    rb, cb = r0 + j // sub.shape[1], c0 + j % sub.shape[1]
    kp_dep = (kp_map[:, rb, cb] / float(D['z'][i])).float().cpu().numpy().astype(np.float32)
    hw = torch.stack(hms)[:, r0:r1, c0:c1].float().cpu().numpy()
    kw = torch.stack(maps)[:, :, r0:r1, c0:c1].float().cpu().numpy()
    hm_win = np.full((M, S, S), np.nan, np.float32)
    kp_win = np.full((M, 16, S, S), np.nan, np.float32)
    ar, ac = r0 - (rc - search), c0 - (cc - search)
    hm_win[:, ar:ar + (r1 - r0), ac:ac + (c1 - c0)] = hw
    kp_win[:, :, ar:ar + (r1 - r0), ac:ac + (c1 - c0)] = kw
    return hm_win, kp_win, (rb, cb), kp_dep


def build_models(torch, dev, ckpts):
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils
    cwd = os.getcwd()
    os.chdir(TOOLS)
    kcfg = EasyDict()
    cfg_from_yaml_file(C.KP_CFG, kcfg)
    kcfg.DATA_CONFIG.DATA_SPLIT['test'] = 'val'       # dataset object only constructs the network; never the test index
    kds, _, _ = build_dataloader(kcfg.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=common_utils.create_logger(), training=False)
    os.chdir(cwd)
    ms = []
    for ck in ckpts:
        km = build_network(kcfg.MODEL, kds)
        n, t = km.load_params_from_file(ck, to_cpu=(dev.type == 'cpu'))
        assert n == t, ck
        ms.append(km.to(dev).eval())
    return ms, kcfg, kds, (int(kds.new_im_width), int(kds.new_im_hight))


def decode_many(D, idx, src, in_wh, workers):
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda i: recut(D, i, src, in_wh), idx))


def verify_replay(D, idx, members_in_npz, models, src, in_wh, nm, ns, search, torch, dev, tol_kp, tol_hm, tag, workers):
    crops = decode_many(D, idx, src, in_wh, workers)
    dk, dh, cell_ok = [], [], 0
    for i, crop in zip(idx, crops):
        outs = forward(models, to_input(crop, nm, ns)[None], torch, dev)
        hm_win, kp_win, (rb, cb), kp_dep = window(D, i, [o[0][0] for o in outs], [o[1][0] for o in outs], search, torch)
        ref_h = D['hm_win'][i][members_in_npz]
        ref_k = D['kp_win'][i][members_in_npz]
        ok = np.isfinite(ref_h)
        dh.append(float(np.abs(hm_win[ok] - ref_h[ok]).max()))
        okk = np.isfinite(ref_k)
        dk.append(float(np.abs(kp_win[okk] - ref_k[okk]).max()))
        cell_ok += int((rb, cb) == tuple(int(v) for v in D['rb_cb'][i]))
    r = {'n': len(idx), 'max_abs_kp': float(max(dk)), 'max_abs_hm': float(max(dh)), 'read_cell_same': cell_ok,
         'tol_kp': tol_kp, 'tol_hm': tol_hm}
    r['pass'] = bool(r['max_abs_kp'] <= tol_kp and r['max_abs_hm'] <= tol_hm and cell_ok >= len(idx) - max(1, len(idx) // 50))
    print('[adabn] replay check (%s, original weights): %s' % (tag, r), flush=True)
    return r


def adabn(models, D, idx, src, in_wh, nm, ns, batch, seed, torch, dev, workers):
    bn_all = []
    for km in models:
        km.eval()
        bns = [m for m in km.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()
        bn_all.append(bns)
    order = np.asarray(idx)[np.random.default_rng(seed).permutation(len(idx))]
    t0 = time.time()
    nb = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut = None
        chunks = [order[j:j + batch] for j in range(0, len(order), batch)]
        for ci, ch in enumerate(chunks):
            if fut is None:
                fut = ex.submit(lambda c: [recut(D, i, src, in_wh) for i in c], ch)
            crops = fut.result()
            fut = ex.submit(lambda c: [recut(D, i, src, in_wh) for i in c], chunks[ci + 1]) if ci + 1 < len(chunks) else None
            x = np.stack([to_input(c, nm, ns) for c in crops])
            forward(models, x, torch, dev)
            nb += 1
            if nb % 50 == 0:
                print('[adabn]   %d / %d batches, %.0f s' % (nb, len(chunks), time.time() - t0), flush=True)
    for bns in bn_all:
        for bn in bns:
            bn.eval()
        assert all(int(bn.num_batches_tracked) == nb for bn in bns)
    return nb, len(bn_all[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=HERE)
    ap.add_argument('--train-tag', default=C.TRAIN_TAG)
    ap.add_argument('--prereg', default=None)
    ap.add_argument('--out', default=None, help='output folder (default <dir>/adabn)')
    ap.add_argument('--tag', default='adabn')
    ap.add_argument('--test-npz', default=C.TEST_NPZ)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=C.ADABN['batch'])
    ap.add_argument('--verify-n', type=int, default=24)
    ap.add_argument('--workers', type=int, default=6, help='JPEG decode threads')
    ap.add_argument('--max-train-frames', type=int, default=0, help='smoke tests only')
    ap.add_argument('--max-test-frames', type=int, default=0, help='smoke tests only')
    ap.add_argument('--smoke-any-frames', action='store_true', help='smoke tests only: skip the g0 rule')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    pre, _ = C.check_prereg(a.prereg)
    t00 = time.time()
    out = os.path.abspath(a.out or os.path.join(a.dir, 'adabn'))
    if not os.path.normcase(out).startswith(os.path.normcase(HERE)):
        raise SystemExit('refusing to write outside %s' % HERE)
    os.makedirs(out, exist_ok=True)
    out_npz = os.path.join(out, 'kz_members_%s_test.npz' % a.tag)
    if os.path.exists(out_npz) and not a.force:
        raise SystemExit('%s exists (use --force)' % out_npz)
    import torch
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        assert torch.cuda.is_available()
    smoke = bool(a.max_train_frames or a.max_test_frames or a.smoke_any_frames)

    # ---------------- re-estimation frames (train, label-free)
    tr_npz = os.path.join(a.dir, 'kz_members_%s.npz' % a.train_tag)
    tr_pkl = os.path.join(a.dir, 'real_ens5flip_kpKZavg_zoom_%s.pkl' % a.train_tag)
    Dtr, itr = C.load_npz_info(tr_npz)
    assert itr.get('split') == 'train', 'AdaBN re-estimation must use the TRAIN dump (got split=%r)' % itr.get('split')
    if itr.get('dump_finished_ok') is not True:
        raise SystemExit('the train member dump did not finish -- rerun S1')
    frames_tr, _ = C.load_dump_nogt(tr_pkl)
    if a.smoke_any_frames:
        T = C.frame_table(Dtr, frames_tr)
        sel = np.where(T['hook'])[0]
    else:
        sel, T, _, _ = C.adabn_selection(Dtr, frames_tr)
    if a.max_train_frames:
        sel = sel[:a.max_train_frames]
    Dte, ite = C.load_npz_info(a.test_npz)
    test_seqs = {str(s) for s in Dte['seq']}
    assert not ({str(T['seq'][i]) for i in sel} & test_seqs), 'a TEST sequence entered the AdaBN re-estimation set'
    keys = [list(T['keys'][i]) for i in sel]
    lj = {'prereg_sha256': pre, 'train_npz_sha256': C.sha256(tr_npz), 'n': len(sel), 'smoke': smoke, 'keys': keys,
          'per_class': {c: int((T['cls'][sel] == c).sum()) for c in C.CLASSES}, 'n_seq': int(len(set(T['seq'][sel])))}
    C.jdump(lj, os.path.join(out if smoke else a.dir, 'adabn_frames_%s.json' % a.train_tag))
    print('[adabn] re-estimation frames: %d (%s, %d sequences, train split only; no test sequence)' % (len(sel), lj['per_class'], lj['n_seq']), flush=True)

    # ---------------- models, normalisation, sources
    models, kcfg, kds, in_wh = build_models(torch, dev, C.RELEASE_KZ)
    for k in ('norm_mean', 'norm_std'):
        assert itr.get(k) == ite.get(k), (k, itr.get(k), ite.get(k))
    nm = np.asarray(itr['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(itr['norm_std'], np.float32).reshape(3, 1, 1)
    assert list(in_wh) == list(itr['in_wh']) == list(ite['in_wh'])
    search = int(itr.get('search', 2))
    src_tr, src_te = NpzSource(Dtr), NpzSource(Dte)
    te_members = C.member_index(ite)                 # positions of KZs0..4 in the e1sa npz (first five)
    tol = (2e-3, 2e-4) if dev.type == 'cuda' else (2e-2, 2e-3)
    nv = max(1, min(a.verify_n, len(sel)))
    vr_tr = verify_replay(Dtr, sel[np.linspace(0, len(sel) - 1, nv).astype(int)], list(range(5)), models, src_tr, in_wh, nm, ns,
                          search, torch, dev, tol[0], tol[1], 'train', a.workers)
    nte = len(Dte['seq']) if not a.max_test_frames else a.max_test_frames
    vi = np.linspace(0, nte - 1, max(1, min(a.verify_n, nte))).astype(int)
    vr_te = verify_replay(Dte, vi, te_members, models, src_te, in_wh, nm, ns, search, torch, dev, tol[0], tol[1], 'test', a.workers)
    if not (vr_tr['pass'] and vr_te['pass']):
        raise SystemExit('replay check failed -- the re-cut crops do not reproduce the recorded member outputs; stop')

    # ---------------- AdaBN
    stats_before = [{n: (b.running_mean.detach().cpu().numpy().copy(), b.running_var.detach().cpu().numpy().copy())
                     for n, b in km.named_modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)} for km in models]
    nb, nbn = adabn(models, Dtr, sel, src_tr, in_wh, nm, ns, a.batch, C.ADABN['order_seed'], torch, dev, a.workers)
    shift = []
    ck_out = []
    for k, km in enumerate(models):
        d_mean, r_var = [], []
        for n, b in km.named_modules():
            if isinstance(b, torch.nn.modules.batchnorm._BatchNorm):
                m0, v0 = stats_before[k][n]
                m1, v1 = b.running_mean.detach().cpu().numpy(), b.running_var.detach().cpu().numpy()
                d_mean.append(float(np.median(np.abs(m1 - m0) / np.sqrt(v0 + 1e-5))))
                r_var.append(float(np.median(v1 / (v0 + 1e-12))))
        shift.append({'median_abs_mean_shift_in_sim_std': float(np.median(d_mean)), 'median_var_ratio_real_over_sim': float(np.median(r_var)),
                      'first_layer_mean_shift': d_mean[0], 'last_layer_mean_shift': d_mean[-1]})
        src_ck = torch.load(C.RELEASE_KZ[k], map_location='cpu', weights_only=False)
        sd = {kk: v.detach().cpu() for kk, v in km.state_dict().items()}
        # weights untouched: every non-BN-buffer tensor equals the release checkpoint
        for kk, v in src_ck['model_state'].items():
            if not any(kk.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked')):
                assert torch.equal(sd[kk], v.cpu()), kk
        p = os.path.join(out, 'kz_s%d_%s.pth' % (k, a.tag))
        torch.save({'model_state': sd, 'epoch': src_ck.get('epoch'), 'src': '%s | AdaBN on %d train crops (kzst1/adabn_kz.py)' % (src_ck.get('src'), len(sel)),
                    'adabn': {'n_crops': int(len(sel)), 'n_batches': nb, 'batch': a.batch, 'order_seed': C.ADABN['order_seed'],
                              'base': C.RELEASE_KZ[k], 'prereg_sha256': pre, 'smoke': smoke}}, p)
        ck_out.append(p)
    # reload check: the saved files load into a fresh network (289/289 tensors) and carry the re-estimated statistics
    fresh, _, _, _ = build_models(torch, torch.device('cpu'), ck_out)
    for km, fm in zip(models, fresh):
        a_sd, b_sd = km.state_dict(), fm.state_dict()
        assert all(torch.equal(a_sd[k].detach().cpu(), b_sd[k]) for k in a_sd), 'saved AdaBN checkpoint does not reload identically'
    del fresh
    print('[adabn] re-estimated %d BN layers x %d members on %d crops (%d batches); checkpoints reload identically; stat shift %s' % (
        nbn, len(models), len(sel), nb, shift), flush=True)

    # ---------------- test replay with the AdaBN members
    N = nte
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
            crops = list(ex.map(lambda i: recut(Dte, i, src_te, in_wh), idx))
            for i, crop in zip(idx, crops):
                outs = forward(models, to_input(crop, nm, ns)[None], torch, dev)
                HM[i], KP[i], RB[i], KD[i] = window(Dte, i, [o[0][0] for o in outs], [o[1][0] for o in outs], search, torch)
            if (j0 // chunk) % 10 == 9:
                print('[adabn]   test %d / %d frames, %.0f s' % (j0 + len(idx), N, time.time() - t0), flush=True)
    keep = ('seq', 'frame', 'pos_rc', 'center_res', 'hm_peak', 'u', 'v', 'w2d', 'z', 'crop', 'uc_vc', 'rc_cc', 'win', 'raw_img_wh',
            'K_raw_src', 'K_in', 'dist', 'raw_size', 'new_size', 'stride_b', 'known_size')
    Z = {k: Dte[k][:N] for k in keep}
    Z.update({'hm_win': HM, 'kp_win': KP, 'rb_cb': RB, 'kp_deployed': KD})
    with open(out_npz + '.tmp', 'wb') as f:
        np.savez(f, **Z)
    os.replace(out_npz + '.tmp', out_npz)
    info = {k: ite[k] for k in ('n_rgb', 'search', 'target_w2d', 'z_range', 'stride', 'in_wh', 'KP_SCALE', 'W', 'H', 'dec_kwargs',
                               'KNOWN_SIZE_at_install', 'SIZE_SOURCE_at_install', 'norm_mean', 'norm_std') if k in ite}
    info.update({'member_names': list(C.RGB), 'arm': 'adabn', 'members': [{'name': n, 'path': p, 'role': 'adabn'} for n, p in zip(C.RGB, ck_out)],
                 'n_frames': int(N), 'npz': os.path.basename(out_npz), 'npz_sha256': C.sha256(out_npz),
                 'kp2d_dtype': ['torch.float32'], 'device': str(dev), 'prereg_sha256': pre, 'smoke': smoke,
                 'positions_crops_from': os.path.basename(a.test_npz), 'positions_crops_from_sha256': C.sha256(a.test_npz),
                 'replay_check': {'train': vr_tr, 'test': vr_te}, 'bn_stat_shift': shift, 'n_bn_layers': nbn,
                 'reestimation': {'n_crops': int(len(sel)), 'n_batches': nb, 'batch': a.batch, 'order_seed': C.ADABN['order_seed'],
                                  'frames_json': 'adabn_frames_%s.json' % a.train_tag, 'train_npz_sha256': lj['train_npz_sha256']},
                 'checkpoints_sha256': {os.path.basename(p): C.sha256(p) for p in ck_out},
                 'elapsed_s': time.time() - t00, 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    C.jdump(info, os.path.splitext(out_npz)[0] + '.json')
    print('[adabn] wrote %s (%d test frames) and %d checkpoints in %.0f s' % (out_npz, N, len(ck_out), time.time() - t00), flush=True)
    return out_npz


if __name__ == '__main__':
    main()
