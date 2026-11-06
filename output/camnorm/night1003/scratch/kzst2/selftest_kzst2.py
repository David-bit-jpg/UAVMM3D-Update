# -*- coding: utf-8 -*-
"""CPU self-tests for KZ-ST stage 2 (CUDA hidden; no ground truth of any real split is read).

  t1  targets: on the smoke build, the dataset's kp2d (repo make_kp2d_map on the pseudo box in the crop camera) equals
      g(R_T) in crop pixels; D_rho decoding in the crop intrinsics returns R_T (< 0.01 deg) for 20 crops x 5 students;
      pseudo corners drawn on the 20 crops -> smoke/vis/pseudo_corners_s<k>.jpg; hflip (forced) keeps the decoded pose
      = M R_T S and the 32-ch pair = [g, P g] on quotient frames, [g, g] otherwise; non-target cells are 0
  t2  loss: CenterHeadKZST with a 16-ch target == stock CenterHead.get_loss bit-for-bit; 32-ch pair with identical
      halves == stock L1 (to float rounding); swapping the halves leaves the loss unchanged (bit-for-bit); the quotient
      term picks the closer half
  t3  config: merged AUG == prereg AUG key by key (and the base config WOULD have target_ring 0.3 / target_blur 0.2),
      FREEZE_BN false, 44 BN layers, LOSS_WEIGHTS / LR / epochs; crop sx == sy
  t4  crops: the stored crops are the deployed crops -- original kz_s0..4 on 4 stored crops reproduce the recorded train
      member windows (CPU tolerance kp 2e-2, hm 2e-3)
  t5  selection integrity: every pseudo-label frame is a TRAIN frame of the stage-1 selection; no test sequence
  t6  eval end-to-end on a synthetic fixture (fabricated GT): freeze + evaluate, then evaluate again with a stage-1
      results file (Holm family / A0** = BN path)
  t7  LABEL-FREE eval 'freeze' on the REAL test inputs with a stand-in LOO npz (the 5 multimodal KZ students): asserts
      A0 == frozen_e1sa, BN == stage-1 BN, A0' == stage-1 A0'; times the 126 x 4 multisets (never runs 'evaluate')
  t8  data-loader cost per training sample (CPU)
    python selftest_kzst2.py [t1 ... t8]
"""
import sys

sys.dont_write_bytecode = True
import os            # noqa: E402

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import json          # noqa: E402
import shutil        # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

import kzst2_common as S   # noqa: E402

C1, KFO = S.C1, S.KFO
SM = S.SMOKE
SMOKE_DATA = os.path.join(SM, 'data')
SMOKE_PREREG = os.path.join(SM, 'prereg_smoke.json')


def _ds(student, training, aug=None, max_frames=0):
    import kzst2_torch as KT
    cfg = KT.build_cfg(student, os.path.join(SMOKE_DATA, 'pseudo_stage2.npz'), os.path.join(SMOKE_DATA, 'crops_train_u8.npy'), max_frames)
    if aug is not None:
        from easydict import EasyDict
        cfg.DATA_CONFIG.AUG = EasyDict(aug)
    return KT.KZST2_Crop_Dataset(cfg.DATA_CONFIG, training=training), cfg


def _center_cell(kp):
    m = np.abs(kp[0]).sum(0) > 0
    assert m.sum() == 1, m.sum()
    r, c = np.argwhere(m)[0]
    return int(r), int(c)


def t1():
    import cv2
    from scipy.spatial.transform import Rotation as Rot
    Z = dict(np.load(os.path.join(SMOKE_DATA, 'pseudo_stage2.npz')))
    crops = np.load(os.path.join(SMOKE_DATA, 'crops_train_u8.npy'), mmap_mode='r')
    os.makedirs(os.path.join(SM, 'vis'), exist_ok=True)
    worst, worst_flip, n = 0.0, 0.0, 0
    for k in range(5):
        ds, _ = _ds(k, training=False)
        picks = list(range(min(20, len(ds))))
        tiles = []
        for item in picks:
            i = int(ds.valid_idx[item])
            d = ds[item]
            kp32 = d['kp2d']
            assert kp32.shape[1] == 32
            r, c = _center_cell(kp32[:, :16])
            ga, gb = kp32[0, :16, r, c].astype(np.float64), kp32[0, 16:, r, c].astype(np.float64)
            r16 = C1.rho16(Z['rho'][i])
            g_plain = S.g16(Z['K_crop'][i], Z['t'][i], Z['lwh'][i], Z['R_T'][i, k])
            assert np.allclose(ga / r16, g_plain, atol=2e-6), np.abs(ga / r16 - g_plain).max()
            assert np.allclose(ga, Z['target16'][i, k], atol=2e-6)
            assert np.allclose(gb, ga[S.P16] if Z['quotient'][i, k] else ga, atol=0)
            Rd = S.decode_crop(ga, Z['rho'][i], Z['K_crop'][i], Z['t'][i], Z['lwh'][i])
            worst = max(worst, KFO.geodesic(Rd, Z['R_T'][i, k]))
            n += 1
            if k == 0 or k == 4:
                img = np.ascontiguousarray(crops[i]).copy()
                c0 = S.proj(Z['K_crop'][i], Z['t'][i])[0]
                pts = (ga / r16).reshape(8, 2) * S.KP_SCALE + c0[None]
                for e0, e1 in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                    col = (0, 0, 255) if e0 in (1, 2, 5, 6) and e1 in (1, 2, 5, 6) else (0, 255, 0)          # +x (front) face red
                    cv2.line(img, tuple(int(round(v)) for v in pts[e0]), tuple(int(round(v)) for v in pts[e1]), col, 1, cv2.LINE_AA)
                cv2.circle(img, tuple(int(round(v)) for v in c0), 2, (255, 0, 0), -1)
                cv2.putText(img, '%s q=%d' % (Z['seq'][i], int(Z['quotient'][i, k])), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                tiles.append(img)
        if tiles:
            rows = [np.concatenate(tiles[j:j + 4], 1) for j in range(0, len(tiles) - len(tiles) % 4, 4)]
            cv2.imwrite(os.path.join(SM, 'vis', 'pseudo_corners_s%d.jpg' % k), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
        # forced horizontal flip, photometric off: geometry must stay consistent (R -> M R S, pair -> [g', P g'])
        aug = dict(S.AUG, hflip=1.0, photometric=False)
        dsf, _ = _ds(k, training=True, aug=aug)
        for item in picks[:8]:
            i = int(dsf.valid_idx[item])
            np.random.seed(item)
            d = dsf[item]
            kp32 = d['kp2d']
            r, c = _center_cell(kp32[:, :16])
            ga, gb = kp32[0, :16, r, c].astype(np.float64), kp32[0, 16:, r, c].astype(np.float64)
            Kf = d['intrinsic'][0]
            tf = d['gt_box9d'][0, :3].astype(np.float64)
            Rf = Rot.from_euler('xyz', d['gt_box9d'][0, 6:9].astype(np.float64)).as_matrix()
            Mx, Sy = np.diag([-1.0, 1, 1]), np.diag([1.0, -1, 1])
            assert KFO.geodesic(Rf, Mx @ Z['R_T'][i, k] @ Sy) < 1e-3       # float32 box storage
            Rd = S.decode_crop(ga, Z['rho'][i], Kf, tf, Z['lwh'][i])
            worst_flip = max(worst_flip, KFO.geodesic(Rd, Mx @ Z['R_T'][i, k] @ Sy))
            assert np.allclose(gb, ga[S.P16] if Z['quotient'][i, k] else ga, atol=0)
            # the flipped target = mirrored + body-y relabelled original target
            g0 = Z['target16'][i, k].reshape(8, 2)[S.FLIP_SIGMA] * np.array([-1.0, 1.0])
            assert np.allclose(ga, g0.reshape(16), atol=5e-4), np.abs(ga - g0.reshape(16)).max()
            # P commutes with the flip relabelling: the pair stays consistent after flipping
            assert np.allclose(g0.reshape(16)[S.P16], (Z['target16'][i, k][S.P16].reshape(8, 2)[S.FLIP_SIGMA] * np.array([-1.0, 1.0])).reshape(16))
    assert worst < S.MAX_ROUNDTRIP_DEG and worst_flip < 0.01, (worst, worst_flip)
    print('[t1] OK: %d crop targets == g(R_T) in crop px (repo make_kp2d_map); D_rho PnP in crop intrinsics -> R_T max %.1e deg; '
          'hflip: decoded pose == M R_T S (max %.1e deg), target == mirrored + relabelled, pair stays [g, Pg]/[g, g]; '
          'drawings in %s' % (n, worst, worst_flip, os.path.join(SM, 'vis')))


def t2():
    import torch
    import kzst2_torch as KT
    from easydict import EasyDict
    _, cfg = _ds(0, training=False)
    hcfg = cfg.MODEL.DENSE_HEAD_2D
    torch.manual_seed(0)
    stock = KT.CenterHead(hcfg)
    mine = KT.CenterHeadKZST(hcfg)
    mine.load_state_dict(stock.state_dict())
    B, h, w = 4, 36, 64
    pred = {k: torch.randn(B, int(hcfg.SEPARATE_HEAD_CFG.HEAD_DICT[k]['out_channels']), h, w) for k in hcfg.SEPARATE_HEAD_CFG.HEAD_ORDER}
    gt = {}
    for k, p in pred.items():
        g = torch.zeros(B, 1, p.shape[1], h, w)
        g[:, :, :, 10, 20] = torch.randn(B, 1, p.shape[1])
        gt[k] = g
    gt['hm'] = torch.zeros(B, 1, 1, h, w)
    gt['hm'][:, :, :, 10, 20] = 1.0
    fg = torch.zeros(B, 1, 1, h, w)
    fg[:, :, :, 10, 20] = 1.0

    def loss_of(head, gtd):
        head.forward_loss_dict = {'pred_center_dict': pred, 'gt_center_dict': dict(gtd), 'fg_mask': fg, 'rot_mask': None}
        return head.get_loss()
    l_stock = loss_of(stock, gt)
    l_mine16 = loss_of(mine, gt)
    assert torch.equal(l_stock, l_mine16), (l_stock, l_mine16)
    g16 = gt['kp2d']
    same = dict(gt, kp2d=torch.cat([g16, g16], 2))
    l_same = loss_of(mine, same)
    assert abs(float(l_same) - float(l_stock)) < 1e-5, (float(l_same), float(l_stock))
    gp = g16[:, :, torch.as_tensor(S.P16)]
    pair = dict(gt, kp2d=torch.cat([g16, gp], 2))
    swap = dict(gt, kp2d=torch.cat([gp, g16], 2))
    l_pair, l_swap = loss_of(mine, pair), loss_of(mine, swap)
    assert torch.equal(l_pair, l_swap)
    pk = pred['kp2d'].reshape(B, 1, 16, h, w)[:, :, :, 10, 20]
    la = (pk - g16[:, :, :, 10, 20]).abs().sum(2)
    lb = (pk - gp[:, :, :, 10, 20]).abs().sum(2)
    manual = torch.minimum(la, lb).sum() / (B * 16)
    term = mine.loss_terms['kp2d']
    assert abs(term - float(manual)) < 1e-6 and float(l_pair) <= float(l_same) + 1e-6
    assert mine.loss_weights['kp2d'] == float(hcfg.LOSS_WEIGHTS['kp2d'])            # weight restored after the call
    print('[t2] OK: 16-ch target -> stock loss bit-for-bit (%.6f); identical halves == stock L1 (%.6f vs %.6f); '
          'swapped halves bit-identical (%.6f); quotient term == manual min (%.6f)' % (
              float(l_stock), float(l_same), float(l_stock), float(l_pair), term))
    del EasyDict


def t3():
    import torch
    import kzst2_torch as KT
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.model import build_network
    ds, cfg = _ds(0, training=True)
    KT.assert_cfg(cfg)
    base = EasyDict()
    cwd = os.getcwd()
    os.chdir(S.TOOLS)
    try:
        cfg_from_yaml_file(S.SIM_KP_CFG, base)
    finally:
        os.chdir(cwd)
    assert float(base.DATA_CONFIG.AUG.target_ring) == 0.3 and float(base.DATA_CONFIG.AUG.target_blur) == 0.2
    m = build_network(cfg.MODEL, ds)
    n, t = m.load_params_from_file(S.RELEASE_KZ[0], to_cpu=True)
    assert n == t and KT.count_bn(m) == 44
    Z = np.load(os.path.join(SMOKE_DATA, 'pseudo_stage2.npz'))
    s_ = Z['crop'][:, 4]
    assert np.allclose(s_, S.IN_WH[0] / Z['crop'][:, 2]) and np.allclose(Z['crop'][:, 3] * S.IN_WH[0], Z['crop'][:, 2] * S.IN_WH[1])
    assert np.allclose(Z['K_crop'][:, 0, 0] / Z['K_crop'][:, 1, 1], Z['K_raw'][:, 0, 0] / Z['K_raw'][:, 1, 1])
    print('[t3] OK: merged AUG %s == prereg (base config would carry target_ring 0.3 / target_blur 0.2 -> overridden); '
          'FREEZE_BN false, %d BN layers, %d/%d tensors loaded; crop sx == sy' % (dict(cfg.DATA_CONFIG.AUG), KT.count_bn(m), n, t))
    del torch


def t4():
    import torch
    import adabn_kz as AB
    Z = np.load(os.path.join(SMOKE_DATA, 'pseudo_stage2.npz'))
    crops = np.load(os.path.join(SMOKE_DATA, 'crops_train_u8.npy'), mmap_mode='r')
    D, info = C1.load_npz_info(S.STAGE1['train_npz'])
    dev = torch.device('cpu')
    models, _, _, _ = AB.build_models(torch, dev, S.RELEASE_KZ)
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    dk, dh = [], []
    for n in np.linspace(0, len(Z['src_idx']) - 1, 4).astype(int):
        i = int(Z['src_idx'][n])
        outs = AB.forward(models, AB.to_input(np.ascontiguousarray(crops[n]), nm, ns)[None], torch, dev)
        hm_win, kp_win, rbcb, _ = AB.window(D, i, [o[0][0] for o in outs], [o[1][0] for o in outs], int(info['search']), torch)
        ok = np.isfinite(D['hm_win'][i])
        dh.append(float(np.abs(hm_win[ok] - D['hm_win'][i][ok]).max()))
        okk = np.isfinite(D['kp_win'][i])
        dk.append(float(np.abs(kp_win[okk] - D['kp_win'][i][okk]).max()))
        assert tuple(rbcb) == tuple(int(v) for v in D['rb_cb'][i])
    assert max(dk) < 2e-2 and max(dh) < 2e-3, (dk, dh)
    print('[t4] OK: stored crops reproduce the recorded deployed train member windows (CPU): max |kp| %.1e, |hm| %.1e, read cells equal' % (max(dk), max(dh)))


def t5():
    Z = np.load(os.path.join(SMOKE_DATA, 'pseudo_stage2.npz'))
    F = np.load(S.STAGE1['frozen_train'])
    assert set(Z['src_idx'].tolist()) <= set(F['sel_idx'].tolist())
    Dte, _ = C1.load_npz_info(S.TEST_NPZ)
    test_seqs = {str(s) for s in Dte['seq']}
    assert not ({str(s) for s in Z['seq']} & test_seqs)
    D, info = C1.load_npz_info(S.STAGE1['train_npz'])
    assert info['split'] == 'train'
    assert all(str(D['seq'][i]) == s for i, s in zip(Z['src_idx'], Z['seq']))
    print('[t5] OK: pseudo-label frames are stage-1 selected TRAIN frames; no test sequence among them')


def _fake_members_npz(D, info, members, out_npz, arm, prereg_path):
    import selftest_stage1 as ST1
    ST1.fake_bn_npz(D, info, members, out_npz, prereg_path)
    J = json.load(open(os.path.splitext(out_npz)[0] + '.json', encoding='utf-8'))
    J['arm'] = arm
    S.jdump(J, os.path.splitext(out_npz)[0] + '.json')
    return out_npz


def t6():
    import kzst2_eval as E
    import selftest_stage1 as ST1
    t0 = time.time()
    root = os.path.join(SM, 'fixture_eval')
    if os.path.exists(root):
        shutil.rmtree(root)
    KFO.make_fixture(root, N=400, seed=11)
    D, info = C1.load_npz_info(os.path.join(root, 'kz_members_synth.npz'))
    _fake_members_npz(D, info, [6, 7, 8, 9, 10], os.path.join(root, 'kz_members_loo_test.npz'), 'loo', SMOKE_PREREG)
    _fake_members_npz(D, info, [5, 6, 7, 8, 9], os.path.join(root, 'kz_members_adabn_test.npz'), 'adabn', SMOKE_PREREG)
    ST1.fake_frozen_train(root, {'phantom4': (0.95, 1.0), 'mavic2': (1.0, 0.9)}, SMOKE_PREREG)
    none = os.path.join(root, 'none')
    args = ['--out-dir', root, '--prereg', SMOKE_PREREG, '--test-npz', os.path.join(root, 'kz_members_synth.npz'),
            '--test-dump', os.path.join(root, 'real_ens5flip_kpKZavg_zoom_synth.pkl'), '--test-frozen', none + '.npz',
            '--loo-npz', os.path.join(root, 'kz_members_loo_test.npz'), '--bn-npz', os.path.join(root, 'kz_members_adabn_test.npz'),
            '--sim-json', os.path.join(SM, 'sim', 'sim_diag_loo.json'), '--stage1-results', none + '.json',
            '--stage1-frozen-test', none + '.npz', '--frozen-train', os.path.join(root, 'frozen_train_stage1.npz'),
            '--train-root', os.path.join(SM, 'train'), '--pseudo-json', os.path.join(SMOKE_DATA, 'pseudo_stage2.json'),
            '--allow-smoke', '--allow-nontrain-rho', '--workers', '8']
    J = E.main(['freeze'] + args)
    assert J['have_bn']
    R = E.main(['evaluate'] + args)
    p = R['P1_primary_vs_A0_2star']
    assert set(p['criteria']) == {'delta_le_-1', 'ci_upper_lt_0', 'seq_better_ge_7', 'paired_ge_4', 'holm'}
    assert R['decision_LOO'] in ('deploy_LOO', 'pending_sim_guardrail', 'do_not_deploy_LOO')
    # second pass with a stage-1 results file built from the first (A0* / BN reproduction asserts, Holm with p_BN)
    s1 = {'table': {'BN': R['table']['BN'], 'A0p': R['table']['A0p']}, 'A0_star': {'arm': R['A0_star']['arm']},
          'P1_BN': {'p_one_sided': 0.0032}, 'decision_BN': 'do_not_deploy_BN'}
    S.jdump(s1, os.path.join(root, 'results_stage1_fake.json'))
    args2 = list(args)
    args2[args2.index('--stage1-results') + 1] = os.path.join(root, 'results_stage1_fake.json')
    R2 = E.main(['evaluate'] + args2)
    assert R2['P1_primary_vs_A0_2star']['holm_family']['P1_BN_stage1'] == 0.0032 and R2['decision_BN_stage1'] == 'do_not_deploy_BN'
    assert R2['A0_2star']['arm'] == ('BN' if R2['table']['BN']['all']['head'] < R2['A0_star']['head'] else R2['A0_star']['arm'])
    print('[t6] OK: eval freeze + evaluate on a synthetic fixture: A0 %.2f A0p %.2f BN %.2f LOO %.2f; A0** = %s; primary %s; '
          'secondary would-pass %s; decision %s; second pass with stage-1 results file OK (%.0f s)' % (
              R['table']['A0']['all']['head'], R['table']['A0p']['all']['head'], R['table']['BN']['all']['head'], R['table']['LOO']['all']['head'],
              R2['A0_2star']['arm'], p['criteria'], R['P1_secondary_vs_A0_star']['would_pass_report_only'], R['decision_LOO'], time.time() - t0))


def t6b():
    """fixture variant where BN is better than A0*: the fake BN members are rebuilt from the FABRICATED fixture ground
    truth (+ small noise), so A0** = BN; exercises the bn multisets as the primary baseline and the bn_rho paired singles."""
    import kzst2_eval as E
    import selftest_stage1 as ST1
    from scipy.spatial.transform import Rotation as Rot
    t0 = time.time()
    root = os.path.join(SM, 'fixture_eval')
    root2 = os.path.join(SM, 'fixture_eval_bn')
    if os.path.exists(root2):
        shutil.rmtree(root2)
    shutil.copytree(root, root2)
    D, info = C1.load_npz_info(os.path.join(root2, 'kz_members_synth.npz'))
    fr, _ = KFO.load_dump(os.path.join(root2, 'real_ens5flip_kpKZavg_zoom_synth.pkl'), with_gt=True)     # fabricated GT
    rng = np.random.default_rng(3)
    D2 = dict(D)
    kp = D['kp_win'].copy()
    lwh = np.array([0.3489, 0.3487, 0.1996])
    for i in range(len(D['seq'])):
        gt = fr[(str(D['seq'][i]), str(D['frame'][i]))]['gt']
        g = KFO.frame_geo(D, i, lwh, 64.0)
        x = KFO.synth_member_vec(np.asarray(D['K_in'][i]), Rot.from_euler('xyz', gt[6:9]).as_matrix(), gt[:3], g, lwh)
        for m in range(5):
            ok = np.isfinite(kp[i, m, 0])
            kp[i, m][:, ok] = ((x + rng.normal(0, 0.002, 16)) * D['z'][i])[:, None].astype(np.float32)
    D2['kp_win'] = kp
    _fake_members_npz(D2, info, [0, 1, 2, 3, 4], os.path.join(root2, 'kz_members_adabn_test.npz'), 'adabn', SMOKE_PREREG)
    ST1.fake_frozen_train(root2, {'phantom4': (0.95, 1.0), 'mavic2': (1.0, 0.9)}, SMOKE_PREREG)
    none = os.path.join(root2, 'none')
    args = ['--out-dir', root2, '--prereg', SMOKE_PREREG, '--test-npz', os.path.join(root2, 'kz_members_synth.npz'),
            '--test-dump', os.path.join(root2, 'real_ens5flip_kpKZavg_zoom_synth.pkl'), '--test-frozen', none + '.npz',
            '--loo-npz', os.path.join(root2, 'kz_members_loo_test.npz'), '--bn-npz', os.path.join(root2, 'kz_members_adabn_test.npz'),
            '--sim-json', os.path.join(SM, 'sim', 'sim_diag_loo.json'), '--stage1-results', none + '.json',
            '--stage1-frozen-test', none + '.npz', '--frozen-train', os.path.join(root2, 'frozen_train_stage1.npz'),
            '--train-root', os.path.join(SM, 'train'), '--pseudo-json', os.path.join(SMOKE_DATA, 'pseudo_stage2.json'),
            '--allow-smoke', '--allow-nontrain-rho', '--workers', '8', '--skip-ads']
    E.main(['freeze'] + args)
    R = E.main(['evaluate'] + args)
    assert R['A0_2star']['arm'] == 'BN', (R['A0_2star'], R['A0_star'])
    p = R['P1_primary_vs_A0_2star']
    assert p['base'] == 'BN' and p['paired_base_singles'] == 'bn_rho' and R['P1_secondary_vs_A0_star']['paired_base_singles'] == 'kz_rho'
    assert p['delta'] > 0 and not p['pass']          # LOO (noisy members) is far worse than the near-GT BN
    print('[t6b] OK: near-GT BN %.2f < A0* %.2f -> A0** = BN; primary vs BN delta %+.2f, two-level CI %s, paired vs bn_rho singles '
          '(%d/5 better); secondary vs A0* delta %+.2f (%.0f s)' % (R['table']['BN']['all']['head'], R['A0_star']['head'], p['delta'],
                                                               [round(x, 2) for x in p['two_level_ci95']], p['n_paired_better'],
                                                               R['P1_secondary_vs_A0_star']['delta'], time.time() - t0))


def t7():
    import kzst2_eval as E
    t0 = time.time()
    d = os.path.join(SM, 'real_freeze')
    os.makedirs(d, exist_ok=True)
    D, info = C1.load_npz_info(S.TEST_NPZ)
    loo = _fake_members_npz(D, info, [5, 6, 7, 8, 9], os.path.join(d, 'kz_members_loo_test.npz'), 'loo', SMOKE_PREREG)
    J = E.main(['freeze', '--out-dir', d, '--prereg', SMOKE_PREREG, '--loo-npz', loo, '--allow-smoke', '--workers', '8'])
    assert J.get('bn_equals_stage1_frozen') is True
    print('[t7] OK: label-free eval freeze on the REAL test dump with a stand-in LOO (multimodal students): A0 == frozen_e1sa, '
          'BN == stage-1 BN, A0p == stage-1 A0p (asserted), identity multisets == arms; %.0f s (no ground truth read)' % (time.time() - t0))


def t8():
    ds, _ = _ds(1, training=True)
    t0 = time.time()
    n = 0
    for _ in range(3):
        for item in range(len(ds)):
            ds[item]
            n += 1
    dt = (time.time() - t0) / n
    print('[t8] OK: %.1f ms per training sample on CPU (workers 0 -> ~%.2f s of loading per batch of 8)' % (1000 * dt, 8 * dt))


if __name__ == '__main__':
    todo = sys.argv[1:] or ['t1', 't2', 't3', 't4', 't5', 't6', 't6b', 't7', 't8']
    for t in todo:
        globals()[t]()
