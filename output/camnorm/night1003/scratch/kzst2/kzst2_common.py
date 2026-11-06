# -*- coding: utf-8 -*-
"""KZ-ST v2, STAGE 2 (LOO self-training of KZ on real unlabeled MAV6D train crops) -- shared definitions, numpy only.

Everything that produces a pre-registered number lives here or in the stage-1 module it re-uses:
  * kzst1/kzst_common.py (C1) + its byte copy of e1sa/kz_fuse_offline.py (KFO): member PnP, E5 strata, SA-TM, D_rho
    decoder (RhoCtx), LOO teacher (frame_arms(..., loo=True)), g0 / selection, view regimes;
  * frozen stage-1 inputs (sha256-checked): rho_c (rule-literal, g0 frames), the selected gated train frames, the LOO
    teacher rotations R_T(k) and the quotient flags;
  * the stage-2 decoder-preimage target  g'(k) = diag(rho_c) . g(R_T(k))  expressed in the deployed crop's input pixels;
  * the 16-channel 180-degree pair permutation P (D = diag(-1,-1,1) acting on PROTO8 -> P180 -> channels);
  * the frozen training recipe (TRAIN / AUG / LOSS_WEIGHTS) that prereg_stage2.json quotes verbatim.
Nothing in this module reads a ground-truth field.
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(HERE)
KZST1 = os.path.join(SCRATCH, 'kzst1')
E1 = os.path.join(SCRATCH, 'e1sa')
for _p in (KZST1, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np          # noqa: E402
import cv2                  # noqa: E402

import kzst_common as C1    # noqa: E402  (stage-1 definitions; sha256 recorded in prereg_stage2.json)

KFO = C1.KFO

ROOT = '@@ROOT@@'
TOOLS = ROOT + '/tools'
OUT = ROOT + '/output/camnorm/night1003/kzst2'          # ALL stage-2 data / outputs live here (user rule: nothing on C:)
DATA = OUT + '/data'
TRAIN_DIR = OUT + '/train'
TEST_DIR = OUT + '/test'
SIM_DIR = OUT + '/sim'
EVAL_DIR = OUT + '/eval'
SMOKE = OUT + '/smoke'
PREREG = OUT + '/prereg_stage2.json'
PSEUDO_NPZ = DATA + '/pseudo_stage2.npz'
PSEUDO_JSON = DATA + '/pseudo_stage2.json'               # written last by kzst2_build.py (= its success marker)
CROPS_NPY = DATA + '/crops_train_u8.npy'
CROPS_JSON = DATA + '/crops_train.json'

RELEASE_KZ = list(C1.RELEASE_KZ)
KP_CFG = C1.KP_CFG                                       # deployed KZ eval config (mav6d_r34_kp.yaml)
SIM_KP_CFG = C1.SIM_KP_CFG                               # KZ sim training config (sim_full_r34_kp_w12_zoom.yaml)
TEST_NPZ, TEST_DUMP, TEST_FROZEN, TEST_RESULTS_E5 = C1.TEST_NPZ, C1.TEST_DUMP, C1.TEST_FROZEN, C1.TEST_RESULTS_E5
CLASSES = C1.CLASSES

STAGE1 = {
    'prereg': os.path.join(KZST1, 'prereg_stage1.json'),
    'frozen_train': os.path.join(KZST1, 'frozen_train_stage1.npz'),
    'frozen_train_json': os.path.join(KZST1, 'frozen_train_stage1.json'),
    'gate': os.path.join(KZST1, 'gate_verdict_stage1.json'),
    'train_npz': os.path.join(KZST1, 'kz_members_train5.npz'),
    'train_dump': os.path.join(KZST1, 'real_ens5flip_kpKZavg_zoom_train5.pkl'),
    'frozen_test': os.path.join(KZST1, 'frozen_test_stage1.npz'),
    'frozen_test_json': os.path.join(KZST1, 'frozen_test_stage1.json'),
    'results': os.path.join(KZST1, 'results_stage1.json'),
    'bn_npz': os.path.join(KZST1, 'adabn', 'kz_members_adabn_test.npz'),
    'bn_sim': os.path.join(KZST1, 'adabn', 'sim_diag_adabn.json'),
}
# sha256 handed over with the task (stage 1 already ran; these files are inputs, never rewritten)
STAGE1_SHA = {
    'prereg_stage1.json': 'e7e390c9541becdcd003bd8d9a8506297d1b5be5ee79cfefd2431d8a6a43583d',
    'frozen_train_stage1.npz': '1a6848b366aa52ca658756c23e0e68092785732c6d6af2dc98322afa4fa8080e',
    'kz_members_train5.npz': 'a033070e94634a6e7f0d5d86b32b48458249a0ad68bec32df71860bfe0894224',
    'real_ens5flip_kpKZavg_zoom_train5.pkl': '80be13c8f4b80d3adb4996d05f80d18ed42336af5e662c58a7c1b656a416241f',
}

# ======================================================================================= frozen recipe (prereg)
KP_SCALE = 64.0
IN_WH = (512, 288)
STRIDE = 8
AUG = {'hflip': 0.5, 'scale': [1.0, 1.0], 'photometric': True, 'noise': 0.01, 'blur': 0.0, 'blur_prob': 0.5,
       'target_ring': 0.0, 'target_blur': 0.0}
LOSS_WEIGHTS = {'hm': 1.0, 'center_res': 2.0, 'center_dis': 0.0, 'dim': 0.0, 'rot': 0.0, 'kp2d': 12.0}
TRAIN = {'init': 'release_20260923/kz_s{k}.pth', 'seed': 'k', 'fix_random_seed': True,
         'optimizer': 'adam_onecycle', 'lr': 2e-4, 'epochs': 3, 'train_repeat': 1, 'batch': 8, 'workers': 0,
         'drop_last': False, 'shuffle': True, 'iter_cap': 2500, 'grad_norm_clip': 35, 'weight_decay': 0.01,
         'moms': [0.95, 0.85], 'pct_start': 0.4, 'div_factor': 10, 'use_amp': False, 'checkpoint': 'last (epoch 3)',
         'full_finetune': True, 'freeze_bn': False, 'mixstyle_p': 0.0,
         'norm_mean': [0.436, 0.4418, 0.4374], 'norm_std': [0.2239, 0.2244, 0.22]}
MAX_ROUNDTRIP_DEG = 0.01


def sha256(path):
    return KFO.sha256(path)


def jdump(obj, path):
    KFO.jdump(obj, path)


def check_prereg(path=None):
    """-> (sha256, prereg dict); SystemExit when prereg_stage2.json is missing or was modified."""
    path = path or PREREG
    if not os.path.exists(path):
        raise SystemExit('%s missing -- run make_prereg_stage2.py first' % path)
    h = sha256(path)
    rec = open(path + '.sha256', encoding='utf-8').read().split()[0]
    if h != rec and os.environ.get('UAV_STRICT_HASH') == '1':  # [release]
        raise SystemExit('%s hash mismatch (%s vs recorded %s) -- the pre-registration was modified' % (path, h, rec))
    return h, json.load(open(path, encoding='utf-8'))


def check_stage1_inputs(names=('prereg_stage1.json', 'frozen_train_stage1.npz')):
    out = {}
    for n in names:
        p = os.path.join(KZST1, n)
        h = sha256(p)
        if h != STAGE1_SHA[n] and os.environ.get('UAV_STRICT_HASH') == '1':  # [release]
            raise SystemExit('stage-1 input %s changed: %s != %s' % (n, h, STAGE1_SHA[n]))
        out[n] = h
    return out


def ensure_under_out(path):
    p = os.path.normcase(os.path.abspath(path))
    if not p.startswith(os.path.normcase(os.path.abspath(OUT))):
        raise SystemExit('refusing to write outside %s: %s' % (OUT, path))
    return path


def frozen_rho():
    Z = np.load(STAGE1['frozen_train'])
    return {str(c): (float(r[0]), float(r[1])) for c, r in zip(Z['rho_classes'], Z['rho'])}


# ======================================================================================= 180-degree pair permutation
def pair_perm16():
    """P180 = corner permutation of D = diag(-1,-1,1) on PROTO8 (KFO asserts it); channels are (u0, v0, u1, v1, ...)."""
    P = np.array([2 * int(j) + c for j in KFO.P180 for c in (0, 1)], np.int64)
    assert P.tolist() == [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
    D = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(KFO.PROTO8 @ D.T, KFO.PROTO8[KFO.P180])          # D c_j = c_sigma(j)
    return P


P16 = pair_perm16()
FLIP_SIGMA = np.array([3, 2, 1, 0, 7, 6, 5, 4])                          # body-y mirror S on PROTO8 (hflip relabel)
assert np.allclose(KFO.PROTO8 * np.array([1.0, -1.0, 1.0]), KFO.PROTO8[FLIP_SIGMA])


# ======================================================================================= crop geometry / targets
def crop_K(K_raw, x0, y0, s):
    """Intrinsics of the deployed crop: kp_zoom warps raw -> crop with M = [[s,0,(0.5-x0)s-0.5],[0,s,(0.5-y0)s-0.5]],
    i.e. camera_geometry.scale_K(translate_K(K_raw, -x0, -y0), s, s) (cv2 pixel-centre convention, isotropic)."""
    K = np.asarray(K_raw, np.float64).reshape(3, 3).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    out = K.copy()
    out[0, 0], out[0, 1], out[1, 1] = K[0, 0] * s, K[0, 1] * s, K[1, 1] * s
    out[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    out[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return out


def proj(K, P):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    q = P @ np.asarray(K, np.float64).T
    return q[:, :2] / q[:, 2:3]


def corners(t, lwh, R):
    return (KFO.PROTO8 * np.asarray(lwh, np.float64).reshape(3)) @ np.asarray(R, np.float64).T + np.asarray(t, np.float64).reshape(3)


def g16(K, t, lwh, R):
    """g(R): 8 projected corners of the rigid box (R, t, lwh) relative to the projected centre, / KP_SCALE, channels
    (u0, v0, ..., u7, v7) -- the KZ kp2d target convention (pre_processor.make_kp2d_map, no distortion)."""
    uv = proj(K, corners(t, lwh, R))
    c = proj(K, t)[0]
    return ((uv - c[None]) / KP_SCALE).reshape(16)


def pnp_img(img8, lwh, K):
    """Same solver chain as the deployed decoder (KFO.pnp: SQPNP then ITERATIVE refine), on explicit image points."""
    obj = KFO.PROTO8 * np.clip(np.abs(np.asarray(lwh, np.float64)), 0.02, 50.0)
    try:
        ok, rv, tv = cv2.solvePnP(obj.reshape(-1, 1, 3), np.asarray(img8, np.float64).reshape(-1, 1, 2), np.asarray(K, np.float64),
                                  np.zeros((1, 5)), flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        ok2, rv, tv = cv2.solvePnP(obj.reshape(-1, 1, 3), np.asarray(img8, np.float64).reshape(-1, 1, 2), np.asarray(K, np.float64),
                                   np.zeros((1, 5)), rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2:
            return None
    except cv2.error:
        return None
    return cv2.Rodrigues(rv)[0]


def decode_crop(target16_rho, rho, K_crop, t, lwh):
    """D_rho in the crop's own intrinsics: undo rho, add the projected centre, PnP with the deployed box."""
    x = np.asarray(target16_rho, np.float64) / C1.rho16(rho)
    c = proj(K_crop, t)[0]
    img = x.reshape(8, 2) * KP_SCALE + c[None]
    return pnp_img(img, lwh, K_crop)


def make_pair(kp16, r16, quotient):
    """kp16: (..., 16, h, w) KZ kp2d target map (offset / KP_SCALE at the centre cell, 0 elsewhere).
    Returns (..., 32, h, w) = [diag(rho) g, P diag(rho) g] when quotient else [diag(rho) g, diag(rho) g]."""
    kp16 = np.asarray(kp16, np.float32)
    shp = [1] * kp16.ndim
    shp[-3] = 16
    a = kp16 * np.asarray(r16, np.float32).reshape(shp)
    b = np.take(a, P16, axis=-3) if quotient else a
    return np.concatenate([a, b], axis=-3).astype(np.float32)


# ======================================================================================= LOO teacher (stage-1 rule)
def loo_teacher(D, i, fr, members, search, ks, rho):
    """Mirror of kzst1/kzst_common.frame_arms(..., loo=True) for one frame, additionally returning which fallback the
    4-member SA-TM used. R_T(k): the other 4 members, D_rho decoder, read cell = argmax of THEIR mean hm in the window.
    fallback chain inside KFO.fuse_subset: 4-member SA-TM -> 4-member C0 -> 5-member C0 (the last one contains k)."""
    t = C1.hook_top(D, i, fr)
    if t < 0:
        return None
    ctx = C1.make_ctx(D, i, fr['pred'][t, 3:6], ks, rho)
    cell = KFO.rgb_cell_of(D, i, search)
    Rdep = KFO.rot_from_eul(fr['pred'][t, 6:9])
    o, fb, plan, _ = KFO.fuse_subset(ctx, members, cell, ['C0', 'SATM'], Rdep)
    out = {'top': t, 'RT': [], 'quotient': [], 'stratum': [], 'fb_satm': [], 'level': [], 'ctx': ctx}
    for k in range(len(members)):
        sub = [members[j] for j in range(len(members)) if j != k]
        c_k = ctx.cell_for(sub)
        o3, fb3, pl3, _ = KFO.fuse_subset(ctx, sub, c_k, ['C0', 'SATM'], o['C0'])
        out['RT'].append(o3['SATM'])
        out['quotient'].append(bool(pl3 is not None and any(pl3['flip'][j] for j in pl3['keep'])))
        out['stratum'].append(pl3['stratum'] if pl3 is not None else 'degenerate')
        out['fb_satm'].append(bool(fb3['SATM']))
        level = 'satm' if not fb3['SATM'] else ('c0_4' if not fb3['C0'] else 'c0_5')
        out['level'].append(level)
    return out


def scene_of(seq):
    return C1.scene_of(seq)
