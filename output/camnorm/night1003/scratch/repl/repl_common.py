# -*- coding: utf-8 -*-
"""Experiments R (val replication of AdaBN + SA-TM) and C (ST2 positions + AdaBN KZ + SA-TM, descriptive): shared code.

Re-uses, unchanged, the byte copies in this folder of
  kz_fuse_offline.py  (e1sa SA-TM / PnP replica / metrics / ADS wrapper; sha256 9ae9e963...)
  kzst_common.py      (kzst1 frame_arms / hook_top / member_index; sha256 76353feb...)
  dump_kz_members.py  (e1sa recording hook; sha256 c25c6e33...)
  adabn_kz.py         (kzst1 crop replay: NpzSource / recut / forward / window / build_models / verify_replay; sha256 066984cf...)
Ground truth lives only in sealed_gt_<tag>.pkl (written by dump_members_repl.py) and is read only by load_frames(with_gt=True).
"""
import hashlib
import json
import os
import pickle
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(HERE)
ROOT = '@@ROOT@@'
TOOLS = ROOT + '/tools'
for _p in (HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np            # noqa: E402

import kzst_common as C       # noqa: E402  (byte copy of kzst1/kzst_common.py)

KFO = C.KFO
OUT_ROOT = '@@ROOT@@/output/camnorm/night1003/repl'
RELEASE_KZ = list(C.RELEASE_KZ)
ADABN_CKPTS = [os.path.join(SCRATCH, 'kzst1', 'adabn', 'kz_s%d_adabn.pth' % k) for k in range(5)]
ADABN_STAGE1_JSON = os.path.join(SCRATCH, 'kzst1', 'adabn', 'kz_members_adabn_test.json')
RAMP20 = [ROOT + '/output/release_20260920/ramp_s%d.pth' % k for k in range(5)]
ST2 = [ROOT + '/output/release_20260924/st2_s%d.pth' % k for k in range(5)]
# stage-1 / e1sa TEST inputs (pure-sim positions): used as the reference pipe of experiment C and as the self-test fixture
E1_NPZ = os.path.join(SCRATCH, 'e1sa', 'kz_members_e1sa.npz')
E1_DUMP = os.path.join(SCRATCH, 'e1sa', 'real_ens5flip_kpKZavg_zoom_e1sa.pkl')
S1_BN_NPZ = os.path.join(SCRATCH, 'kzst1', 'adabn', 'kz_members_adabn_test.npz')
S1_RESULTS = os.path.join(SCRATCH, 'kzst1', 'results_stage1.json')
EXPECTED = {'byte_copies': {'kz_fuse_offline.py': '9ae9e963eec9dfb14c8c1b6d027ff65ab9891c90e5228960aab8db4f48d80f2f',
                            'kzst_common.py': '76353febd5b9990a9c6f66d8435aee88ae8be37f69a081fe39a8fdaae98aa512',
                            'dump_kz_members.py': 'c25c6e33f67287d3af37e6a427891a8ab8a33cbaaacdac204d8252fa58ca236d',
                            'adabn_kz.py': '066984cf07a8536f0c93fa80a5e8f4b703dc2a5502a2383d93aa701b1acdfed9'}}


# ======================================================================================= io
def sha256(path):
    return KFO.sha256(path)


def jdump(obj, path):
    KFO.jdump(obj, path)


def say(msg, log=None):
    KFO.say(msg, log)


def check_byte_copies():
    bad = {f: sha256(os.path.join(HERE, f)) for f, h in EXPECTED['byte_copies'].items() if sha256(os.path.join(HERE, f)) != h}
    if bad and os.environ.get('UAV_STRICT_HASH') == '1':  # [release] paths rewritten -> copy hashes change; check only when strict
        raise SystemExit('byte copies modified: %s' % bad)
    return dict(EXPECTED['byte_copies'])


def under_out_root(path):
    p = os.path.normcase(os.path.abspath(path))
    return p.startswith(os.path.normcase(os.path.abspath(OUT_ROOT)))


def check_prereg(path):
    """-> (sha256, prereg dict); SystemExit when missing or modified (recorded hash in <path>.sha256)."""
    if not os.path.exists(path):
        raise SystemExit('%s missing -- run make_prereg_R.py first (before any val ground truth is read)' % path)
    h = sha256(path)
    rec = open(path + '.sha256', encoding='utf-8').read().split()[0]
    if h != rec and os.environ.get('UAV_STRICT_HASH') == '1':  # [release]
        raise SystemExit('%s hash mismatch (%s vs recorded %s) -- the pre-registration was modified' % (path, h, rec))
    return h, json.load(open(path, encoding='utf-8'))


def tag_paths(tag, out_root=OUT_ROOT):
    d = os.path.join(out_root, tag)
    return {'dir': d, 'pkl': os.path.join(d, 'real_ens5flip_%s.pkl' % tag), 'sealed': os.path.join(d, 'sealed_gt_%s.pkl' % tag),
            'npz': os.path.join(d, 'kz_members_%s.npz' % tag), 'json': os.path.join(d, 'kz_members_%s.json' % tag)}


# ======================================================================================= ground-truth sealing
def seal_gt(pkl, sealed_path, split, interval):
    """Moves gt / matched / det_rate of a dump_real_preds pkl into sealed_path; the pkl keeps only label-free fields.
    Nothing is printed or returned that depends on the labels."""
    D = pickle.load(open(pkl, 'rb'))
    S = {'info': {'split': split, 'sampled_interval': int(interval), 'n': len(D['frames']), 'det_rate': D['info'].get('det_rate'),
                  'source_pkl': os.path.basename(pkl), 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')},
         'frames': {}}
    for f in D['frames']:
        S['frames'][(str(f['seq']), str(f['frame']))] = {'gt': f['gt'], 'matched': f['matched'], 'cache_idx': f.get('cache_idx')}
        f['gt'] = None
        f['matched'] = None
    D['info']['det_rate'] = None
    D['info']['gt_sealed_in'] = os.path.basename(sealed_path)
    D['info']['sampled_interval'] = int(interval)
    with open(sealed_path + '.tmp', 'wb') as f:
        pickle.dump(S, f)
    os.replace(sealed_path + '.tmp', sealed_path)
    with open(pkl + '.tmp', 'wb') as f:
        pickle.dump(D, f)
    os.replace(pkl + '.tmp', pkl)
    return len(D['frames'])


def load_frames(pkl, sealed=None, with_gt=False):
    """{(seq, frame): frame}. with_gt=False: label-free fields only. with_gt=True: gt / matched from the sealed file, or
    (sealed=None, fixtures / stage-1 test pkls only) from the pkl itself."""
    fr, info = KFO.load_dump(pkl, with_gt=False)
    info = {k: v for k, v in info.items() if k != 'det_rate'}
    if not with_gt:
        return fr, info
    if sealed is None:
        g, _ = KFO.load_dump(pkl, with_gt=True)
        for k, f in fr.items():
            f['gt'], f['matched'] = g[k]['gt'], g[k]['matched']
        assert all(f['gt'] is not None for f in fr.values()), 'pkl carries no ground truth and no sealed file was given'
        return fr, info
    S = pickle.load(open(sealed, 'rb'))
    assert set(S['frames']) == set(fr), 'sealed ground truth does not cover exactly the dump frames'
    for k, f in fr.items():
        f['gt'] = np.asarray(S['frames'][k]['gt'], np.float64)
        f['matched'] = S['frames'][k]['matched']
    info['sealed_info'] = {k: v for k, v in S['info'].items() if k != 'det_rate'}
    return fr, info


# ======================================================================================= label-free arms and checks
POS_FIELDS = ('seq', 'frame', 'pos_rc', 'center_res', 'hm_peak', 'u', 'v', 'w2d', 'z', 'crop', 'uc_vc', 'rc_cc', 'win', 'raw_img_wh',
              'K_raw_src', 'K_in', 'dist', 'raw_size', 'new_size', 'stride_b', 'known_size')


def assert_same_positions(Dk, Db):
    """The BN member npz must carry exactly the positions / crops / geometry of the KZ npz (replay copies them)."""
    for k in POS_FIELDS:
        if k in Dk and k in Db:
            assert np.array_equal(Dk[k], Db[k]), 'field %s differs between the KZ and the BN member files' % k
    assert len(Dk['seq']) == len(Db['seq'])


def labelfree_checks(D, info, frames, log=None):
    """e1sa check (i)-style replication of the deployed decoder on this dump (no ground truth):
    top conf == hook peak, PnP(recorded deployed vector) == dumped rotation, offline 5-member C0 == dumped rotation."""
    members = C.member_index(info)
    search = int(info.get('search', 2))
    ks = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    N = len(D['seq'])
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    assert len(set(keys)) == N and set(keys) == set(frames), 'recorded member frames != dump frames'
    n_top = n_hook = n_fail_dep = 0
    d_dep, d_c0 = [], []
    for i, k in enumerate(keys):
        fr = frames[k]
        t = KFO.top_index(fr)
        if t < 0:
            continue
        n_top += 1
        if abs(fr['conf'][t] - D['hm_peak'][i]) >= 1e-5:
            continue
        n_hook += 1
        ctx = KFO.frame_ctx(D, i, fr['pred'][t, 3:6], ks)
        Rdep = KFO.rot_from_eul(fr['pred'][t, 6:9])
        Rp = KFO.pnp(D['kp_deployed'][i], ctx.g)
        if Rp is None:
            n_fail_dep += 1
            continue
        d_dep.append(KFO.geodesic(Rp, Rdep))
        cell = KFO.rgb_cell_of(D, i, search)
        Rc = KFO.pnp(KFO.f32(KFO.mean_vec([ctx.x(m, cell) for m in members])), ctx.g)
        d_c0.append(KFO.geodesic(Rc, Rdep) if Rc is not None else np.inf)
    d_dep, d_c0 = np.asarray(d_dep), np.asarray(d_c0)
    r = {'n_frames': N, 'n_top': n_top, 'top_at_hook_frac': n_hook / max(n_top, 1), 'n_deployed_pnp_failed': n_fail_dep,
         'pnp_deployed_vec_frac_lt_1e-3': float((d_dep < 1e-3).mean()) if len(d_dep) else None,
         'offline_C0_frac_lt_0p05': float((d_c0 < 0.05).mean()) if len(d_c0) else None,
         'offline_C0_median_deg': float(np.median(d_c0)) if len(d_c0) else None}
    # (frames whose top detection is not at the hook cell stay unchanged in every arm; 0 of 4800 on test)
    r['pass'] = bool(len(d_dep) > 0 and r['top_at_hook_frac'] >= 0.99 and r['pnp_deployed_vec_frac_lt_1e-3'] >= 0.999
                     and r['offline_C0_frac_lt_0p05'] >= 0.999)
    say('[check] %s' % r, log)
    return r


def compute_arms(D, info, frames, members=None):
    """Per frame (npz order): SA-TM and C0 of the 5 members (deployed decoder, recorded read cell), E5 stratum, top index.
    Frames without a detection at the hook cell keep the dumped rotation (top = -1, never changed)."""
    members = C.member_index(info) if members is None else members
    search = int(info.get('search', 2))
    ks = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    N = len(keys)
    o = {'SATM': np.full((N, 3, 3), np.nan), 'C0': np.full((N, 3, 3), np.nan), 'top': np.full(N, -1, np.int64),
         'stratum': np.full(N, 'no_det', dtype='<U12'), 'fb_SATM': np.zeros(N, bool), 'fb_C0': np.zeros(N, bool)}
    for i, k in enumerate(keys):
        a = C.frame_arms(D, i, frames[k], members, search, ks)
        if a is None:
            continue
        o['top'][i] = a['top']
        o['SATM'][i], o['C0'][i], o['stratum'][i] = a['SATM'], a['C0'], a['stratum']
        o['fb_SATM'][i], o['fb_C0'][i] = a['fb_SATM'], a['fb_C0']
    return keys, o


def deployed_rot(frames, keys, top):
    """The rotation the dump itself carries for the top detection (= the deployed KZ C0 for hook frames)."""
    R = np.full((len(keys), 3, 3), np.nan)
    for i, k in enumerate(keys):
        if top[i] >= 0:
            R[i] = KFO.rot_from_eul(frames[k]['pred'][top[i], 6:9])
    return R


# ======================================================================================= metrics (read ground truth)
def _sd():
    for p in (ROOT, TOOLS, TOOLS + '/annot_audit'):
        if p not in sys.path:
            sys.path.insert(0, p)
    import summarize_dumps as SD
    return SD


def arm_metrics(frames_gt, keys, Rarm, top, cls):
    """KZ_SUMMARY numbers of one arm: summarize_dumps.stats (det / position / heading fold-180 / tilt) + per-frame metrics."""
    SD = _sd()
    F = KFO.build_arm_frames(frames_gt, keys, Rarm, top)
    M = KFO.per_frame_metrics(F)
    unf = np.full(len(F), np.nan)
    for i, f in enumerate(F):
        if f['matched'] is not None:
            E = KFO.rot_from_eul(f['gt'][6:9]).T @ KFO.rot_from_eul(f['pred'][f['matched']][6:9])
            unf[i] = abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
    M['ang_unf'] = unf
    tab = {}
    for c, sel in (('all', np.ones(len(keys), bool)), ('phantom4', cls == 'phantom4'), ('mavic2', cls == 'mavic2')):
        if not sel.any():
            continue
        s = SD.stats([F[i] for i in np.where(sel)[0]])
        t = KFO.summary_numbers(M, sel)
        hit = np.isfinite(M['he']) & sel
        tab[c] = {'n': int(sel.sum()), 'det': s['det'], 'pos_median': float(s['pos']), 'pos_lt20': s['lt20'], 'pos_lt50': s['lt50'],
                  'head': float(s['head']), 'head_lt10': t['head_lt10'], 'head_lt20': t['head_lt20'], 'tilt': float(s['tilt']),
                  'acc_10deg10cm': t['acc_10deg10cm'], 'add_0p5d': t['add_0p5d'],
                  'flip_rate_unf_gt90': float(np.mean(unf[hit] > 90.0)) if hit.any() else float('nan')}
        assert abs(float(np.median(M['he'][hit])) - s['head']) < 1e-9
    return F, M, tab


def ads(F, W, H, out_dir, tag, profiles=('indoor', 'laa')):
    """Repository LAA3D_ADS (tools/eval_camnorm.ads_eval, same call as KFO.ads_indoor) for several profiles."""
    os.environ.setdefault('MPLBACKEND', 'Agg')
    for p in (ROOT, TOOLS):
        if p not in sys.path:
            sys.path.insert(0, p)
    import eval_camnorm as EC
    recs = [{'gt': np.asarray(f['gt'], np.float64).reshape(-1, 9), 'pred': f['pred'], 'conf': f['conf'], 'K': f['K_in'],
             'seq_id': f['seq'], 'frame_id': f['frame']} for f in F]
    raw_whs = [tuple(f['raw_wh']) for f in F]
    out = {}
    try:
        rep = EC.ads_eval(recs, raw_whs, float(W), float(H), KFO.SEQ, list(profiles), out_dir, tag)
        for p in profiles:
            m = re.search(r'LAA3D_ADS_\S+ \(%\) : ([0-9.]+)', rep[p])
            out[p] = float(m.group(1)) if m else None
    except Exception as e:      # noqa: BLE001
        print('[ads] failed for %s: %r' % (tag, e), flush=True)
        for p in profiles:
            out.setdefault(p, None)
    return out


def seq_cluster_ci(he_a, he_b, seqs, B, seed):
    """Sequence-cluster bootstrap of median(he_a) - median(he_b) over the same frames (KFO.boot_draws / cluster_boot)."""
    groups, draws = KFO.boot_draws(seqs, B, seed)

    def st(ii):
        return float(np.median(he_a[ii]) - np.median(he_b[ii]))
    bs = KFO.cluster_boot(st, seqs, groups, draws)
    per_seq = {s: st(np.where(seqs == s)[0]) for s in groups}
    return {'delta': st(np.arange(len(he_a))), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'p_one_sided_ge0': float((1 + np.sum(bs >= 0)) / (len(bs) + 1)), 'per_seq_delta': per_seq,
            'per_seq_n': {s: int((seqs == s).sum()) for s in groups}, 'n_seq': len(groups),
            'n_seq_better': int(sum(v < 0 for v in per_seq.values())), 'B': int(B), 'seed': int(seed)}
