# -*- coding: utf-8 -*-
"""KZ-ST v2, stage 1 -- shared definitions (CPU / numpy only).

Every rule that produces a pre-registered number lives here and is quoted in prereg_stage1.json:
  * per-frame member PnP / E5 strata / SA-TM, re-using the byte-identical copy of e1sa/kz_fuse_offline.py (KFO);
  * D_rho decoder (u / rho_u, v / rho_v before the deployed PnP), bit-identical to the deployed decoder for rho = None;
  * the label-free train-side gate statistics (verbatim replica of author_kzst/test_ref_gated.py / test_ref_labelfree.py);
  * g0 (verbatim replica of tools/make_pseudo_index.py, the ST1 acceptance rule), uniform per-sequence cap, sample sizes;
  * rho_c estimation (label-free, like r3_alt/diag_scale.py but per member at the common read cell).
Nothing in this module reads a ground-truth field.
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(HERE)
E1 = os.path.join(SCRATCH, 'e1sa')
AUTHOR = os.path.join(SCRATCH, 'author_kzst')
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np           # noqa: E402
import cv2                   # noqa: E402

import kz_fuse_offline as KFO   # noqa: E402  (byte copy of e1sa/kz_fuse_offline.py, sha256 in the prereg)

ROOT = '@@ROOT@@'
TOOLS = ROOT + '/tools'
PREREG = os.path.join(HERE, 'prereg_stage1.json')
RELEASE_KZ = [ROOT + '/output/release_20260923/kz_s%d.pth' % i for i in range(5)]
RGB = list(KFO.RGB_NAMES)
KP_CFG = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml'
SIM_KP_CFG = 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom.yaml'

ST1_TRAIN_DUMP = ROOT + '/output/camnorm/st/real_train_preds.pkl'
ST1_PSEUDO_INDEX = '@@MMCACHE@@/mav6d_cn_obj/train/index_pseudo_st1.pkl'
TEST_NPZ = os.path.join(E1, 'kz_members_e1sa.npz')
TEST_DUMP = os.path.join(E1, 'real_ens5flip_kpKZavg_zoom_e1sa.pkl')
TEST_FROZEN = os.path.join(E1, 'frozen_e1sa.npz')
TEST_RESULTS_E5 = os.path.join(E1, 'results_e1sa.json')
TEST_REF_GATED = os.path.join(AUTHOR, 'test_ref_gated.json')
TEST_REF_LABELFREE = os.path.join(AUTHOR, 'test_ref_labelfree.json')

TRAIN_TAG = 'train5'
CLASSES = ('phantom4', 'mavic2')
GATED = ('agree', 'flip_only')
STRATA = ('agree', 'flip_only', 'trim_only', 'both', 'degenerate', 'no_det')
GATE = {'share_drop': 0.10, 'jump_factor': 1.3, 'pairwise_factor': 1.3, 'gap_raw_frames': 2}
MIN_SAMPLE = {'total': 800, 'n_seq': 20, 'per_class': 300}
G0 = {'conf': 0.5, 'win': 3, 'min_nb': 2, 'pos_tol': 0.3}
CAP_PER_SEQ = 80
ADABN = {'batch': 16, 'order_seed': 0, 'cap_per_seq': 80}


# ======================================================================================= io / prereg
def sha256(path):
    return KFO.sha256(path)


def check_prereg(path=None):
    """-> (sha256, prereg dict); SystemExit when prereg_stage1.json is missing or was modified."""
    path = path or PREREG
    if not os.path.exists(path):
        raise SystemExit('%s missing -- run make_prereg_stage1.py first (before any ground truth is read)' % path)
    h = sha256(path)
    rec = open(path + '.sha256', encoding='utf-8').read().split()[0]
    if h != rec and os.environ.get('UAV_STRICT_HASH') == '1':  # [release]
        raise SystemExit('%s hash mismatch (%s vs recorded %s) -- the pre-registration was modified' % (path, h, rec))
    return h, json.load(open(path, encoding='utf-8'))


def jdump(obj, path):
    KFO.jdump(obj, path)


def load_npz_info(npz_path):
    return KFO.load_members(npz_path)


def load_dump_nogt(path):
    """Label-free dump reader: KFO.load_dump(with_gt=False) drops gt / matched right after unpickling."""
    fr, info = KFO.load_dump(path, with_gt=False)
    for v in fr.values():
        assert 'gt' not in v and 'matched' not in v
    info = {k: v for k, v in info.items() if k != 'det_rate'}      # det_rate is a ground-truth number -> never shown
    return fr, info


# ======================================================================================= small geometry
def fk_digits(f):
    """Frame order key of author_kzst/test_ref_gated.py (digits of the file name; mavic2 frames are ns timestamps)."""
    return int(''.join(c for c in str(f) if c.isdigit()) or 0)


def hd(Rm):
    """Camera-relative heading of the body x axis (make_pseudo_index.heading_deg / test_ref_gated.hd)."""
    return float(np.degrees(np.arctan2(Rm[0, 0], Rm[2, 0])))


def f180(d):
    d = abs((d + 180.0) % 360.0 - 180.0)
    return min(d, 180.0 - d)


def rho16(rho):
    r = np.asarray(rho, np.float64).reshape(2)
    return np.tile(r, 8)          # kp2d channels are (u0, v0, u1, v1, ...)


class RhoCtx(KFO.FrameCtx):
    """KFO.FrameCtx whose member vectors are decoded with D_rho (u / rho_u, v / rho_v, applied to the float64 vector
    x = kp / z before the float32 cast and the deployed PnP). rho=None is the deployed decoder (bit-identical code path)."""

    def __init__(self, hm_win, kp_win, z, geo, rho=None):
        super().__init__(hm_win, kp_win, z, geo)
        self.r16 = None if rho is None else rho16(rho)

    def x(self, m, cell):
        key = (m, cell)
        if key not in self._x:
            v = self.kp[m, :, cell[0], cell[1]].astype(np.float64) / self.z
            if self.r16 is not None:
                v = v / self.r16
            self._x[key] = v
        return self._x[key]


def make_ctx(D, i, lwh, kp_scale, rho=None):
    return RhoCtx(D['hm_win'][i], D['kp_win'][i], D['z'][i], KFO.frame_geo(D, i, lwh, kp_scale), rho)


def member_index(info, names=RGB):
    mn = list(info.get('member_names', names))
    return [mn.index(n) for n in names]


def hook_top(D, i, fr):
    """Top detection index if it is the cell the KZ hook wrote kp2d into (E5 rule), else -1 (frame left unchanged)."""
    t = KFO.top_index(fr)
    if t < 0 or abs(fr['conf'][t] - D['hm_peak'][i]) >= 1e-5:
        return -1
    return t


def frame_arms(D, i, fr, members, search, kp_scale, rho=None, selfpeak=False, loo=False):
    """All label-free per-frame quantities for one frame (E5 definitions). Returns None when the top detection is not
    at the hook cell (no detection / rot-head rotation) -- such frames are 'no_det' and never change."""
    t = hook_top(D, i, fr)
    if t < 0:
        return None
    ctx = make_ctx(D, i, fr['pred'][t, 3:6], kp_scale, rho)
    cell = KFO.rgb_cell_of(D, i, search)
    Rdep = KFO.rot_from_eul(fr['pred'][t, 6:9])
    o, fb, plan, (xs, Rs) = KFO.fuse_subset(ctx, members, cell, ['C0', 'SATM'], Rdep)
    out = {'top': t, 'cell': cell, 'C0': o['C0'], 'SATM': o['SATM'], 'fb_C0': fb['C0'], 'fb_SATM': fb['SATM'],
           'stratum': plan['stratum'] if plan is not None else 'degenerate', 'plan': plan, 'member_R': Rs, 'ctx': ctx,
           'Rdep': Rdep}
    if selfpeak:
        sp = []
        for m in members:
            R_ = ctx.R(m, ctx.cell_for([m]))
            sp.append((R_, False) if R_ is not None else (o['C0'], True))
        out['selfpeak'] = sp
    if loo:
        RT, qf, ls = [], [], []
        for k in range(len(members)):
            sub = [members[j] for j in range(len(members)) if j != k]
            c_k = ctx.cell_for(sub)
            o3, fb3, pl3, _ = KFO.fuse_subset(ctx, sub, c_k, ['C0', 'SATM'], o['C0'])
            RT.append(o3['SATM'])
            qf.append(bool(pl3 is not None and any(pl3['flip'][j] for j in pl3['keep'])))
            ls.append(pl3['stratum'] if pl3 is not None else 'degenerate')
        out['loo_RT'], out['loo_quotient'], out['loo_stratum'] = RT, qf, ls
    return out


# ======================================================================================= gate statistics
def gate_stats(cls, seq, fk, stratum, R_satm, member_R, gap_ranks=2, classes=CLASSES):
    """Replica of author_kzst/test_ref_gated.py (+ the gated share of test_ref_labelfree.py), per class:
      gated_share  = share of the class's dumped frames whose 5-member E5 stratum is agree or flip_only (denominator =
                     all dumped frames of the class, as on test)
      jump         = within each sequence, frames in time order (fk_digits), pairs (r, r + gap_ranks) with BOTH frames
                     gated: f180(hd(R_SATM[a]) - hd(R_SATM[b])); median over all pairs of the class
      pairwise     = per gated frame, median over the 10 member pairs of fold(rel_yaw(R_m, R_n)) (common read cell,
                     deployed decoder); median over the gated frames of the class
    'all_det' (report only): the same with every frame that has a usable top detection."""
    gated = np.isin(stratum, GATED)
    has = stratum != 'no_det'
    out = {}
    for c in classes:
        sel = cls == c
        o = {'n_frames': int(sel.sum()), 'n_seq': int(len(set(seq[sel]))),
             'strata': {s: float((stratum[sel] == s).mean()) if sel.any() else None for s in STRATA},
             'gated_share': float(gated[sel].mean()) if sel.any() else float('nan')}
        for tag, G in (('gated', gated), ('all_det', has)):
            J = []
            for s in sorted(set(seq[sel])):
                idx = np.where(seq == s)[0]
                idx = idx[np.argsort(fk[idx], kind='stable')]
                for r in range(len(idx) - gap_ranks):
                    a, b = idx[r], idx[r + gap_ranks]
                    if G[a] and G[b]:
                        J.append(f180(hd(R_satm[a]) - hd(R_satm[b])))
            pdis = []
            for i in np.where(sel & G)[0]:
                Ms = [member_R[i, m] for m in range(member_R.shape[1]) if np.isfinite(member_R[i, m]).all()]
                if len(Ms) < 2:
                    continue
                pdis.append(np.median([KFO.fold(KFO.rel_yaw(Ms[x], Ms[y])) for x in range(len(Ms)) for y in range(x + 1, len(Ms))]))
            o[tag] = {'n': int((sel & G).sum()), 'jump_median': float(np.median(J)) if J else float('nan'), 'n_pairs': len(J),
                      'jump_gt20': float(np.mean(np.array(J) > 20)) if J else float('nan'),
                      'pairwise_fold_median': float(np.median(pdis)) if pdis else float('nan')}
        out[c] = o
    return out


def gate_bounds(ref_gated, ref_labelfree, gate=GATE):
    """Frozen thresholds from the frozen test references (values exactly as stored in the two JSON files)."""
    out = {}
    for c in CLASSES:
        sh = float(ref_labelfree[c]['agree+flip_only'])
        jp = float(ref_gated[c]['gated']['jump_gap2_median'])
        pw = float(ref_gated[c]['gated']['pairwise_fold_median'])
        out[c] = {'test_gated_share': sh, 'test_jump_gap2_median': jp, 'test_pairwise_fold_median': pw,
                  'share_min': sh - gate['share_drop'], 'jump_max': gate['jump_factor'] * jp,
                  'pairwise_max': gate['pairwise_factor'] * pw}
    return out


def gate_verdict(stats, bounds):
    v = {}
    for c in CLASSES:
        s, b = stats[c], bounds[c]
        r = {'share': s['gated_share'] >= b['share_min'],
             'jump': s['gated']['jump_median'] <= b['jump_max'],
             'pairwise': s['gated']['pairwise_fold_median'] <= b['pairwise_max']}
        r = {k: bool(x) for k, x in r.items()}
        r['pass'] = bool(all(r.values()))
        r['values'] = {'gated_share': s['gated_share'], 'jump_median': s['gated']['jump_median'],
                       'pairwise_fold_median': s['gated']['pairwise_fold_median'], 'n_gated': s['gated']['n'],
                       'n_jump_pairs': s['gated']['n_pairs']}
        r['ratios_to_test'] = {'share_minus_test': s['gated_share'] - b['test_gated_share'],
                               'jump_ratio': s['gated']['jump_median'] / b['test_jump_gap2_median'],
                               'pairwise_ratio': s['gated']['pairwise_fold_median'] / b['test_pairwise_fold_median']}
        v[c] = r
    if not v['phantom4']['pass']:
        disp = 'no_LOO'
    elif v['mavic2']['pass']:
        disp = 'train_both'
    else:
        disp = 'train_phantom4_only'
    return v, disp


# ======================================================================================= g0 / selection
def g0_mask(seq, fk, conf_top, pos_top, rule=G0):
    """Verbatim replica of tools/make_pseudo_index.py steps 1-2 (the ST1 acceptance rule): top-1 conf >= 0.5; within the
    sequence in frame order, the other frames within +-3 ranks that pass the conf rule are the neighbours; < 2 neighbours
    -> reject; |p - median(neighbour positions)| > 0.3 m -> reject. conf_top = NaN for frames without a prediction."""
    N = len(seq)
    ok_conf = np.isfinite(conf_top) & (np.nan_to_num(conf_top, nan=-1.0) >= rule['conf'])
    acc = np.zeros(N, bool)
    reasons = {'no_pred_or_low_conf': 0, 'few_neighbors': 0, 'pos_jump': 0, 'accepted': 0}
    for s in sorted(set(seq)):
        idx = np.where(seq == s)[0]
        idx = idx[np.argsort(fk[idx], kind='stable')]
        for k, i in enumerate(idx):
            if not ok_conf[i]:
                reasons['no_pred_or_low_conf'] += 1
                continue
            nb = [idx[j] for j in range(max(0, k - rule['win']), min(len(idx), k + rule['win'] + 1)) if j != k and ok_conf[idx[j]]]
            if len(nb) < rule['min_nb']:
                reasons['few_neighbors'] += 1
                continue
            med = np.median(pos_top[nb], axis=0)
            if np.linalg.norm(pos_top[i] - med) > rule['pos_tol']:
                reasons['pos_jump'] += 1
                continue
            acc[i] = True
            reasons['accepted'] += 1
    return acc, reasons


def cap_uniform(idx_sorted, cap):
    """At most `cap` frames per sequence, uniform in time: positions round(linspace(0, n-1, cap))."""
    idx_sorted = list(idx_sorted)
    n = len(idx_sorted)
    if n <= cap:
        return idx_sorted
    pick = np.round(np.linspace(0, n - 1, cap)).astype(int)
    assert len(set(pick.tolist())) == cap
    return [idx_sorted[j] for j in pick]


def select_per_seq(mask, seq, fk, cap=CAP_PER_SEQ):
    sel = []
    for s in sorted(set(seq[mask])):
        idx = np.where(mask & (seq == s))[0]
        idx = idx[np.argsort(fk[idx], kind='stable')]
        sel += cap_uniform(idx, cap)
    return np.array(sorted(sel), np.int64)


def frame_table(D, frames):
    """Label-free per-frame table in npz order: seq, frame, cls, fk, top-1 conf / position (any cell, as ST1), hook flag."""
    seq = np.array([str(s) for s in D['seq']])
    frm = np.array([str(f) for f in D['frame']])
    N = len(seq)
    keys = list(zip(seq.tolist(), frm.tolist()))
    assert len(set(keys)) == N and set(keys) == set(frames.keys()), 'recorded frames != dump frames'
    cls = np.array([frames[k]['cls'] for k in keys])
    fk = np.array([fk_digits(f) for f in frm], dtype=np.int64)
    conf_top = np.full(N, np.nan)
    pos_top = np.full((N, 3), np.nan)
    hook = np.zeros(N, bool)
    for i, k in enumerate(keys):
        fr = frames[k]
        t = KFO.top_index(fr)
        if t >= 0:
            conf_top[i] = float(fr['conf'][t])
            pos_top[i] = fr['pred'][t, :3]
        hook[i] = hook_top(D, i, fr) >= 0
    return {'seq': seq, 'frame': frm, 'keys': keys, 'cls': cls, 'fk': fk, 'conf_top': conf_top, 'pos_top': pos_top, 'hook': hook}


def adabn_selection(D, frames):
    """AdaBN re-estimation frames (train only): g0 AND top detection at the hook cell, <= 80 per sequence uniform in time,
    both classes, every sequence. Returns (indices into the npz order, frame table, g0 mask, g0 reasons)."""
    T = frame_table(D, frames)
    g0, reasons = g0_mask(T['seq'], T['fk'], T['conf_top'], T['pos_top'])
    sel = select_per_seq(g0 & T['hook'], T['seq'], T['fk'], ADABN['cap_per_seq'])
    return sel, T, g0, reasons


def sample_size_check(sel, cls, seq, trained_classes, rule=MIN_SAMPLE):
    per = {c: int((cls[sel] == c).sum()) for c in CLASSES}
    tot = int(len(sel))
    nseq = int(len(set(seq[sel])))
    ok = {'total': tot >= rule['total'], 'n_seq': nseq >= rule['n_seq'],
          'per_class': all(per[c] >= rule['per_class'] for c in trained_classes)}
    return {'n_total': tot, 'n_seq': nseq, 'per_class': per, 'trained_classes': list(trained_classes),
            'checks': ok, 'pass': bool(all(ok.values()) and len(trained_classes) > 0)}


# ======================================================================================= rho (label-free)
def rho_ratios(ctx, members, cell, t_pos):
    """Per member (deployed decoder, common read cell): span of its own 8 projected corners (u, v) divided by the span
    of the rigid PnP box g.obj = PROTO8 * lwh placed at the position network's centre t with the member's own PnP
    rotation, projected with the decoder's K_in / distortion. Returns a list aligned with `members`: (ratio_u, ratio_v)
    or None (member PnP failed / box behind the camera / degenerate span)."""
    out = []
    for m in members:
        R = ctx.R(m, cell)
        if R is None:
            out.append(None)
            continue
        img = ctx.g.img(KFO.f32(ctx.x(m, cell)).astype(np.float64))
        C = ctx.g.obj @ R.T + np.asarray(t_pos, np.float64).reshape(3)
        if C[:, 2].min() <= 1e-6:
            out.append(None)
            continue
        pr, _ = cv2.projectPoints(C.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), ctx.g.K, ctx.g.dist)
        pr = pr.reshape(-1, 2)
        wb, hb = float(np.ptp(pr[:, 0])), float(np.ptp(pr[:, 1]))
        if wb < 1e-6 or hb < 1e-6:
            out.append(None)
            continue
        out.append((float(np.ptp(img[:, 0])) / wb, float(np.ptp(img[:, 1])) / hb))
    return out


def rigid_rms_rel(ctx, vec):
    """Reprojection RMS of a 16-d vector against its own rigid PnP fit (same box, same decoder), divided by the vector's
    2D width -- label-free 'rigid residual'."""
    sol = KFO.pnp(KFO.f32(vec), ctx.g, return_rt=True)
    if sol is None:
        return float('nan')
    _, rv, tv = sol
    img = ctx.g.img(KFO.f32(vec).astype(np.float64))
    pr, _ = cv2.projectPoints(ctx.g.obj.reshape(-1, 1, 3), rv, tv, ctx.g.K, ctx.g.dist)
    w = float(np.ptp(img[:, 0]))
    return float(np.sqrt(((pr.reshape(-1, 2) - img) ** 2).sum(1).mean())) / max(w, 1e-9)


def side_regime(h_deg):
    """0 = side view (heading axis along the camera x axis), 90 = along the optical axis (critic_kzst/view_regime.py)."""
    a = f180(h_deg - 90.0)
    return 'side' if a < 30 else ('along' if a > 60 else 'oblique')


def scene_of(seq):
    p = str(seq).split('/')
    return p[1] if len(p) >= 3 else '?'
