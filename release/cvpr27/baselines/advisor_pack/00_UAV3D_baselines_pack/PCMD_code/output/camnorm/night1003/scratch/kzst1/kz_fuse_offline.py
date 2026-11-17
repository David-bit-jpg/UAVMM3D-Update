# -*- coding: utf-8 -*-
"""E1 / SA-TM offline analysis (CPU only). Pre-registration: prereg.json in this folder (sha256 checked at every stage).

Stages (run in this order; each refuses to run if the previous one did not pass):
  check     label-free replication checks of the GPU dump (never touches gt / matched):
            (iii) positions / conf identical to the deployed KZ dump, (i) PnP replica + offline C0 reproduce the deployed
            rotation per frame, (v) read-out cell re-selection, (ii) single members (self-peak) reproduce the existing
            single-member dumps, (iv) two-member MM C0 reproduces the existing kd avg dumps
  freeze    label-free: per-frame member PnPs, medoid, kept set, strata, every arm's rotation, gate, structural asserts;
            writes frozen_<tag>.npz + .json with sha256
  evaluate  reads ground truth (first time): C0 reproduction gate, KZ_SUMMARY metrics (summarize_dumps / paper_metrics /
            ADS indoor), sequence-cluster bootstrap, Holm, sign counts, guardrails, deploy decision, report-only arms
  synth     signed-weight noise amplification demo (sigma = 10 deg, k = 0/1/2 exact Pi180 relabels), label-free
  selftest  CPU unit tests + a synthetic end-to-end run of check / freeze / evaluate on fabricated files

    python kz_fuse_offline.py check | freeze | evaluate          (defaults point at the GPU dump in this folder)
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('MPLBACKEND', 'Agg')           # LAA3D_ADS draws scatter plots; never open a GUI backend

import argparse      # noqa: E402
import hashlib       # noqa: E402
import json          # noqa: E402
import pickle        # noqa: E402
import re            # noqa: E402
import time          # noqa: E402

import cv2           # noqa: E402
import numpy as np   # noqa: E402
from scipy.spatial.transform import Rotation as Rot   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '@@ROOT@@'
TOOLS = ROOT + '/tools'
OLD_ROOT = ROOT + '/output/camnorm'
PREREG = os.path.join(HERE, 'prereg.json')
SEQ = 'xyz'
KP_SCALE_DEFAULT = 64.0
PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]], dtype=np.float64)
P180 = np.array([2, 3, 0, 1, 6, 7, 4, 5])
KEEP_FOLD = 45.0
FLIP_RAW = 90.0
RGB_NAMES = ['KZs%d' % i for i in range(5)]
MM_NAMES = ['KZKD_s0', 'KZKD_s1', 'KZKDir_s0', 'KZKDir_s1', 'KZMMH_s0', 'KZMMH_s1']
MM_PAIRS = [('KZKD', ('KZKD_s0', 'KZKD_s1')), ('KZKDir', ('KZKDir_s0', 'KZKDir_s1')), ('KZMMH', ('KZMMH_s0', 'KZMMH_s1'))]
# existing dumps (relative to --old-root) used by the replication checks
OLD_AVG = 'kz/real_ens5flip_kpKZavg_zoom.pkl'
OLD_SINGLE = {**{'KZs%d' % i: 'kz/real_ens5flip_kpKZs%d_zoom.pkl' % i for i in range(5)},
              'KZKD_s0': 'kd/real_ens5flip_kpKZKDs0_zoom.pkl', 'KZKD_s1': 'kd/real_ens5flip_kpKZKDs1_zoom.pkl',
              'KZKDir_s0': 'kd/real_ens5flip_kpKZKDirs0_zoom.pkl', 'KZKDir_s1': 'kd/real_ens5flip_kpKZKDirs1_zoom.pkl',
              'KZMMH_s0': 'kd/real_ens5flip_kpKZMMHs0_zoom.pkl', 'KZMMH_s1': 'kd/real_ens5flip_kpKZMMHs1_zoom.pkl'}
OLD_PAIR = {'KZKD': 'kd/real_ens5flip_kpKZKDavg_zoom.pkl', 'KZKDir': 'kd/real_ens5flip_kpKZKDiravg_zoom.pkl',
            'KZMMH': 'kd/real_ens5flip_kpKZMMHavg_zoom.pkl'}
PRIMARY_ARMS = ['C0', 'SATM', 'a_medoid', 'b_TMnoalign', 'c_coordmedian', 'd_geoL1', 'e_C1', 'f_C1b', 'g_C2', 'h_SATM_selfpeak']


# ======================================================================================= small utils
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def frame_key(f):
    stem = os.path.splitext(os.path.basename(str(f)))[0]
    return (0, int(stem), '') if stem.isdigit() else (1, 0, stem)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=conv)
    os.replace(path + '.tmp', path)


def say(msg, log=None):
    print(msg, flush=True)
    if log is not None:
        log.append(str(msg))


def rot_from_eul(e):
    return Rot.from_euler(SEQ, np.asarray(e, np.float64)).as_matrix()


def rel_yaw(Ra, Rb):
    """|atan2(E[1,0],E[0,0])| in degrees, E = Ra^T Rb (summarize_dumps heading definition)."""
    E = Ra.T @ Rb
    return abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))


def signed_yaw(Ra, Rb):
    E = Ra.T @ Rb
    return np.degrees(np.arctan2(E[1, 0], E[0, 0]))


def fold(a):
    return min(a, 180.0 - a)


def tilt_between(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip(Ra[:, 2] @ Rb[:, 2], -1, 1))))


def geodesic(Ra, Rb):
    """Rotation angle of Ra^T Rb in degrees via the chordal distance ||Ra - Rb||_F = 2 sqrt(2) sin(theta / 2)
    (accurate near 0, unlike arccos of the trace)."""
    d = np.linalg.norm(np.asarray(Ra) - np.asarray(Rb)) / (2.0 * np.sqrt(2.0))
    return float(np.degrees(2.0 * np.arcsin(np.clip(d, 0.0, 1.0))))


def relabel(x16, perm):
    return np.asarray(x16).reshape(8, 2)[perm].reshape(16)


def rz_perms():
    """SIG[k][j] = j' with Rz(90k deg) P_j = P_j' on the unit cube; INV[k] = argsort(SIG[k]).
    A member whose pose is R_ref Rz(90k) is re-labelled to the reference labelling by x[INV[k]]."""
    sig, inv = {}, {}
    for k in range(4):
        Rz = Rot.from_euler('z', 90 * k, degrees=True).as_matrix()
        Q = PROTO8 @ Rz.T
        s = np.array([int(np.argmin(np.linalg.norm(PROTO8 - q, axis=1))) for q in Q])
        assert np.allclose(PROTO8[s], Q, atol=1e-12) and len(set(s)) == 8
        sig[k], inv[k] = s, np.argsort(s)
    assert np.array_equal(inv[2], P180) and np.array_equal(sig[2], P180)
    return sig, inv


SIG90, INV90 = rz_perms()


# ======================================================================================= geometry / PnP replica
class Geo:
    """Per-frame decoder geometry. uv_raw is re-computed along the decoder's float32 path
    (object_encoder_mav6d.center_point_decoder: u_hm = cols.float() + res_x; uv_raw = u_hm / scale_x)."""

    def __init__(self, K, dist, raw_wh, new_wh, stride, rc, res, lwh, kp_scale=KP_SCALE_DEFAULT):
        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, np.float64).reshape(1, -1)
        raw_w, raw_h = float(raw_wh[0]), float(raw_wh[1])
        new_w, new_h = float(new_wh[0]), float(new_wh[1])
        scale_x = new_w / raw_w / float(stride)
        scale_y = new_h / raw_h / float(stride)
        r, c = int(rc[0]), int(rc[1])
        u_hm = np.array([c], np.float32) + np.array([res[0]], np.float32)
        v_hm = np.array([r], np.float32) + np.array([res[1]], np.float32)
        uv = np.stack([u_hm / scale_x, v_hm / scale_y], axis=1)          # float32 like the decoder
        self.uv = uv[0]
        self.sx = raw_w / new_w
        self.sy = raw_h / new_h
        self.lwh = np.asarray(lwh, np.float64).reshape(3)
        self.kp_scale = float(kp_scale)
        lw = np.clip(np.abs(self.lwh).astype(np.float64), 0.02, 50.0)
        self.obj = PROTO8 * lw

    def img(self, x16):
        p = np.asarray(x16, np.float64).reshape(8, 2) * self.kp_scale
        return np.stack([p[:, 0] * self.sx + self.uv[0], p[:, 1] * self.sy + self.uv[1]], axis=1)


def pnp(x16, g, return_rt=False):
    """Verbatim replica of center_point_decoder lines 435-468 (rotation part). x16 must already be the float32 value.
    Returns R (3x3) or None when the decoder would have skipped / failed."""
    img = g.img(np.asarray(x16, np.float32).astype(np.float64))
    obj = g.obj
    if not np.isfinite(img).all() or np.ptp(img[:, 0]) < 1e-3:
        return None
    if float(obj.var(axis=0).sum()) < 1e-6 or float(np.ptp(img[:, 1])) < 1e-3:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), g.K, g.dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 1e-6:
            return None
        ok2, rvec, tvec = cv2.solvePnP(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), g.K, g.dist, rvec, tvec,
                                       useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2 or not np.isfinite(tvec).all() or tvec[2, 0] <= 1e-6:
            return None
    except cv2.error:
        return None
    Rm = cv2.Rodrigues(rvec)[0]
    if return_rt:
        return Rm, rvec.copy(), tvec.copy()
    return Rm


def weighted_pnp(img_obs, w, g, rvec0, tvec0):
    """EPro-PnP eq.(1): minimise sum_i ||w_i (pi(R P_i + t) - x_i)||^2 (with distortion), LM from (rvec0, tvec0)."""
    from scipy.optimize import least_squares
    obj = g.obj.reshape(-1, 1, 3)

    def res(p):
        pr, _ = cv2.projectPoints(obj, p[:3].reshape(3, 1), p[3:].reshape(3, 1), g.K, g.dist)
        return (w[:, None] * (pr.reshape(-1, 2) - img_obs)).ravel()
    try:
        sol = least_squares(res, np.r_[rvec0.ravel(), tvec0.ravel()], method='lm', xtol=1e-12, ftol=1e-12, max_nfev=400)
    except Exception:      # noqa: BLE001
        return None
    p = sol.x
    if not np.isfinite(p).all() or p[5] <= 1e-6:
        return None
    return cv2.Rodrigues(p[:3].reshape(3, 1))[0]


def chordal_l2(Rs):
    u, _, vt = np.linalg.svd(np.mean(np.asarray(Rs), 0))
    M = u @ vt
    if np.linalg.det(M) < 0:
        u[:, -1] *= -1
        M = u @ vt
    return M


def geo_l1(Rs, iters=100):
    """Hartley et al. 2013 §5.4: geodesic L1 mean on SO(3) by Weiszfeld, initialised at the chordal L2 mean."""
    Rs = [np.asarray(R, np.float64) for R in Rs]
    if len(Rs) == 1:
        return Rs[0]
    Rm = chordal_l2(Rs)
    for _ in range(iters):
        vs = np.array([Rot.from_matrix(Rm.T @ Ri).as_rotvec() for Ri in Rs])
        n = np.linalg.norm(vs, axis=1)
        ok = n > 1e-9
        if not ok.any():
            break
        delta = (vs[ok] / n[ok, None]).sum(0) / (1.0 / n[ok]).sum()
        Rm = Rm @ Rot.from_rotvec(delta).as_matrix()
        if np.linalg.norm(delta) < 1e-10:
            break
    return Rm


# ======================================================================================= SA-TM and arms
def plan_satm(Rs, hv):
    """Rs: per-member rotation or None (PnP failed); hv: per-member hm at the read cell (medoid tie-break).
    Returns the SA-TM plan (prereg primary_arm_SATM steps 1-5) and the label-free stratum."""
    M = len(Rs)
    valid = [k for k in range(M) if Rs[k] is not None]
    if not valid:
        return None
    if len(valid) == 1:
        med = valid[0]
    else:
        F = np.array([[fold(rel_yaw(Rs[a], Rs[b])) for b in valid] for a in valid])
        tot = F.sum(1)
        cand = [valid[i] for i in range(len(valid)) if tot[i] <= tot.min() + 1e-9]
        best = max(hv[k] for k in cand)
        med = min(k for k in cand if hv[k] >= best)
    r = np.full(M, np.nan)
    for k in valid:
        r[k] = 0.0 if k == med else rel_yaw(Rs[med], Rs[k])
    f = np.minimum(r, 180.0 - r)
    keep = [k for k in valid if f[k] <= KEEP_FOLD]
    flip = np.array([bool(np.isfinite(r[k]) and r[k] > FLIP_RAW) for k in range(M)])
    others = [k for k in valid if k != med]
    n_flipk = sum(1 for k in others if r[k] >= 180.0 - KEEP_FOLD)
    n_trim = sum(1 for k in others if KEEP_FOLD < r[k] < 180.0 - KEEP_FOLD)
    if len(valid) < M or len(valid) < 2:
        stratum = 'degenerate'
    elif n_flipk == 0 and n_trim == 0:
        stratum = 'agree'
    elif n_trim == 0:
        stratum = 'flip_only'
    elif n_flipk == 0:
        stratum = 'trim_only'
    else:
        stratum = 'both'
    return {'medoid': med, 'valid': valid, 'r': r, 'f': f, 'keep': keep, 'flip': flip, 'stratum': stratum,
            'n_flipk': n_flipk, 'n_trim': n_trim}


def mean_vec(vs):
    return np.mean(np.stack(vs, 0), axis=0)


def satm_vec(xs, plan):
    al = [relabel(xs[k], P180) if plan['flip'][k] else xs[k] for k in plan['keep']]
    v = mean_vec(al)
    nfl = int(sum(plan['flip'][k] for k in plan['keep']))
    if nfl > len(plan['keep']) / 2.0:
        v = relabel(v, P180)
    return v


def c1_vec(xs, plan):
    al = [relabel(xs[k], P180) if plan['flip'][k] else xs[k] for k in plan['valid']]
    v = mean_vec(al)
    nfl = int(sum(plan['flip'][k] for k in plan['valid']))
    if nfl > len(plan['valid']) / 2.0:
        v = relabel(v, P180)
    return v


def c1b_vec(xs, Rs, plan):
    med = plan['medoid']
    al = []
    for k in plan['valid']:
        kk = int(np.round(signed_yaw(Rs[med], Rs[k]) / 90.0)) % 4
        al.append(relabel(xs[k], INV90[kk]))
    return mean_vec(al)


def f32(v):
    return np.asarray(v, np.float64).astype(np.float32)


class FrameCtx:
    """Member vectors / PnPs of one frame, cached per (member, cell)."""

    def __init__(self, hm_win, kp_win, z, geo):
        self.hm = hm_win.astype(np.float64)          # (M,S,S) NaN where clipped
        self.kp = kp_win                              # (M,16,S,S) float32 raw (before /z)
        self.z = float(z)
        self.g = geo
        self.valid_cell = np.isfinite(self.hm[0])
        self._x, self._R = {}, {}

    def cell_for(self, members):
        m = np.mean(self.hm[list(members)], axis=0)
        m = np.where(self.valid_cell, m, -np.inf)
        j = int(np.argmax(m.reshape(-1)))
        return (j // m.shape[1], j % m.shape[1])

    def x(self, m, cell):
        key = (m, cell)
        if key not in self._x:
            self._x[key] = self.kp[m, :, cell[0], cell[1]].astype(np.float64) / self.z
        return self._x[key]

    def R(self, m, cell):
        key = (m, cell)
        if key not in self._R:
            self._R[key] = pnp(f32(self.x(m, cell)), self.g)
        return self._R[key]

    def hmv(self, m, cell):
        return float(self.hm[m, cell[0], cell[1]])


def fuse_subset(ctx, members, cell, kinds, fallback_R, per_member_cells=None):
    """members: indices into the member axis. cell: common read cell (ignored when per_member_cells is given).
    kinds: subset of PRIMARY arm names (without the h_ prefix logic). Returns ({kind: R}, {kind: used_fallback}, plan)."""
    cells = per_member_cells if per_member_cells is not None else [cell] * len(members)
    xs = [ctx.x(m, c) for m, c in zip(members, cells)]
    Rs = [ctx.R(m, c) for m, c in zip(members, cells)]
    hv = [ctx.hmv(m, c) for m, c in zip(members, cells)]
    out, fb = {}, {}

    def solve(name, vec, fbR):
        R_ = pnp(f32(vec), ctx.g)
        out[name], fb[name] = (R_, False) if R_ is not None else (fbR, True)

    c0v = mean_vec(xs)
    if 'C0' in kinds:
        solve('C0', c0v, fallback_R)
    c0R = out.get('C0', fallback_R)
    plan = plan_satm(Rs, hv)
    if plan is None:
        for k in kinds:
            if k != 'C0':
                out[k], fb[k] = c0R, True
        return out, fb, None, (xs, Rs)
    if 'SATM' in kinds:
        solve('SATM', satm_vec(xs, plan), c0R)
    if 'a_medoid' in kinds:
        solve('a_medoid', xs[plan['medoid']], c0R)
    if 'b_TMnoalign' in kinds:
        solve('b_TMnoalign', mean_vec([xs[k] for k in plan['keep']]), c0R)
    if 'c_coordmedian' in kinds:
        solve('c_coordmedian', np.median(np.stack(xs, 0), axis=0), c0R)
    if 'd_geoL1' in kinds:
        out['d_geoL1'], fb['d_geoL1'] = geo_l1([Rs[k] for k in plan['valid']]), False
    if 'e_C1' in kinds:
        solve('e_C1', c1_vec(xs, plan), c0R)
    if 'f_C1b' in kinds:
        solve('f_C1b', c1b_vec(xs, Rs, plan), c0R)
    if 'g_C2' in kinds:
        sv = satm_vec(xs, plan)
        base = pnp(f32(sv), ctx.g, return_rt=True)
        if base is None:
            out['g_C2'], fb['g_C2'] = c0R, True
        elif len(plan['keep']) < 2:
            out['g_C2'], fb['g_C2'] = base[0], False
        else:
            al = [relabel(xs[k], P180) if plan['flip'][k] else xs[k] for k in plan['keep']]
            ims = np.stack([ctx.g.img(f32(a).astype(np.float64)) for a in al])        # (K,8,2) raw px
            sig = np.sqrt(ims[..., 0].var(0) + ims[..., 1].var(0))                      # (8,)
            nfl = int(sum(plan['flip'][k] for k in plan['keep']))
            if nfl > len(plan['keep']) / 2.0:
                sig = sig[P180]
            s0 = float(np.median(sig))
            if not s0 > 0:
                out['g_C2'], fb['g_C2'] = base[0], False
            else:
                w = 1.0 / (sig + s0)
                R_ = weighted_pnp(ctx.g.img(f32(sv).astype(np.float64)), w, ctx.g, base[1], base[2])
                out['g_C2'], fb['g_C2'] = (R_, False) if R_ is not None else (base[0], True)
    return out, fb, plan, (xs, Rs)


def compute_frame(ctx, names, deployed_R, rgb_cell):
    """All arms of one frame. rgb_cell = recorded deployed read cell (window coords).
    deployed_R = rotation of the top detection in the dump (fallback when the offline C0 PnP fails, as the decoder does)."""
    idx = {n: i for i, n in enumerate(names)}
    rgb = [idx[n] for n in RGB_NAMES]
    res = {'R': {}, 'fb': {}}
    o, fb, plan, (xs, Rs) = fuse_subset(ctx, rgb, rgb_cell, PRIMARY_ARMS[:-1], deployed_R)
    res['R'].update(o)
    res['fb'].update(fb)
    res['plan'] = plan
    res['member_R'] = Rs
    c0R = o['C0']
    # (h) self-peak SA-TM
    sp = [ctx.cell_for([m]) for m in rgb]
    o2, fb2, plan_sp, _ = fuse_subset(ctx, rgb, None, ['C0', 'SATM'], c0R, per_member_cells=sp)
    res['R']['h_SATM_selfpeak'], res['fb']['h_SATM_selfpeak'] = o2['SATM'], fb2['SATM']
    res['R']['h0_C0_selfpeak'], res['fb']['h0_C0_selfpeak'] = o2['C0'], fb2['C0']
    res['plan_selfpeak'] = plan_sp
    # single members, self-peak read-out (replicates the existing single-member dumps)
    for n in names:
        m = idx[n]
        c = ctx.cell_for([m])
        R_ = ctx.R(m, c)
        res['R']['single_' + n], res['fb']['single_' + n] = (R_, False) if R_ is not None else (c0R, True)
    # (i) leave-one-out 4-member
    for i in range(5):
        sub = [m for m in rgb if m != rgb[i]]
        cell = ctx.cell_for(sub)
        o3, fb3, _, _ = fuse_subset(ctx, sub, cell, ['C0', 'SATM'], c0R)
        for k in ('C0', 'SATM'):
            res['R']['i_LOO%d_%s' % (i, k)], res['fb']['i_LOO%d_%s' % (i, k)] = o3[k], fb3[k]
    # (j) two-member MM ensembles
    res['mm2_r'] = {}
    for tag, (a, b) in MM_PAIRS:
        if a not in idx or b not in idx:
            continue
        sub = [idx[a], idx[b]]
        cell = ctx.cell_for(sub)
        o4, fb4, _, (_, Rs4) = fuse_subset(ctx, sub, cell, ['C0', 'SATM'], c0R)
        for k in ('C0', 'SATM'):
            res['R']['j_%s_%s' % (tag, k)], res['fb']['j_%s_%s' % (tag, k)] = o4[k], fb4[k]
        res['mm2_r'][tag] = rel_yaw(Rs4[0], Rs4[1]) if Rs4[0] is not None and Rs4[1] is not None else np.nan
    # MM substitution (4 RGB minus i + MM_j) and the 11-member reference
    mm = [idx[n] for n in MM_NAMES if n in idx]
    for i in range(5):
        for jn, j in zip([n for n in MM_NAMES if n in idx], mm):
            sub = [m for m in rgb if m != rgb[i]] + [j]
            cell = ctx.cell_for(sub)
            o5, fb5, _, _ = fuse_subset(ctx, sub, cell, ['C0', 'SATM'], c0R)
            res['R']['sub_%d_%s_SATM' % (i, jn)], res['fb']['sub_%d_%s_SATM' % (i, jn)] = o5['SATM'], fb5['SATM']
    if mm:
        allm = rgb + mm
        cell = ctx.cell_for(allm)
        o6, fb6, _, _ = fuse_subset(ctx, allm, cell, ['C0', 'SATM'], c0R)
        res['R']['MM11_SATM'], res['fb']['MM11_SATM'] = o6['SATM'], fb6['SATM']
        res['R']['MM11_C0'], res['fb']['MM11_C0'] = o6['C0'], fb6['C0']
    return res


# ======================================================================================= loading
def load_members(npz_path):
    D = dict(np.load(npz_path, allow_pickle=False))
    jp = os.path.splitext(npz_path)[0] + '.json'
    info = json.load(open(jp, encoding='utf-8')) if os.path.exists(jp) else {}
    return D, info


def load_dump(path, with_gt=False):
    """{(seq, frame): frame dict}. Without with_gt the ground-truth fields are dropped right after unpickling."""
    D = pickle.load(open(path, 'rb'))
    out = {}
    for f in D['frames']:
        g = {k: f[k] for k in ('seq', 'frame', 'cls', 'pred', 'conf', 'K_in', 'raw_wh')}
        if with_gt:
            g['gt'] = f['gt']
            g['matched'] = f['matched']
        out[(str(f['seq']), str(f['frame']))] = g
    info = D['info']
    del D
    return out, info


def top_index(fr):
    return int(np.argmax(fr['conf'])) if fr['pred'] is not None and len(fr['conf']) else -1


def frame_geo(D, i, lwh, kp_scale):
    return Geo(D['K_in'][i], D['dist'][i], D['raw_size'][i], D['new_size'][i], D['stride_b'][i], D['pos_rc'][i],
               D['center_res'][i], lwh, kp_scale)


def frame_ctx(D, i, lwh, kp_scale):
    return FrameCtx(D['hm_win'][i], D['kp_win'][i], D['z'][i], frame_geo(D, i, lwh, kp_scale))


def rgb_cell_of(D, i, search):
    rb, cb = D['rb_cb'][i]
    rc, cc = D['rc_cc'][i]
    return (int(rb - (rc - search)), int(cb - (cc - search)))


def check_prereg(expect=None):
    h = sha256(PREREG)
    rec = open(PREREG + '.sha256', encoding='utf-8').read().split()[0]
    if h != rec or (expect is not None and h != expect):
        raise SystemExit('prereg.json hash mismatch (%s vs recorded %s / expected %s) -- pre-registration was modified' % (h, rec, expect))
    return h


def paths(a):
    tag = a.tag
    d = a.dir
    return {'npz': os.path.join(d, 'kz_members_%s.npz' % tag), 'dump': os.path.join(d, 'real_ens5flip_kpKZavg_zoom_%s.pkl' % tag),
            'check': os.path.join(d, 'check_%s.json' % tag), 'frozen': os.path.join(d, 'frozen_%s.npz' % tag),
            'frozen_json': os.path.join(d, 'frozen_%s.json' % tag), 'results': os.path.join(d, 'results_%s.json' % tag),
            'report': os.path.join(d, 'RESULTS_%s.md' % tag), 'ads': os.path.join(d, 'ads_%s' % tag)}


# ======================================================================================= stage: check (label-free)
def stage_check(a):
    P = paths(a)
    pre = check_prereg()
    log = []
    D, info = load_members(P['npz'])
    names = list(info.get('member_names', RGB_NAMES + MM_NAMES))
    search = int(info.get('search', 2))
    kp_scale = float(info.get('KP_SCALE') or KP_SCALE_DEFAULT)
    new, new_info = load_dump(P['dump'])
    old, old_info = load_dump(os.path.join(a.old_root, OLD_AVG))
    N = len(D['seq'])
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    say('[check] members npz %d frames, %d members %s; dump %d frames; old avg dump %d frames' % (N, len(names), names, len(new), len(old)), log)
    R = {'prereg_sha256': pre, 'npz_sha256': sha256(P['npz']), 'dump_sha256': sha256(P['dump'])}
    assert len(set(keys)) == N and set(keys) == set(new.keys()), 'recorded frames != dump frames'
    if info.get('dec_kwargs') is not None:
        dk = info['dec_kwargs']
        R['dec_kwargs'] = dk
        if 'use_distortion' in dk and not dk['use_distortion']:
            raise SystemExit('dec_kwargs use_distortion=False: the PnP replica assumes the decoder default True; adapt pnp() first')
        if dk.get('euler_seq') not in (None, SEQ):
            raise SystemExit('euler_seq %r != %r' % (dk.get('euler_seq'), SEQ))
    if info.get('kp2d_dtype') and info['kp2d_dtype'] != ['torch.float32']:
        say('[check] NOTE kp2d dtype %s (the replica casts to float32)' % info['kp2d_dtype'], log)

    # (iii) positions / conf / detections identical to the deployed KZ dump; deployed rotation difference
    pos_bad, rot_d = 0, []
    for k in keys:
        fn, fo = new[k], old.get(k)
        if fo is None or fn['pred'].shape != fo['pred'].shape or not np.array_equal(fn['pred'][:, :6], fo['pred'][:, :6]) \
                or not np.array_equal(fn['conf'], fo['conf']):
            pos_bad += 1
            continue
        t = top_index(fn)
        if t >= 0:
            rot_d.append(geodesic(rot_from_eul(fn['pred'][t, 6:9]), rot_from_eul(fo['pred'][t, 6:9])))
    rot_d = np.asarray(rot_d)
    R['iii_position_mismatch_frames'] = pos_bad
    R['iii_deployed_rot_vs_old_max_deg'] = float(rot_d.max()) if len(rot_d) else None
    R['iii_deployed_rot_vs_old_frac_lt_0p05'] = float((rot_d < 0.05).mean()) if len(rot_d) else None
    say('[check] (iii) position/conf mismatches vs deployed KZ dump: %d / %d; deployed rotation vs old: max %.2e deg, <0.05 deg %.4f' % (
        pos_bad, N, R['iii_deployed_rot_vs_old_max_deg'] or 0, R['iii_deployed_rot_vs_old_frac_lt_0p05'] or 0), log)

    # (i) PnP replica on the recorded deployed vector, offline C0, top-detection consistency; (v) cell re-selection
    d_dep, d_c0, hm_ok, cell_ok, n_top = [], [], 0, 0, 0
    n_incons, n_fail_dep, n_fail_c0 = 0, 0, 0
    single_fail = {n: 0 for n in names}
    idx = {n: i for i, n in enumerate(names)}
    rgb = [idx[n] for n in RGB_NAMES]
    singles = {n: [] for n in names}
    old_single = {}
    for n in names:
        p = os.path.join(a.old_root, OLD_SINGLE[n])
        if os.path.exists(p):
            old_single[n] = load_dump(p)[0]
    pairs = {}
    for tag, (x, y) in MM_PAIRS:
        p = os.path.join(a.old_root, OLD_PAIR[tag])
        if os.path.exists(p) and x in idx and y in idx:
            pairs[tag] = (load_dump(p)[0], [idx[x], idx[y]], [])
    t0 = time.time()
    for i, k in enumerate(keys):
        fn = new[k]
        t = top_index(fn)
        cell = rgb_cell_of(D, i, search)
        if t < 0:
            continue
        n_top += 1
        lwh = fn['pred'][t, 3:6]
        ctx = frame_ctx(D, i, lwh, kp_scale)
        Rdep = rot_from_eul(fn['pred'][t, 6:9])
        cell_ok += ctx.cell_for(rgb) == cell
        if abs(fn['conf'][t] - D['hm_peak'][i]) >= 1e-5:
            n_incons += 1               # top detection is not the cell the hook wrote kp2d into (its rotation is the rot head's)
            continue
        hm_ok += 1
        Rp = pnp(D['kp_deployed'][i], ctx.g)
        if Rp is None:
            n_fail_dep += 1             # decoder PnP failed in deployment too -> rot-head rotation kept; nothing to compare
        else:
            d_dep.append(geodesic(Rp, Rdep))
        Rc = pnp(f32(mean_vec([ctx.x(m, cell) for m in rgb])), ctx.g)
        if Rc is None or Rp is None:
            n_fail_c0 += int(Rc is None)
        else:
            d_c0.append(geodesic(Rc, Rdep))
        for n, od in old_single.items():
            fo = od.get(k)
            to = top_index(fo) if fo is not None else -1
            if to < 0:
                continue
            m = idx[n]
            Rs = ctx.R(m, ctx.cell_for([m]))
            if Rs is None:
                single_fail[n] += 1
            else:
                singles[n].append(geodesic(Rs, rot_from_eul(fo['pred'][to, 6:9])))
        for tag, (od, sub, acc) in pairs.items():
            fo = od.get(k)
            to = top_index(fo) if fo is not None else -1
            if to < 0:
                continue
            c2 = ctx.cell_for(sub)
            R2 = pnp(f32(mean_vec([ctx.x(m, c2) for m in sub])), ctx.g)
            acc.append(geodesic(R2, rot_from_eul(fo['pred'][to, 6:9])) if R2 is not None else np.inf)
        if i % 1000 == 999:
            say('[check]   %d / %d frames, %.0f s' % (i + 1, N, time.time() - t0))
    d_dep, d_c0 = np.asarray(d_dep), np.asarray(d_c0)
    R['n_frames_with_detection'] = n_top
    R['top_conf_equals_hook_peak_frac'] = hm_ok / max(n_top, 1)
    R['n_top_not_hook_cell'] = n_incons
    R['n_deployed_pnp_failed'] = n_fail_dep
    R['n_offline_C0_pnp_failed'] = n_fail_c0
    R['single_pnp_failed'] = single_fail
    R['i_pnp_on_deployed_vector_frac_lt_1e-3'] = float((d_dep < 1e-3).mean())
    R['i_pnp_on_deployed_vector_max_deg'] = float(np.max(d_dep))
    R['i_offline_C0_frac_lt_0p05'] = float((d_c0 < 0.05).mean())
    R['i_offline_C0_median_deg'] = float(np.median(d_c0))
    R['v_cell_reselect_frac'] = cell_ok / max(n_top, 1)
    say('[check] top detections not at the hook cell: %d; PnP failed: deployed %d, offline C0 %d' % (n_incons, n_fail_dep, n_fail_c0), log)
    say('[check] top conf == hook peak: %.4f | (i) PnP(deployed vector) <1e-3 deg: %.4f (max %.2e) | offline C0 <0.05 deg: %.4f (median %.2e) | (v) cell re-select %.4f' % (
        R['top_conf_equals_hook_peak_frac'], R['i_pnp_on_deployed_vector_frac_lt_1e-3'], R['i_pnp_on_deployed_vector_max_deg'],
        R['i_offline_C0_frac_lt_0p05'], R['i_offline_C0_median_deg'], R['v_cell_reselect_frac']), log)
    R['ii_single'] = {}
    for n, v in singles.items():
        if n in old_single and len(v):
            v = np.asarray(v)
            R['ii_single'][n] = {'n': int(len(v)), 'frac_lt_0p05': float((v < 0.05).mean()), 'median_deg': float(np.median(v)),
                                 'pnp_failed': single_fail[n]}
            say('[check] (ii) single %-10s self-peak vs existing dump: <0.05 deg %.4f (n=%d, median %.2e)' % (n, (v < 0.05).mean(), len(v), np.median(v)), log)
        else:
            R['ii_single'][n] = None
            say('[check] (ii) single %-10s: existing dump not found -> skipped' % n, log)
    R['iv_mm2'] = {}
    for tag, (_, _, acc) in pairs.items():
        v = np.asarray(acc)
        R['iv_mm2'][tag] = {'n': int(len(v)), 'frac_lt_0p05': float((v < 0.05).mean()) if len(v) else None}
        say('[check] (iv) 2-member %s C0 vs existing avg dump: <0.05 deg %.4f (n=%d)' % (tag, (v < 0.05).mean() if len(v) else np.nan, len(v)), log)

    fails = []
    if n_incons > 0.005 * max(n_top, 1):
        fails.append('top detection != hook cell on %d frames (>0.5%%)' % n_incons)
    if n_fail_dep > 0.005 * max(n_top, 1) or n_fail_c0 > n_fail_dep + 0.001 * max(n_top, 1):
        fails.append('PnP failures: deployed %d, offline C0 %d' % (n_fail_dep, n_fail_c0))
    if pos_bad:
        fails.append('iii positions/conf differ from the deployed KZ dump on %d frames' % pos_bad)
    if R['i_pnp_on_deployed_vector_frac_lt_1e-3'] < 0.999:
        fails.append('i PnP replica on the deployed vector reproduces only %.4f' % R['i_pnp_on_deployed_vector_frac_lt_1e-3'])
    if R['i_offline_C0_frac_lt_0p05'] < 0.995:
        fails.append('i offline C0 reproduces only %.4f (<0.995)' % R['i_offline_C0_frac_lt_0p05'])
    if R['v_cell_reselect_frac'] < 0.999:
        fails.append('v cell re-selection %.4f (<0.999)' % R['v_cell_reselect_frac'])
    for n, v in R['ii_single'].items():
        if v is not None and v['frac_lt_0p05'] < 0.995:
            fails.append('ii single %s reproduces only %.4f' % (n, v['frac_lt_0p05']))
    R['fails'] = fails
    R['pass'] = not fails
    R['log'] = log
    jdump(R, P['check'])
    say('[check] %s -> %s' % ('PASS' if R['pass'] else 'FAIL: ' + '; '.join(fails), P['check']))
    return R


# ======================================================================================= stage: freeze (label-free)
def stage_freeze(a):
    P = paths(a)
    pre = check_prereg()
    ck = json.load(open(P['check'], encoding='utf-8'))
    if not ck.get('pass') and not a.allow_check_fail:
        raise SystemExit('check did not pass -- stop and fix (prereg replication_checks_stop_if_fail)')
    if ck['npz_sha256'] != sha256(P['npz']) or ck['dump_sha256'] != sha256(P['dump']):
        raise SystemExit('inputs changed since check')
    D, info = load_members(P['npz'])
    names = list(info.get('member_names', RGB_NAMES + MM_NAMES))
    search = int(info.get('search', 2))
    kp_scale = float(info.get('KP_SCALE') or KP_SCALE_DEFAULT)
    new, _ = load_dump(P['dump'])
    N = len(D['seq'])
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    arm_names, Rarr, FB = None, None, None
    strata, strata_sp = np.full(N, 'no_det', dtype='<U12'), np.full(N, 'no_det', dtype='<U12')
    medoid = np.full(N, -1, np.int64)
    keep = np.zeros((N, 5), bool)
    flip = np.zeros((N, 5), bool)
    rmed = np.full((N, 5), np.nan)
    memR = np.full((N, 5, 3, 3), np.nan)
    mm2_r = np.full((N, len(MM_PAIRS)), np.nan)
    top = np.full(N, -1, np.int64)
    n_incons = 0
    cls = np.array([new[k]['cls'] for k in keys])
    t0 = time.time()
    for i, k in enumerate(keys):
        fn = new[k]
        t = top_index(fn)
        if t >= 0 and abs(fn['conf'][t] - D['hm_peak'][i]) >= 1e-5:
            n_incons += 1               # top detection is not the hook cell: leave the frame unchanged in every arm
            t = -1
        top[i] = t
        if t < 0:
            continue
        ctx = frame_ctx(D, i, fn['pred'][t, 3:6], kp_scale)
        res = compute_frame(ctx, names, rot_from_eul(fn['pred'][t, 6:9]), rgb_cell_of(D, i, search))
        if arm_names is None:
            arm_names = sorted(res['R'].keys())
            Rarr = np.full((len(arm_names), N, 3, 3), np.nan)
            FB = np.zeros((len(arm_names), N), bool)
        for j, an in enumerate(arm_names):
            Rarr[j, i] = res['R'][an]
            FB[j, i] = res['fb'][an]
        pl = res['plan']
        if pl is not None:
            strata[i] = pl['stratum']
            medoid[i] = pl['medoid']
            for m in pl['keep']:
                keep[i, m] = True
            flip[i] = pl['flip'][:5]
            rmed[i] = pl['r'][:5]
        else:
            strata[i] = 'degenerate'
        strata_sp[i] = res['plan_selfpeak']['stratum'] if res['plan_selfpeak'] is not None else 'degenerate'
        for m, Rm in enumerate(res['member_R'][:5]):
            if Rm is not None:
                memR[i, m] = Rm
        for j, (tag, _) in enumerate(MM_PAIRS):
            mm2_r[i, j] = res['mm2_r'].get(tag, np.nan)
        if i % 500 == 499:
            say('[freeze]   %d / %d frames, %.0f s' % (i + 1, N, time.time() - t0))
    A = {an: j for j, an in enumerate(arm_names)}
    has = top >= 0

    # structural identities (prereg strata_label_free.structural_identities_asserted)
    def same(x, y, sel):
        return bool(np.array_equal(Rarr[A[x]][sel], Rarr[A[y]][sel]))
    ident = {'agree: SATM==C0': same('SATM', 'C0', strata == 'agree'),
             'flip_only: SATM==e_C1': same('SATM', 'e_C1', strata == 'flip_only'),
             'flip_only: b_TMnoalign==C0': same('b_TMnoalign', 'C0', strata == 'flip_only'),
             'trim_only: SATM==b_TMnoalign': same('SATM', 'b_TMnoalign', strata == 'trim_only')}
    # gate: SA-TM vs C0 heading change > 5 deg or tilt change > 2 deg on >= 5 % of the frames with a detection
    dh = np.array([fold(rel_yaw(Rarr[A['C0']][i], Rarr[A['SATM']][i])) if has[i] else np.nan for i in range(N)])
    dt = np.array([tilt_between(Rarr[A['C0']][i], Rarr[A['SATM']][i]) if has[i] else np.nan for i in range(N)])
    changed = ((dh > 5) | (dt > 2))[has]
    gate_frac = float(changed.mean())
    gate_pass = gate_frac >= 0.05
    S = ['agree', 'flip_only', 'trim_only', 'both', 'degenerate', 'no_det']
    share = {}
    for c in ('phantom4', 'mavic2', 'all'):
        sel = np.ones(N, bool) if c == 'all' else (cls == c)
        share[c] = {s: float((strata[sel] == s).mean()) for s in S}
        share[c + '_selfpeak'] = {s: float((strata_sp[sel] == s).mean()) for s in S}
        share[c + '_trim_ge1'] = float(((keep[sel] == False) & np.isfinite(rmed[sel])).any(1).mean())   # noqa: E712
    fbr = {an: int(FB[A[an]][has].sum()) for an in arm_names if FB[A[an]][has].any()}
    out = {'seq': D['seq'], 'frame': D['frame'], 'cls': cls, 'top': top, 'stratum': strata, 'stratum_selfpeak': strata_sp,
           'medoid': medoid, 'keep': keep, 'flip': flip, 'r_to_medoid': rmed, 'member_R': memR, 'mm2_r': mm2_r,
           'arm_names': np.array(arm_names), 'R': Rarr, 'fallback': FB, 'dh_satm_c0': dh, 'dt_satm_c0': dt}
    with open(P['frozen'] + '.tmp', 'wb') as f:
        np.savez(f, **out)
    os.replace(P['frozen'] + '.tmp', P['frozen'])
    J = {'prereg_sha256': pre, 'frozen_sha256': sha256(P['frozen']), 'npz_sha256': ck['npz_sha256'], 'dump_sha256': ck['dump_sha256'],
         'script_sha256': sha256(os.path.abspath(__file__)), 'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
         'n_frames': N, 'n_with_detection': int(has.sum()), 'n_top_not_hook_cell_left_unchanged': n_incons, 'arms': arm_names,
         'gate': {'frac_changed': gate_frac, 'threshold': 0.05, 'pass': gate_pass,
                  'median_heading_change_deg': float(np.nanmedian(dh)), 'median_tilt_change_deg': float(np.nanmedian(dt))},
         'structural_identities': ident, 'strata_share': share, 'fallback_counts': fbr,
         'unfolded_branch_differs_frac': float((np.array([rel_yaw(Rarr[A['C0']][i], Rarr[A['SATM']][i]) if has[i] else np.nan
                                                          for i in range(N)])[has] > 90).mean())}
    jdump(J, P['frozen_json'])
    say('[freeze] strata (all): %s' % ' '.join('%s %.3f' % (s, share['all'][s]) for s in S))
    say('[freeze] strata self-peak (all): %s' % ' '.join('%s %.3f' % (s, share['all_selfpeak'][s]) for s in S))
    say('[freeze] >=1 member trimmed: all %.3f phantom4 %.3f mavic2 %.3f' % (share['all_trim_ge1'], share['phantom4_trim_ge1'], share['mavic2_trim_ge1']))
    say('[freeze] gate: SA-TM changes heading>5 or tilt>2 on %.3f of detected frames -> %s' % (gate_frac, 'PASS' if gate_pass else 'STOP (null effect)'))
    say('[freeze] structural identities: %s' % ident)
    say('[freeze] fallbacks: %s' % fbr)
    say('[freeze] wrote %s (sha256 %s)' % (P['frozen'], J['frozen_sha256']))
    if not all(ident.values()):
        raise SystemExit('structural identity violated -- implementation bug, do not evaluate')
    return J


# ======================================================================================= stage: evaluate (reads GT)
def frame_errors(Rp, g):
    """summarize_dumps.stats per-frame quantities: heading fold-180 error and tilt (deg)."""
    Rg = rot_from_eul(g[6:9])
    E = Rg.T @ Rp
    a = abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))
    return min(a, 180 - a), float(np.degrees(np.arccos(np.clip(Rg[:, 2] @ Rp[:, 2], -1, 1))))


def build_arm_frames(frames, keys, Rarm, top):
    """Dump-format frame list with the top detection's rotation replaced by the arm's (euler 'xyz' like the decoder)."""
    out = []
    for i, k in enumerate(keys):
        f = frames[k]
        g = dict(f)
        if top[i] >= 0 and np.isfinite(Rarm[i]).all():
            p = f['pred'].copy()
            p[top[i], 6:9] = Rot.from_matrix(Rarm[i]).as_euler(SEQ)
            g['pred'] = p
        out.append(g)
    return out


def per_frame_metrics(F, seq_xyz=SEQ):
    from uavdet3d.utils import pose_eval
    n = len(F)
    he = np.full(n, np.nan)
    ti = np.full(n, np.nan)
    ang = np.full(n, np.inf)
    pos = np.full(n, np.inf)
    add = np.full(n, np.inf)
    diam = np.full(n, np.nan)
    for i, f in enumerate(F):
        if f['matched'] is None:
            continue
        p, g = f['pred'][f['matched']], f['gt']
        Rp = rot_from_eul(p[6:9])
        he[i], ti[i] = frame_errors(Rp, g)
        e = pose_eval._pair_errors(p, g, f['K_in'], seq_xyz)
        ang[i], pos[i], add[i], diam[i] = e['ang'], e['pos'], e['add'], e['diam']
    return {'he': he, 'ti': ti, 'ang': ang, 'pos': pos, 'add': add, 'diam': diam}


def summary_numbers(m, sel=None):
    n = len(m['he']) if sel is None else int(sel.sum())
    s = np.ones(len(m['he']), bool) if sel is None else sel
    hit = np.isfinite(m['he']) & s
    d_med = float(np.nanmedian(m['diam'][np.isfinite(m['diam'])]))       # paper_metrics: over all matched (whole set)
    return {'n': n, 'det': 100.0 * hit.sum() / max(n, 1), 'head': float(np.median(m['he'][hit])) if hit.any() else np.nan,
            'head_lt10': 100.0 * (m['he'][hit] < 10).mean() if hit.any() else np.nan,
            'head_lt20': 100.0 * (m['he'][hit] < 20).mean() if hit.any() else np.nan,
            'tilt': float(np.median(m['ti'][hit])) if hit.any() else np.nan,
            'acc_10deg10cm': 100.0 * ((m['pos'][s] < 0.1) & (m['ang'][s] < 10.0)).sum() / max(n, 1),
            'add_0p5d': 100.0 * (m['add'][s] < 0.5 * d_med).sum() / max(n, 1)}


def ads_indoor(F, info_m, out_dir, tag):
    try:
        os.environ.setdefault('MPLBACKEND', 'Agg')
        for p in (ROOT, TOOLS):
            if p not in sys.path:
                sys.path.insert(0, p)
        import eval_camnorm as EC
        recs = [{'gt': np.asarray(f['gt'], np.float64).reshape(-1, 9), 'pred': f['pred'], 'conf': f['conf'], 'K': f['K_in'],
                 'seq_id': f['seq'], 'frame_id': f['frame']} for f in F]
        raw_whs = [tuple(f['raw_wh']) for f in F]
        rep = EC.ads_eval(recs, raw_whs, float(info_m['W']), float(info_m['H']), SEQ, ['indoor'], out_dir, tag)['indoor']
        m = re.search(r'LAA3D_ADS_\S+ \(%\) : ([0-9.]+)', rep)
        return float(m.group(1)) if m else None
    except Exception as e:      # noqa: BLE001
        print('[evaluate] ADS failed: %r' % (e,), flush=True)
        return None


def boot_draws(seq_ids, B, seed):
    u = sorted(set(seq_ids))
    rng = np.random.default_rng(seed)
    return u, rng.integers(0, len(u), size=(B, len(u)))


def cluster_boot(stat, seq_of, groups, draws):
    """stat(index_array) -> float; seq_of: per-frame seq id array (only frames that enter the statistic)."""
    idx_by = [np.where(seq_of == s)[0] for s in groups]
    out = np.empty(len(draws))
    for b, d in enumerate(draws):
        ii = np.concatenate([idx_by[j] for j in d])
        out[b] = stat(ii)
    return out


def holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    rej = np.zeros(len(pvals), bool)
    for rank, j in enumerate(order):
        if pvals[j] <= alpha / (len(pvals) - rank):
            rej[j] = True
        else:
            break
    return rej


def stage_evaluate(a):
    P = paths(a)
    pre = check_prereg()
    FJ = json.load(open(P['frozen_json'], encoding='utf-8'))
    if FJ['frozen_sha256'] != sha256(P['frozen']) or FJ['prereg_sha256'] != pre:
        raise SystemExit('frozen file / prereg hash mismatch')
    if not FJ['gate']['pass'] and not a.allow_gate_fail:
        raise SystemExit('gate failed at freeze (%.3f < 0.05): pre-registered STOP, ground truth is not read' % FJ['gate']['frac_changed'])
    for p in (ROOT, TOOLS, TOOLS + '/annot_audit'):
        if p not in sys.path:
            sys.path.insert(0, p)
    import summarize_dumps as SD
    Z = dict(np.load(P['frozen'], allow_pickle=False))
    _, info_m = load_members(P['npz'])
    arms = list(Z['arm_names'])
    A = {an: j for j, an in enumerate(arms)}
    keys = [(str(s), str(f)) for s, f in zip(Z['seq'], Z['frame'])]
    top = Z['top']
    frames, _ = load_dump(P['dump'], with_gt=True)                   # ---- ground truth read from here on ----
    old, _ = load_dump(os.path.join(a.old_root, OLD_AVG), with_gt=True)
    cls = Z['cls']
    seqs = np.array([k[0] for k in keys])
    log = []
    R = {'prereg_sha256': pre, 'frozen_sha256': FJ['frozen_sha256'], 'gt_first_read_at': time.strftime('%Y-%m-%d %H:%M:%S')}

    # ---------- C0 reproduction gate (before any SA-TM number is computed)
    F_dep = [frames[k] for k in keys]
    F_old = [old[k] for k in keys]
    F_c0 = build_arm_frames(frames, keys, Z['R'][A['C0']], top)
    s_old, s_dep, s_c0 = SD.stats(F_old), SD.stats(F_dep), SD.stats(F_c0)
    R['C0_reproduction'] = {'old_dump': {'head': s_old['head'], 'tilt': s_old['tilt']}, 'new_dump_deployed': {'head': s_dep['head'], 'tilt': s_dep['tilt']},
                            'offline_C0': {'head': s_c0['head'], 'tilt': s_c0['tilt']}}
    say('[evaluate] C0 reproduction: old dump head %.3f tilt %.3f | new dump %.3f / %.3f | offline C0 %.3f / %.3f' % (
        s_old['head'], s_old['tilt'], s_dep['head'], s_dep['tilt'], s_c0['head'], s_c0['tilt']), log)
    if abs(s_c0['head'] - s_old['head']) > 0.2 or abs(s_c0['tilt'] - s_old['tilt']) > 0.2:
        R['stopped'] = 'C0 reproduction outside +-0.2 deg'
        jdump(R, P['results'])
        raise SystemExit('C0 does not reproduce the deployed numbers within 0.2 deg -- stop and fix')

    # ---------- per-arm per-frame metrics
    M = {}
    for an in arms:
        M[an] = per_frame_metrics(build_arm_frames(frames, keys, Z['R'][A[an]], top))
    M['_deployed'] = per_frame_metrics(F_dep)
    F_s = build_arm_frames(frames, keys, Z['R'][A['SATM']], top)
    s_s = SD.stats(F_s)
    for nm_, s_ref in (('C0', s_c0), ('SATM', s_s)):
        hit = np.isfinite(M[nm_]['he'])
        assert np.median(M[nm_]['he'][hit]) == s_ref['head'] and np.median(M[nm_]['ti'][hit]) == s_ref['tilt'], \
            'per-frame metric replica differs from summarize_dumps.stats for %s' % nm_
    hit = np.isfinite(M['C0']['he'])
    assert all(np.array_equal(np.isfinite(M[an]['he']), hit) for an in arms), 'detection set changed between arms'
    assert all(np.array_equal(M[an]['pos'], M['C0']['pos']) for an in arms), 'positions changed between arms'
    classes = {'all': np.ones(len(keys), bool), 'phantom4': cls == 'phantom4', 'mavic2': cls == 'mavic2'}
    R['table'] = {an: {c: summary_numbers(M[an], sel) for c, sel in classes.items()} for an in arms + ['_deployed']}

    # ---------- primary statistics: H1 heading, H2 tilt (SA-TM vs C0, full 5, common cell)
    groups, draws = boot_draws(seqs[hit], a.boot, a.seed)
    sh = seqs[hit]

    def dmed(x, y, key):
        vx, vy = M[x][key][hit], M[y][key][hit]
        return lambda ii: float(np.median(vx[ii]) - np.median(vy[ii]))
    prim = {}
    for H, key in (('H1_heading', 'he'), ('H2_tilt', 'ti')):
        st = dmed('SATM', 'C0', key)
        full = st(np.arange(hit.sum()))
        bs = cluster_boot(st, sh, groups, draws)
        p1 = (1 + np.sum(bs >= 0)) / (len(bs) + 1)
        per_seq = {s: st(np.where(sh == s)[0]) for s in groups}
        n_cons = int(sum(v <= 0 for v in per_seq.values()))
        prim[H] = {'delta': full, 'p_one_sided': float(p1), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   'per_seq_delta': per_seq, 'sign_consistent': n_cons, 'n_seq': len(groups),
                   'C0': float(np.median(M['C0'][key][hit])), 'SATM': float(np.median(M['SATM'][key][hit]))}
    rej = holm(np.array([prim['H1_heading']['p_one_sided'], prim['H2_tilt']['p_one_sided']]), 0.05)
    for H, r_ in zip(('H1_heading', 'H2_tilt'), rej):
        prim[H]['holm_reject'] = bool(r_)
        prim[H]['holds'] = bool(r_ and prim[H]['sign_consistent'] >= 8)
    # paired per-frame
    Rc, Rs_ = Z['R'][A['C0']], Z['R'][A['SATM']]
    matched_is_top = np.array([frames[k]['matched'] is not None and frames[k]['matched'] == top[i] for i, k in enumerate(keys)])
    gd = np.array([geodesic(Rc[i], Rs_[i]) if matched_is_top[i] else 0.0 for i in range(len(keys))])
    sel = matched_is_top & (gd > 1.0)
    for H, key in (('H1_heading', 'he'), ('H2_tilt', 'ti')):
        es, ec = M['SATM'][key][sel], M['C0'][key][sel]
        prim[H]['paired'] = {'n_frames_differ_gt1deg': int(sel.sum()), 'satm_better_frac': float((es < ec).sum() / max((es != ec).sum(), 1)),
                             'paired_median_diff': float(np.median(es - ec)) if sel.any() else np.nan}
    R['primary'] = prim
    # guardrails
    t_all = R['table']
    g1 = t_all['SATM']['all']['acc_10deg10cm'] - t_all['C0']['all']['acc_10deg10cm']
    g2 = t_all['SATM']['all']['add_0p5d'] - t_all['C0']['all']['add_0p5d']
    ads_c0 = ads_indoor(F_c0, info_m, P['ads'], 'C0') if not a.skip_ads else None
    ads_s = ads_indoor(F_s, info_m, P['ads'], 'SATM') if not a.skip_ads else None
    ads_dep = ads_indoor(F_dep, info_m, P['ads'], 'deployed') if (not a.skip_ads and a.ads_deployed) else None
    G = {'G1_10deg10cm_delta_pp': g1, 'G1_pass': g1 >= -0.5, 'G2_add0p5d_delta_pp': g2, 'G2_pass': g2 >= -0.5,
         'G3_ads_indoor_C0': ads_c0, 'G3_ads_indoor_SATM': ads_s, 'G3_ads_indoor_deployed': ads_dep,
         'G3_pass': (ads_s is not None and ads_c0 is not None and round(ads_s, 2) >= round(ads_c0, 2)),
         'G4_pass': True}          # asserted above (positions / detections identical between arms) and in check (iii)
    R['guardrails'] = G
    h1, h2 = prim['H1_heading'], prim['H2_tilt']
    other_ok = (h1['holds'] and h2['delta'] <= 0.2) or (h2['holds'] and h1['delta'] <= 0.2)
    R['deploy_SATM'] = bool((h1['holds'] or h2['holds']) and other_ok and G['G1_pass'] and G['G2_pass'] and G['G3_pass'] and G['G4_pass'])
    say('[evaluate] H1 heading: C0 %.2f -> SATM %.2f (delta %+.2f, CI [%+.2f, %+.2f], p %.4f, holm %s, sign %d/%d) holds=%s' % (
        h1['C0'], h1['SATM'], h1['delta'], h1['ci95'][0], h1['ci95'][1], h1['p_one_sided'], h1['holm_reject'], h1['sign_consistent'], h1['n_seq'], h1['holds']), log)
    say('[evaluate] H2 tilt   : C0 %.2f -> SATM %.2f (delta %+.2f, CI [%+.2f, %+.2f], p %.4f, holm %s, sign %d/%d) holds=%s' % (
        h2['C0'], h2['SATM'], h2['delta'], h2['ci95'][0], h2['ci95'][1], h2['p_one_sided'], h2['holm_reject'], h2['sign_consistent'], h2['n_seq'], h2['holds']), log)
    say('[evaluate] guardrails: 10deg10cm %+.2f pp, ADD(0.5d) %+.2f pp, ADS indoor C0 %s SATM %s -> deploy SA-TM: %s' % (
        g1, g2, ads_c0, ads_s, R['deploy_SATM']), log)

    # ---------- mechanism by stratum (report only)
    strat = Z['stratum']
    mech = {}
    for s in ('agree', 'flip_only', 'trim_only', 'both', 'degenerate'):
        ss = (strat == s) & hit
        if not ss.any():
            continue
        e = {}
        for key in ('he', 'ti'):
            arr = {an: M[an][key][ss] for an in ('C0', 'SATM', 'e_C1', 'b_TMnoalign', 'c_coordmedian', 'd_geoL1', 'a_medoid')}
            seq_s = seqs[ss]
            gr = sorted(set(seq_s))
            st = (lambda vx, vy: (lambda ii: float(np.median(vx[ii]) - np.median(vy[ii]))))(arr['SATM'], arr['C0'])
            dr = np.random.default_rng(a.seed).integers(0, len(gr), size=(a.boot, len(gr)))
            bs = cluster_boot(st, seq_s, gr, dr)
            e[key] = {'n': int(ss.sum()), 'median': {an: float(np.median(v)) for an, v in arr.items()},
                      'delta_SATM_C0': st(np.arange(ss.sum())), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      'by_class': {c: {'n': int((ss & classes[c]).sum()),
                                       'delta': float(np.median(M['SATM'][key][ss & classes[c]]) - np.median(M['C0'][key][ss & classes[c]])) if (ss & classes[c]).any() else None}
                                   for c in ('phantom4', 'mavic2')}}
        mech[s] = e
    R['mechanism'] = mech
    fo = mech.get('flip_only')
    if fo:
        hd = fo['he']
        cond = hd['delta_SATM_C0'] < 0 and hd['ci95'][1] < 0
        better_c = hd['median']['c_coordmedian'] > hd['median']['SATM'] + 0.2
        better_d = hd['median']['d_geoL1'] > hd['median']['SATM'] + 0.2
        R['attribution'] = {'flip_only_SATM_beats_C0': bool(cond), 'beats_c_by_gt0p2': bool(better_c), 'beats_d_by_gt0p2': bool(better_d),
                            'verdict': ('symmetry alignment credited' if cond and better_c and better_d else
                                        ('robust fusion suffices' if cond else 'no alignment gain in flip_only')),
                            'flip_only_tilt_delta': fo['ti']['delta_SATM_C0'],
                            'signed_weight_mechanism_rejected (tilt worse in flip_only)': bool(fo['ti']['delta_SATM_C0'] > 0)}
    # ---------- LOO-4, MM2, MM substitution (report only)
    loo = {}
    for i in range(5):
        c0, s_ = 'i_LOO%d_C0' % i, 'i_LOO%d_SATM' % i
        loo[i] = {k: float(np.median(M[s_][k][hit]) - np.median(M[c0][k][hit])) for k in ('he', 'ti')}
    R['LOO4'] = {'per_i': loo, 'mean_delta_heading': float(np.mean([v['he'] for v in loo.values()])),
                 'mean_delta_tilt': float(np.mean([v['ti'] for v in loo.values()])), 'full5_delta_heading': prim['H1_heading']['delta'],
                 'note': '5 个留一子集两两共享 3 个成员，不独立；M=4 时 C0 可 2:2 退化，偏向本方法'}
    mm2 = {}
    for j, (tag, _) in enumerate(MM_PAIRS):
        c0, s_ = 'j_%s_C0' % tag, 'j_%s_SATM' % tag
        if c0 not in A:
            continue
        r_ = Z['mm2_r'][:, j]
        big = hit & (r_ > 135)
        mm2[tag] = {'head_C0': float(np.median(M[c0]['he'][hit])), 'head_SATM': float(np.median(M[s_]['he'][hit])),
                    'tilt_C0': float(np.median(M[c0]['ti'][hit])), 'tilt_SATM': float(np.median(M[s_]['ti'][hit])),
                    'frac_r_gt135': float(big.sum() / max(hit.sum(), 1)),
                    'r_gt135_head_C0': float(np.median(M[c0]['he'][big])) if big.any() else None,
                    'r_gt135_head_SATM': float(np.median(M[s_]['he'][big])) if big.any() else None}
    R['MM2'] = mm2
    subs = [an for an in arms if an.startswith('sub_')]
    if subs:
        base_h = M['SATM']['he'][hit]
        mmn = sorted({an.split('_', 2)[2].rsplit('_SATM', 1)[0] for an in subs})

        def mm_stat_factory(key):
            base = M['SATM'][key][hit]
            arrs = {(i, n): M['sub_%d_%s_SATM' % (i, n)][key][hit] for i in range(5) for n in mmn}

            def per_i(ii):
                b = np.median(base[ii])
                return [float(np.mean([np.median(arrs[(i, n)][ii]) - b for n in mmn])) for i in range(5)]
            return per_i
        out_mm = {}
        for key in ('he', 'ti'):
            per_i = mm_stat_factory(key)
            full = per_i(np.arange(hit.sum()))
            bs = cluster_boot(lambda ii: float(np.mean(per_i(ii))), sh, groups, draws[:a.boot_mm])
            out_mm[key] = {'delta_i': full, 'mean': float(np.mean(full)), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                           'n_negative': int(sum(v < 0 for v in full))}
        hm_ = out_mm['he']
        out_mm['claim_MM_value'] = bool(hm_['mean'] < 0 and hm_['ci95'][1] < 0 and hm_['n_negative'] >= 4)
        out_mm['MM11_SATM_head'] = float(np.median(M['MM11_SATM']['he'][hit])) if 'MM11_SATM' in A else None
        out_mm['MM11_C0_head'] = float(np.median(M['MM11_C0']['he'][hit])) if 'MM11_C0' in A else None
        out_mm['full5_SATM_head'] = float(np.median(base_h))
        out_mm['bootstrap_B'] = int(min(a.boot_mm, len(draws)))
        R['MM_substitution'] = out_mm
    # mavic2 per-sequence point values
    R['mavic2_per_seq'] = {s: {'C0_head': float(np.median(M['C0']['he'][hit & (seqs == s)])), 'SATM_head': float(np.median(M['SATM']['he'][hit & (seqs == s)])),
                               'C0_tilt': float(np.median(M['C0']['ti'][hit & (seqs == s)])), 'SATM_tilt': float(np.median(M['SATM']['ti'][hit & (seqs == s)]))}
                           for s in sorted(set(seqs[cls == 'mavic2'])) if (hit & (seqs == s)).any()}
    R['frozen_info'] = {k: FJ[k] for k in ('gate', 'strata_share', 'structural_identities', 'fallback_counts')}
    R['log'] = log
    jdump(R, P['results'])
    write_report(R, arms, P['report'])
    say('[evaluate] wrote %s and %s' % (P['results'], P['report']))
    return R


def write_report(R, arms, path):
    L = ['# E1 SA-TM 结果（%s）' % R['gt_first_read_at'], '',
         'prereg sha256 `%s`；frozen sha256 `%s`' % (R['prereg_sha256'], R['frozen_sha256']), '']
    p = R['primary']
    L += ['## 主判据（全 5 RGB，公共读出格，SA-TM 对 C0）', '',
          '| 假设 | C0 | SA-TM | Δ | 95% 区间（序列聚类 bootstrap） | 单侧 p | Holm | 序列一致 | 成立 | SA-TM 更近比例（>1° 帧） |', '|---|---|---|---|---|---|---|---|---|---|']
    for H in ('H1_heading', 'H2_tilt'):
        h = p[H]
        L.append('| %s | %.2f° | %.2f° | %+.2f° | [%+.2f, %+.2f] | %.4f | %s | %d/%d | %s | %.3f（n=%d） |' % (
            H, h['C0'], h['SATM'], h['delta'], h['ci95'][0], h['ci95'][1], h['p_one_sided'], '拒绝' if h['holm_reject'] else '—',
            h['sign_consistent'], h['n_seq'], '是' if h['holds'] else '否', h['paired']['satm_better_frac'], h['paired']['n_frames_differ_gt1deg']))
    G = R['guardrails']
    L += ['', '护栏：10°10cm Δ %+.2f pp（%s）；ADD(0.5d) Δ %+.2f pp（%s）；ADS indoor C0 %s → SA-TM %s（%s）；位置/检出逐帧相同（断言）' % (
        G['G1_10deg10cm_delta_pp'], '过' if G['G1_pass'] else '不过', G['G2_add0p5d_delta_pp'], '过' if G['G2_pass'] else '不过',
        G['G3_ads_indoor_C0'], G['G3_ads_indoor_SATM'], '过' if G['G3_pass'] else '不过'), '',
        '**部署决定：%s**' % ('采用 SA-TM' if R['deploy_SATM'] else '不采用（保持逐元素平均）'), '',
        '## 各臂（只报告；单成员 = 自峰值读出）', '',
        '| 臂 | 航向折180 全部 | phantom4 | mavic2 | <10° | <20° | 倾斜 | 10°10cm | ADD(0.5d) |', '|---|---|---|---|---|---|---|---|---|']
    for an in ['_deployed'] + [x for x in arms if not x.startswith('sub_')]:
        t = R['table'][an]
        L.append('| %s | %.2f° | %.2f° | %.2f° | %.1f%% | %.1f%% | %.2f° | %.2f%% | %.2f%% |' % (
            an, t['all']['head'], t['phantom4']['head'], t['mavic2']['head'], t['all']['head_lt10'], t['all']['head_lt20'],
            t['all']['tilt'], t['all']['acc_10deg10cm'], t['all']['add_0p5d']))
    L += ['', '## 分层机理（无标签分层，冻结于看真值之前）', '', '| 层 | n | Δ航向 [95%] | Δ倾斜 [95%] | C1 | TMnoalign | (c) 中位数 | (d) 测地L1 |', '|---|---|---|---|---|---|---|---|']
    for s, e in R['mechanism'].items():
        L.append('| %s | %d | %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] | %.2f | %.2f | %.2f | %.2f |' % (
            s, e['he']['n'], e['he']['delta_SATM_C0'], e['he']['ci95'][0], e['he']['ci95'][1], e['ti']['delta_SATM_C0'], e['ti']['ci95'][0], e['ti']['ci95'][1],
            e['he']['median']['e_C1'], e['he']['median']['b_TMnoalign'], e['he']['median']['c_coordmedian'], e['he']['median']['d_geoL1']))
    if 'attribution' in R:
        L += ['', '归因：%s' % json.dumps(R['attribution'], ensure_ascii=False)]
    L += ['', '留一 4 成员：%s' % json.dumps(R['LOO4'], ensure_ascii=False), '', '两成员 MM：%s' % json.dumps(R['MM2'], ensure_ascii=False)]
    if 'MM_substitution' in R:
        L += ['', '多模态等规模替换：%s' % json.dumps(R['MM_substitution'], ensure_ascii=False)]
    L += ['', 'mavic2 逐序列：%s' % json.dumps(R['mavic2_per_seq'], ensure_ascii=False), '',
          '冻结信息：%s' % json.dumps(R['frozen_info'], ensure_ascii=False)]
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


# ======================================================================================= stage: synth (signed-weight demo)
def synth_geometry(n, rng):
    """Synthetic but realistic frames: K of the real 512x288 input, level-ish drone 2-7 m in front of a slightly pitched camera."""
    K = np.array([[492.1, 0, 260.1], [0, 492.0, 142.0], [0, 0, 1.0]])
    Rcw = np.array([[1, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])            # world (x east, y north, z up) -> camera looking north
    out = []
    for _ in range(n):
        pitch = Rot.from_euler('x', rng.uniform(-25, -5), degrees=True).as_matrix()
        Rb = pitch @ Rcw @ Rot.from_euler('z', rng.uniform(-180, 180), degrees=True).as_matrix() @ \
            Rot.from_euler('xy', rng.normal(0, 4, 2), degrees=True).as_matrix()
        t = np.array([rng.uniform(-1.2, 1.2), rng.uniform(-0.6, 0.6), rng.uniform(2.0, 7.0)])
        out.append((K, Rb, t))
    return out


def synth_member_vec(K, Rm, t, g, lwh):
    obj = PROTO8 * lwh
    uv, _ = cv2.projectPoints((obj @ Rm.T + t).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, g.dist)
    uv = uv.reshape(8, 2)
    return np.stack([(uv[:, 0] - g.uv[0]) / g.sx, (uv[:, 1] - g.uv[1]) / g.sy], 1).reshape(16) / g.kp_scale


def real_geometry(a):
    """Label-free real geometry (like scratchpad/author_kz_satm_proxy.py): K_in and the deployed position t of the top
    detection from the KZavg dump, base rotation = KZ s0's own rotation; every 6th frame. Never reads gt / matched."""
    avg, _ = load_dump(os.path.join(a.old_root, OLD_AVG))
    s0, _ = load_dump(os.path.join(a.old_root, OLD_SINGLE['KZs0']))
    out = []
    for k in sorted(avg)[::6]:
        fa, fs = avg[k], s0.get(k)
        ta, ts = top_index(fa), (top_index(fs) if fs is not None else -1)
        if ta < 0 or ts < 0 or fa['pred'][ta, 2] <= 0.1:
            continue
        out.append((np.asarray(fa['K_in'], np.float64), rot_from_eul(fs['pred'][ts, 6:9]), fa['pred'][ta, :3].copy()))
    return out


def stage_synth(a):
    rng = np.random.default_rng(a.seed)
    lwh = np.array([0.3489, 0.3487, 0.1996])
    geos = real_geometry(a) if a.geom_from_dump else synth_geometry(a.n_synth, rng)
    a.n_synth = len(geos)
    res = {}
    for sig in (0.0, 10.0):
        for k in (0, 1, 2):
            e0, e1, s0, s1, t0_, t1_ = [], [], [], [], [], []
            for K, Rb, t in geos:
                q = K @ t
                u, v = q[0] / q[2], q[1] / q[2]
                g = Geo(K, np.zeros(5), (512, 288), (512, 288), 4.0, (int(v // 4), int(u // 4)), (u / 4 - int(u // 4), v / 4 - int(v // 4)), lwh)
                hs = rng.normal(0, sig, 5) if sig > 0 else np.zeros(5)
                xs = [synth_member_vec(K, Rb @ Rot.from_euler('z', h, degrees=True).as_matrix(), t, g, lwh) for h in hs]
                X0 = [relabel(xs[m], P180) if m < k else xs[m] for m in range(5)]     # k members output the Pi180 labelling
                R0 = pnp(f32(mean_vec(X0)), g)
                R1 = pnp(f32(mean_vec(xs)), g)                                        # aligned (= C1 / SA-TM when all kept)
                if R0 is None or R1 is None:
                    continue
                d0, d1 = signed_yaw(Rb, R0), signed_yaw(Rb, R1)
                d0 = (d0 + 90) % 180 - 90
                d1 = (d1 + 90) % 180 - 90
                e0.append(abs(d0))
                e1.append(abs(d1))
                s0.append(d0)
                s1.append(d1)
                t0_.append(tilt_between(Rb, R0))
                t1_.append(tilt_between(Rb, R1))
            e0, e1 = np.array(e0), np.array(e1)
            res[(sig, k)] = {'n': len(e0), 'C0_med': float(np.median(e0)), 'C0_p90': float(np.percentile(e0, 90)),
                             'aligned_med': float(np.median(e1)), 'aligned_p90': float(np.percentile(e1, 90)),
                             'ratio_median': float(np.median(e0) / max(np.median(e1), 1e-12)),
                             'ratio_rms': float(np.sqrt(np.mean(np.square(s0))) / max(np.sqrt(np.mean(np.square(s1))), 1e-12)),
                             'tilt_C0_med': float(np.median(t0_)), 'tilt_aligned_med': float(np.median(t1_))}
    pred = {1: np.sqrt(5 * 5 / 9.0), 2: np.sqrt(5 * 5.0)}
    print('signed-weight demo: %d frames (%s geometry), 5 members, heading dispersion sigma, k exact Pi180 relabels; error vs base pose' % (
        a.n_synth, 'real label-free' if a.geom_from_dump else 'synthetic'))
    for (sig, k), r in res.items():
        ratio = ('ratio median %5.2f rms %5.2f' % (r['ratio_median'], r['ratio_rms'])) if sig > 0 else 'ratio n/a (aligned error ~0)'
        print('  sigma=%4.1f k=%d: heading |err| C0 med %6.2f p90 %6.2f | aligned med %6.2f p90 %6.2f | %s%s | tilt C0 %5.2f aligned %5.2f' % (
            sig, k, r['C0_med'], r['C0_p90'], r['aligned_med'], r['aligned_p90'], ratio,
            ('  (predicted sqrt(M*sum w^2) = %.2f)' % pred[k]) if (sig > 0 and k in pred) else '', r['tilt_C0_med'], r['tilt_aligned_med']))
    jdump({'%s_k%d' % (s, k): v for (s, k), v in res.items()},
          os.path.join(a.dir, 'synth_signed_weight%s.json' % ('_realgeom' if a.geom_from_dump else '')))
    return res


# ======================================================================================= stage: selftest
def selftest(a):
    rng = np.random.default_rng(0)
    ok = []
    # 1) Rz(90k) relabel permutations: member R_ref Rz(90k) on a square base maps back onto the reference labelling
    lwh_sq = np.array([0.35, 0.35, 0.2])
    K = np.array([[492.1, 0, 260.1], [0, 492.0, 142.0], [0, 0, 1.0]])
    for (Kk, Rb, t) in synth_geometry(20, rng):
        q = K @ t
        u, v = q[0] / q[2], q[1] / q[2]
        g = Geo(K, np.zeros(5), (512, 288), (512, 288), 4.0, (int(v // 4), int(u // 4)), (u / 4 - int(u // 4), v / 4 - int(v // 4)), lwh_sq)
        xr = synth_member_vec(K, Rb, t, g, lwh_sq)
        for k in range(4):
            Rn = Rb @ Rot.from_euler('z', 90 * k + rng.normal(0, 0.0), degrees=True).as_matrix()
            xn = synth_member_vec(K, Rn, t, g, lwh_sq)
            kk = int(np.round(signed_yaw(Rb, Rn) / 90.0)) % 4
            assert kk == k, (kk, k)
            assert np.allclose(relabel(xn, INV90[kk]), xr, atol=1e-9), 'Rz(90k) relabel wrong'
        # PnP replica recovers the pose exactly from noiseless corners
        Rr = pnp(f32(xr), g)
        assert Rr is not None and geodesic(Rr, Rb) < 1e-2, geodesic(Rr, Rb)
        # Pi180 relabel = same box pose rotated by Rz(pi) under PnP
        Rf = pnp(f32(relabel(xr, P180)), g)
        assert geodesic(Rf, Rb @ np.diag([-1.0, -1.0, 1.0])) < 1e-2
    ok.append('relabel permutations (Rz 0/90/180/270), PnP exact recovery, Pi180 <-> Rz(pi): OK')
    # 2) Weiszfeld: 1-D check inside the yaw subgroup -- L1 mean of yaws {-1, 0, 0.2, 1, 120} is the median 0.2 deg;
    #    the chordal L2 mean is dragged by the 120 deg outlier
    Rb = Rot.random(random_state=1).as_matrix()
    Rs = [Rb @ Rot.from_euler('z', d, degrees=True).as_matrix() for d in (-1.0, 0.0, 0.2, 1.0, 120.0)]
    R02 = Rb @ Rot.from_euler('z', 0.2, degrees=True).as_matrix()
    assert geodesic(geo_l1(Rs), R02) < 1e-3 and geodesic(chordal_l2(Rs), Rb) > 5.0
    ok.append('geodesic L1 (Weiszfeld) = median in the yaw subgroup, robust to one outlier: OK (L1 off by %.1e deg, chordal %.2f deg)' % (
        geodesic(geo_l1(Rs), R02), geodesic(chordal_l2(Rs), Rb)))
    # 3) plan_satm strata and fused structural identities on synthetic members
    lwh = np.array([0.3489, 0.3487, 0.1996])
    counts = {}
    for (Kk, Rb, t) in synth_geometry(200, rng):
        q = Kk @ t
        u, v = q[0] / q[2], q[1] / q[2]
        g = Geo(Kk, np.zeros(5), (512, 288), (512, 288), 4.0, (int(v // 4), int(u // 4)), (u / 4 - int(u // 4), v / 4 - int(v // 4)), lwh)
        offs = [rng.normal(0, 6)] + [rng.choice([rng.normal(0, 6), 180 + rng.normal(0, 6), rng.uniform(50, 130)], p=None) for _ in range(4)]
        xs = [synth_member_vec(Kk, Rb @ Rot.from_euler('z', o, degrees=True).as_matrix(), t, g, lwh) + rng.normal(0, 0.003, 16) for o in offs]
        hv = list(rng.uniform(0.2, 0.9, 5))
        Rs_ = [pnp(f32(x), g) for x in xs]
        pl = plan_satm(Rs_, hv)
        counts[pl['stratum']] = counts.get(pl['stratum'], 0) + 1
        c0 = pnp(f32(mean_vec(xs)), g)
        sat = pnp(f32(satm_vec(xs, pl)), g)
        if pl['stratum'] == 'agree':
            assert np.array_equal(c0, sat)
        if pl['stratum'] == 'flip_only':
            assert np.array_equal(sat, pnp(f32(c1_vec(xs, pl)), g))
            assert np.array_equal(c0, pnp(f32(mean_vec([xs[k] for k in pl['keep']])), g))
        if pl['stratum'] == 'trim_only':
            assert np.array_equal(sat, pnp(f32(mean_vec([xs[k] for k in pl['keep']])), g))
        assert pl['medoid'] in pl['keep']
    ok.append('SA-TM plan + structural identities on 200 synthetic frames: OK %s' % counts)
    # 4) medoid tie-break with M=2 (always a tie): higher hm wins
    Rb = Rot.random(random_state=3).as_matrix()
    pl = plan_satm([Rb, Rb @ Rot.from_euler('z', 170, degrees=True).as_matrix()], [0.3, 0.6])
    assert pl['medoid'] == 1 and pl['keep'] == [0, 1] and pl['flip'][0] and not pl['flip'][1]
    pl = plan_satm([Rb, Rb @ Rot.from_euler('z', 100, degrees=True).as_matrix()], [0.7, 0.6])
    assert pl['medoid'] == 0 and pl['keep'] == [0] and pl['stratum'] == 'trim_only'
    ok.append('M=2 medoid tie-break / trim: OK')
    # 5) Holm and bootstrap p sanity
    assert list(holm(np.array([0.01, 0.04]))) == [True, True] and list(holm(np.array([0.03, 0.04]))) == [False, False]
    assert list(holm(np.array([0.02, 0.2]))) == [True, False]
    ok.append('Holm: OK')
    for s in ok:
        print('[selftest] ' + s)
    return ok


def decoder_crosscheck(a):
    """PnP replica vs the real object_encoder_mav6d.center_point_decoder (CPU), incl. raw != new resolution and distortion."""
    import torch
    for p in (ROOT, TOOLS):
        if p not in sys.path:
            sys.path.insert(0, p)
    from uavdet3d.utils import object_encoder_mav6d as OE
    rng = np.random.default_rng(7)
    lwh = np.array([0.3489, 0.3487, 0.1996], np.float32)
    worst = 0.0
    n = 0
    for trial in range(60):
        raw_w, raw_h = [(512, 288), (1024, 576), (1920, 1080)][trial % 3]
        new_w, new_h, stride = 512, 288, 4
        Kraw = np.array([[492.1 * raw_w / 512, 0, 260.1 * raw_w / 512], [0, 492.0 * raw_h / 288, 142.0 * raw_h / 288], [0, 0, 1.0]])
        dist = np.array([[0.0, 0, 0, 0, 0], [-0.05, 0.02, 0.001, -0.001, 0.0], [-0.23, 0.08, 0, 0, 0]][trial % 3])
        Hh, Ww = new_h // stride, new_w // stride
        r, c = int(rng.integers(5, Hh - 5)), int(rng.integers(5, Ww - 5))
        res = rng.uniform(0, 1, 2).astype(np.float32)
        kp = (rng.normal(0, 0.25, 16)).astype(np.float32)
        # make a plausible box: corners of a real pose, encoded relative to the decoded center
        g = Geo(Kraw, dist, (raw_w, raw_h), (new_w, new_h), stride, (r, c), res, lwh)
        ray = cv2.undistortPoints(np.array([[g.uv]], np.float64), Kraw, dist.reshape(1, -1)).reshape(2)
        t = np.array([ray[0], ray[1], 1.0]) * rng.uniform(2, 6)
        Rb = Rot.from_euler('xyz', rng.uniform(-180, 180, 3), degrees=True).as_matrix()
        kp = (synth_member_vec(Kraw, Rb, t, g, lwh.astype(np.float64)) + rng.normal(0, 0.01, 16)).astype(np.float32)
        hm = torch.full((1, 1, Hh, Ww), 0.01)
        hm[0, 0, r, c] = 0.9
        cres = torch.zeros((1, 2, Hh, Ww))
        cres[0, :, r, c] = torch.from_numpy(res)
        cdis = torch.full((1, 1, Hh, Ww), 3.0)
        dim = torch.zeros((1, 3, Hh, Ww))
        dim[0, :, r, c] = torch.from_numpy(lwh)
        rot = torch.zeros((1, 6, Hh, Ww))
        rot[0, 0, :, :] = 1.0
        rot[0, 2, :, :] = 1.0
        rot[0, 4, :, :] = 1.0
        k2 = torch.zeros((1, 16, Hh, Ww))
        k2[0, :, r, c] = torch.from_numpy(kp)
        boxes, conf = OE.center_point_decoder(hm, cres, cdis, dim, rot, Kraw.reshape(1, 3, 3), np.eye(4).reshape(1, 4, 4),
                                              dist.reshape(1, -1), new_w, new_h, raw_w, raw_h, stride, 1, 10,
                                              euler_seq=SEQ, kp2d=k2, kp_scale=KP_SCALE_DEFAULT, kp_pose_parts='rot')
        Rdec = rot_from_eul(boxes[0, 6:9])
        Rrep = pnp(kp, Geo(Kraw, dist, (raw_w, raw_h), (new_w, new_h), stride, (r, c), res, boxes[0, 3:6]))
        if Rrep is None:
            continue
        worst = max(worst, geodesic(Rdec, Rrep))
        n += 1
    print('[decoder] PnP replica vs center_point_decoder: %d cases (raw 512/1024/1920, k1 0/-0.05/-0.23), max diff %.3e deg' % (n, worst))
    assert n >= 50 and worst < 1e-6, worst          # identical up to the euler round trip
    return worst


# ======================================================================================= synthetic end-to-end fixtures
def make_fixture(root, N=400, seed=11, null=False):
    """Fabricated members npz/json + new dump + old dumps with the real schema, for an end-to-end CPU run."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, 'kz'), exist_ok=True)
    os.makedirs(os.path.join(root, 'kd'), exist_ok=True)
    names = RGB_NAMES + MM_NAMES
    M = len(names)
    S = 5
    lwh = np.array([0.3489, 0.3487, 0.1996])
    seqs_ = ['phantom4/s/%02d' % i for i in range(7)] + ['mavic2/s/%02d' % i for i in range(3)]
    dist = np.array([-0.02, 0.01, 0, 0, 0])
    rows, frames = [], []
    old_single = {n: [] for n in names}
    old_pair = {t: [] for t, _ in MM_PAIRS}
    geos = synth_geometry(N, rng)
    for i, (K, Rg, t) in enumerate(geos):
        seq = seqs_[i % 10]
        frame = '%d.jpg' % (i // 10 + 1)
        q = K @ t
        u, v = q[0] / q[2] + rng.normal(0, 1), q[1] / q[2] + rng.normal(0, 1)
        rpos, cpos = int(v // 4), int(u // 4)
        res = np.array([u / 4 - cpos, v / 4 - rpos], np.float32).astype(np.float64)
        g = Geo(K, dist, (512, 288), (512, 288), 4.0, (rpos, cpos), res, lwh)
        z = rng.uniform(1.0, 2.4)
        rc, cc = (1 if i % 37 == 0 else 36), (126 if i % 41 == 0 else 64)
        hm_win = np.full((M, S, S), np.nan, np.float32)
        kp_win = np.full((M, 16, S, S), np.nan, np.float32)
        r0, r1, c0, c1 = max(0, rc - 2), min(72, rc + 3), max(0, cc - 2), min(128, cc + 3)
        ar, ac = r0 - (rc - 2), c0 - (cc - 2)
        for m in range(M):
            mode = 0.0 if null else rng.random()
            off = rng.normal(0, 2) if null else None
            off = off if null else rng.normal(0, 8) if mode < 0.55 else (180 + rng.normal(0, 8) if mode < 0.7 else (rng.uniform(45, 135) * rng.choice([-1, 1]) if mode < 0.9 else rng.uniform(-180, 180)))
            Rm = Rg @ Rot.from_euler('z', off, degrees=True).as_matrix() @ Rot.from_euler('xy', rng.normal(0, 3, 2), degrees=True).as_matrix()
            x = synth_member_vec(K, Rm, t, g, lwh)
            hm_m = rng.uniform(0.05, 0.3, (S, S))
            hm_m[2, 2] = rng.uniform(0.4, 0.9)
            hm_m[rng.integers(1, 4), rng.integers(1, 4)] += rng.uniform(0, 0.4)
            kp_m = (x[:, None, None] + rng.normal(0, 0.004, (16, S, S))) * z
            hm_win[m, ar:ar + (r1 - r0), ac:ac + (c1 - c0)] = hm_m[ar:ar + (r1 - r0), ac:ac + (c1 - c0)]
            kp_win[m, :, ar:ar + (r1 - r0), ac:ac + (c1 - c0)] = kp_m[:, ar:ar + (r1 - r0), ac:ac + (c1 - c0)]
        mh = np.mean(hm_win[:5], 0, dtype=np.float32)
        j = int(np.argmax(np.where(np.isfinite(mh), mh, -np.inf).reshape(-1)))
        rb, cb = rc - 2 + j // S, cc - 2 + j % S
        kdep = (np.mean(kp_win[:5, :, j // S, j % S], 0, dtype=np.float32) / np.float32(z)).astype(np.float32)
        rows.append({'seq': seq, 'frame': frame, 'pos_rc': (rpos, cpos), 'center_res': tuple(res), 'hm_peak': 0.8, 'u': u, 'v': v,
                     'z': z, 'rc_cc': (rc, cc), 'rb_cb': (rb, cb), 'K_in': K, 'dist': dist, 'raw_size': np.array([512., 288.]),
                     'new_size': np.array([512., 288.]), 'stride_b': 4.0, 'hm_win': hm_win, 'kp_win': kp_win, 'kp_deployed': kdep})
        no_det = (i % 53 == 0)
        Rdep = pnp(kdep, g)
        p_t = t + rng.normal(0, 0.15, 3) * (8.0 if i % 29 == 0 else 1.0)
        pred = np.zeros((0, 9)) if no_det else np.array([np.r_[p_t, lwh, Rot.from_matrix(Rdep).as_euler(SEQ)]])
        if not no_det and i % 7 == 0:          # a second, weaker detection
            pred = np.vstack([pred, np.r_[p_t + 2.0, lwh, 0.1, 0.2, 0.3]])
        conf = np.zeros(0) if no_det else (np.array([0.8]) if len(pred) == 1 else np.array([0.8, 0.3]))
        gt = np.r_[t, lwh * 1.01, Rot.from_matrix(Rg).as_euler(SEQ)]
        matched = None if (no_det or np.linalg.norm(p_t - t) >= 1.0) else 0
        frames.append({'seq': seq, 'frame': frame, 'cls': seq.split('/')[0], 'cache_idx': i, 'gt': gt, 'pred': pred, 'conf': conf,
                       'matched': matched, 'K_in': K, 'K_raw': K * 3.75, 'raw_wh': (1920, 1080)})
        ctx = FrameCtx(hm_win, kp_win, z, Geo(K, dist, (512, 288), (512, 288), 4.0, (rpos, cpos), res, lwh))
        for m, n in enumerate(names):
            Rs_ = ctx.R(m, ctx.cell_for([m]))
            fr = dict(frames[-1])
            if len(pred):
                p2 = pred.copy()
                p2[0, 6:9] = Rot.from_matrix(Rs_ if Rs_ is not None else Rdep).as_euler(SEQ)
                fr['pred'] = p2
            old_single[n].append(fr)
        for tag, (x_, y_) in MM_PAIRS:
            sub = [names.index(x_), names.index(y_)]
            cl = ctx.cell_for(sub)
            R2 = pnp(f32(mean_vec([ctx.x(m, cl) for m in sub])), ctx.g)
            fr = dict(frames[-1])
            if len(pred):
                p2 = pred.copy()
                p2[0, 6:9] = Rot.from_matrix(R2 if R2 is not None else Rdep).as_euler(SEQ)
                fr['pred'] = p2
            old_pair[tag].append(fr)
    A = {'seq': np.array([r['seq'] for r in rows]), 'frame': np.array([r['frame'] for r in rows]),
         'pos_rc': np.array([r['pos_rc'] for r in rows], np.int64), 'center_res': np.array([r['center_res'] for r in rows]),
         'hm_peak': np.array([r['hm_peak'] for r in rows]), 'u': np.array([r['u'] for r in rows]), 'v': np.array([r['v'] for r in rows]),
         'z': np.array([r['z'] for r in rows]), 'rc_cc': np.array([r['rc_cc'] for r in rows], np.int64),
         'rb_cb': np.array([r['rb_cb'] for r in rows], np.int64), 'K_in': np.stack([r['K_in'] for r in rows]),
         'dist': np.stack([r['dist'] for r in rows]), 'raw_size': np.stack([r['raw_size'] for r in rows]),
         'new_size': np.stack([r['new_size'] for r in rows]), 'stride_b': np.array([r['stride_b'] for r in rows]),
         'hm_win': np.stack([r['hm_win'] for r in rows]), 'kp_win': np.stack([r['kp_win'] for r in rows]),
         'kp_deployed': np.stack([r['kp_deployed'] for r in rows])}
    with open(os.path.join(root, 'kz_members_synth.npz'), 'wb') as f:
        np.savez(f, **A)
    jdump({'member_names': names, 'n_rgb': 5, 'search': 2, 'KP_SCALE': 64.0, 'W': 512.0, 'H': 288.0,
           'dec_kwargs': {'euler_seq': SEQ}, 'kp2d_dtype': ['torch.float32']}, os.path.join(root, 'kz_members_synth.json'))
    info = {'calib_lwh': lwh.tolist(), 'n': N}
    for p, F in [('real_ens5flip_kpKZavg_zoom_synth.pkl', frames), (OLD_AVG, frames)] + \
            [(OLD_SINGLE[n], old_single[n]) for n in names] + [(OLD_PAIR[t], old_pair[t]) for t, _ in MM_PAIRS]:
        pickle.dump({'info': info, 'frames': F}, open(os.path.join(root, p), 'wb'))
    return root


def stage_selftest(a):
    t0 = time.time()
    selftest(a)
    decoder_crosscheck(a)
    root = os.path.join(a.dir, 'selftest_fixture')
    make_fixture(root)
    print('[selftest] fixture written to %s (%.0f s)' % (root, time.time() - t0))
    b = argparse.Namespace(**vars(a))
    b.dir, b.tag, b.old_root = root, 'synth', root
    b.boot, b.boot_mm = 500, 200
    ck = stage_check(b)
    assert ck['pass'], ck['fails']
    stage_freeze(b)
    stage_evaluate(b)
    print('[selftest] end-to-end check / freeze / evaluate on the fixture: OK (%.0f s)' % (time.time() - t0))
    # null fixture: all members agree -> SA-TM == C0 everywhere -> the gate must stop the run before ground truth is read
    root0 = os.path.join(a.dir, 'selftest_fixture_null')
    make_fixture(root0, N=200, null=True)
    b.dir, b.old_root = root0, root0
    assert stage_check(b)['pass']
    J = stage_freeze(b)
    assert not J['gate']['pass'] and J['strata_share']['all']['agree'] > 0.9
    try:
        stage_evaluate(b)
        raise AssertionError('evaluate ran although the gate failed')
    except SystemExit as e:
        print('[selftest] null fixture: gate %.3f < 0.05 -> evaluate refused (%s): OK' % (J['gate']['frac_changed'], e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['check', 'freeze', 'evaluate', 'synth', 'selftest'])
    ap.add_argument('--dir', default=HERE)
    ap.add_argument('--tag', default='e1sa')
    ap.add_argument('--old-root', default=OLD_ROOT)
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--boot-mm', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261003)
    ap.add_argument('--n-synth', type=int, default=800)
    ap.add_argument('--geom-from-dump', action='store_true', help='synth: use K_in / deployed t / KZ s0 rotation of the existing dumps (label-free)')
    ap.add_argument('--skip-ads', action='store_true')
    ap.add_argument('--ads-deployed', action='store_true', help='also run ADS on the deployed dump (sanity vs 49.84)')
    ap.add_argument('--allow-check-fail', action='store_true', help='NOT for the real run (prereg: stop and fix)')
    ap.add_argument('--allow-gate-fail', action='store_true', help='NOT for the real run (prereg: stop, null effect)')
    a = ap.parse_args()
    out = {'check': stage_check, 'freeze': stage_freeze, 'evaluate': stage_evaluate, 'synth': stage_synth, 'selftest': stage_selftest}[a.stage](a)
    if a.stage == 'check' and not out['pass']:
        sys.exit(1)                     # prereg: replication failed -> stop and fix
    if a.stage == 'freeze' and not out['gate']['pass']:
        sys.exit(2)                     # prereg: gate failed -> stop, null effect, ground truth is not read


if __name__ == '__main__':
    main()
