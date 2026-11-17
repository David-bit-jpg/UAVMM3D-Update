# -*- coding: utf-8 -*-
"""N2: chance-level / constant orientation baselines (CPU only). GT read only to score.
Metric = paper_table.sym / pack score.py sym: min_q geodesic(Rp Rz(90q), Rg) (arccos-trace form);
yaw = min_q |atan2(E10, E00)|, E = Rg^T Rp Rz(90q)  (0-45 deg), as in strict/final_eval.yawerr / ou_eval.errs."""
import os, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
OUT = os.path.dirname(os.path.abspath(__file__))
RZ = np.stack([Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)])
rng = np.random.default_rng(1113)


def errs(Rp, Rg):
    """Rp, Rg: (N,3,3) -> folded geodesic (deg), folded yaw (deg). Same formulas as the project code, vectorised."""
    geo = np.full(len(Rg), np.inf)
    yaw = np.full(len(Rg), np.inf)
    for q in range(4):
        Rq = Rp @ RZ[q]
        tr = np.einsum('nij,nij->n', Rq, Rg)            # trace(Rq^T Rg)
        geo = np.minimum(geo, np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
        E = np.transpose(Rg, (0, 2, 1)) @ Rq
        yaw = np.minimum(yaw, np.abs(np.degrees(np.arctan2(E[:, 1, 0], E[:, 0, 0]))))
    return geo, yaw


def chordal_sym(Rp, Rg):
    """KFO.geodesic form used by pseudo_build.sym (label-error check)."""
    out = np.full(len(Rg), np.inf)
    for q in range(4):
        d = np.linalg.norm((Rp @ RZ[q] - Rg).reshape(len(Rg), -1), axis=1) / (2 * np.sqrt(2))
        out = np.minimum(out, np.degrees(2 * np.arcsin(np.clip(d, 0, 1))))
    return out


def bc(R, n):
    return np.broadcast_to(R, (n, 3, 3)).copy()


def const_mean(R, Rg):
    return errs(bc(R, len(Rg)), Rg)[0].mean()


def best_const(Rg, n_cand=400, sub=4000):
    """Oracle constant minimising mean folded geodesic on Rg: GT-rotation candidates scored on a subsample,
    top 5 refined by Nelder-Mead over a body-side rotvec perturbation on the full set."""
    S = Rg if len(Rg) <= sub else Rg[rng.choice(len(Rg), sub, replace=False)]
    cand = Rg[rng.choice(len(Rg), min(n_cand, len(Rg)), replace=False)]
    sc = np.array([const_mean(c, S) for c in cand])
    best = []
    for c in cand[np.argsort(sc)[:5]]:
        r = minimize(lambda v: const_mean(c @ Rot.from_rotvec(v).as_matrix(), Rg), np.zeros(3), method='Nelder-Mead',
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=600))
        best.append((r.fun, c @ Rot.from_rotvec(r.x).as_matrix()))
    best.sort(key=lambda x: x[0])
    return best[0][1]


def level_facing(t=None):
    """Level and facing the camera: body z = camera up (-y_cam); body x toward camera, i.e. along -z_cam (t None)
    or along the horizontal projection of -t (GT line of sight)."""
    if t is None:
        xb = np.array([[0., 0., -1.]])
    else:
        v = -np.asarray(t, float).copy()
        v[:, 1] = 0
        xb = v / np.linalg.norm(v, axis=1, keepdims=True)
    zb = np.broadcast_to(np.array([0., -1., 0.]), xb.shape)
    yb = np.cross(zb, xb)
    return np.stack([xb, yb, zb], axis=2)


def row(split, name, geo, yaw, note=''):
    return dict(split=split, predictor=name, aoe_mean=float(np.mean(geo)), aoe_med=float(np.median(geo)),
                yaw_mean=float(np.mean(yaw)), yaw_med=float(np.median(yaw)), n=int(len(geo)), note=note)


def load_meta(f):
    g = np.load(f)['gt'].astype(np.float64)
    return Rot.from_euler('xyz', g[:, 6:9]).as_matrix(), g[:, :3]


def load_index(f):
    d = pickle.load(open(f, 'rb'))
    m = d['metas']
    idx = d.get('valid_idx', np.arange(len(m)))
    keep = [i for i in idx if len(m[i]['boxes9d'])]
    b = np.stack([np.asarray(m[i]['boxes9d'], np.float64).reshape(-1, 9)[0] for i in keep])
    return Rot.from_euler('xyz', b[:, 6:9]).as_matrix(), b[:, :3]


def main():
    sets = {}
    sets['MAV6D test'] = load_meta('E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval/test_meta.npz')
    sets['AntiUAV3D test'] = load_meta('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz')
    TRIDX = 'E:/mmcache/mav6d_cn_obj/train/index.pkl'
    Rtr, ttr = load_index(TRIDX)
    sets['MAV6D train (all GT frames)'] = (Rtr, ttr)
    # teacher-label frames (t6il = the 20.64 / 16.52 set): GT box nearest t_pos within 1 m, as pseudo_build --diag
    P = np.load('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/wf21/orient_uda/pseudo/pseudo_train_t6il.npz')
    d = pickle.load(open(TRIDX, 'rb'))
    key = {(m['seq'], m['frame']): m for m in d['metas']}
    Rg_l, t_l, Rt_l, gate_l = [], [], [], []
    for s, f, tp, R, gt in zip(P['seq'], P['frame'], P['t_pos'], P['R'], P['gate']):
        m = key.get((str(s), str(f)))
        if m is None or not len(m['boxes9d']):
            continue
        b = np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)
        b = b[np.argmin(np.linalg.norm(b[:, :3] - tp, axis=1))]
        if np.linalg.norm(b[:3] - tp) > 1.0:
            continue
        Rg_l.append(Rot.from_euler('xyz', b[6:9]).as_matrix()); t_l.append(b[:3]); Rt_l.append(R); gate_l.append(bool(gt))
    Rg_l, t_l, Rt_l, gate_l = map(np.array, (Rg_l, t_l, Rt_l, gate_l))
    sets['MAV6D train, teacher-label frames (all)'] = (Rg_l, t_l)
    sets['MAV6D train, teacher-label frames (passing)'] = (Rg_l[gate_l], t_l[gate_l])
    gt_, gy_ = errs(Rt_l, Rg_l)
    teacher_chk = dict(all=float(chordal_sym(Rt_l, Rg_l).mean()), passing=float(chordal_sym(Rt_l[gate_l], Rg_l[gate_l]).mean()),
                       all_med=float(np.median(gt_)), passing_med=float(np.median(gt_[gate_l])),
                       yaw_all_mean=float(gy_.mean()), yaw_all_med=float(np.median(gy_)),
                       yaw_pass_mean=float(gy_[gate_l].mean()), yaw_pass_med=float(np.median(gy_[gate_l])),
                       n_all=int(len(Rg_l)), n_pass=int(gate_l.sum()))
    print('teacher check', teacher_chk, flush=True)

    Rsim, _ = load_index('E:/mmcache/full_v2/train/index.pkl')
    C_sim = best_const(Rsim)
    C_tr = best_const(Rtr)
    rows = []
    K = 50
    for name, (Rg, t) in sets.items():
        N = len(Rg)
        RgK = np.repeat(Rg, K, 0)
        Rh = Rot.random(N * K, random_state=int(rng.integers(1 << 31))).as_matrix()
        g, y = errs(Rh, RgK)
        rows.append(row(name, 'Random rotation (Haar)', g, y, '%d draws/frame' % K))
        psi = rng.uniform(-np.pi, np.pi, N * K)
        g, y = errs(RgK @ Rot.from_euler('z', psi).as_matrix(), RgK)
        rows.append(row(name, 'Random yaw about body z (GT tilt kept)', g, y, 'analytic U[0,45]: 22.5 / 22.5'))
        Co = best_const(Rg)
        for lab, C in (('Best constant (oracle, fit on this split)', Co), ('Constant fit on sim train labels', C_sim),
                       ('Constant fit on MAV6D real train GT', C_tr)):
            if 'real train' in lab and name.startswith('MAV6D train'):
                lab += ' (in-sample)'
            g, y = errs(bc(C, N), Rg)
            rows.append(row(name, lab, g, y))
        g, y = errs(bc(np.eye(3), N), Rg)
        rows.append(row(name, 'Identity in camera frame', g, y))
        g, y = errs(bc(level_facing()[0], N), Rg)
        rows.append(row(name, 'Level, body x toward camera (-z_cam)', g, y))
        g, y = errs(level_facing(t), Rg)
        rows.append(row(name, 'Level, facing camera along GT line of sight', g, y, 'uses GT position'))
        print(name, N, flush=True)
    json.dump(dict(rows=rows, teacher_check=teacher_chk,
                   C_sim_euler_xyz_deg=Rot.from_matrix(C_sim).as_euler('xyz', degrees=True).tolist(),
                   C_train_euler_xyz_deg=Rot.from_matrix(C_tr).as_euler('xyz', degrees=True).tolist()),
              open(OUT + '/n2_results.json', 'w'), indent=1)
    for r in rows:
        print('%-46s %-48s %6.2f %6.2f %6.2f %6.2f %6d' % (r['split'], r['predictor'], r['aoe_mean'], r['aoe_med'],
                                                         r['yaw_mean'], r['yaw_med'], r['n']))


if __name__ == '__main__':
    main()
