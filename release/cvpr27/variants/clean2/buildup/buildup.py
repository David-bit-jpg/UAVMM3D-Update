# -*- coding: utf-8 -*-
"""[clean2 item 1] Step-by-step build-up of the clean-protocol MAV6D chain (srcx/in2mav, test UDS 60.81). CPU only, existing predictions only.
Rows (all single-frame, all from the chain's own files):
  (a) sim-only detector            = preds/srconly_<sp>.pkl (3 seeds + flip, det0)
  (b) + step-1 self-trained det    = recs/det1.pkl as is (3 seeds + flip; its own rotation R_det; size = s_bar of det1, i.e. the det's own output)
  (c) + step-2 orientation students: det1 box, rotation replaced by the student ensemble (S_ab0, S_ab1 + mirror; corner PnP with s_bar) by the
                                     deploy rule of chain_x.cmd_deploy (q[6:9] = student R where a reading exists); position / size unchanged
  (c') (c) + LiDAR height ratio r_k  = chain_x.cmd_deploy base() with the refiner correction forced to 0 (diagnostic: isolates the depth step)
  (d) + step-3 depth students      = preds/pcmd_nofuse_<sp>.pkl (refined distance + size coupling + r_k height; no fusion)
  (e) + test-time fusion           = preds/pcmd_fused_<sp>.pkl (= 60.81 on test)
Metrics: pack score.py 'real_test' (via srcx/lib/score_x.score_meta, asserted equal), plus median / mean quadrotor-symmetric yaw on the scorer's TP set
(= wf21/orient_uda/eval_orient.py ang_metrics 'hs': min_k |yaw(R_gt^T R_pred Rz(90k))|; the definition behind tab_orient).
Paired sequence bootstrap (B = 2000, seed 0) of consecutive differences: same resampled sequences for both arms; AP recomputed on each resample.
Labels are read only for scoring."""
import os, sys, json, pickle, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as Rot
os.environ.update(SX_SRC='in', SX_TGT='mav', SX_TEACH='full')
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'; X = G + '/gproto/srcx/in2mav'; L = G + '/gproto/srcx/lib'
sys.path.insert(0, L)
import chain_x as CX  # noqa  (read-only use: zL_table, two_means, cls_of, constants)
import score_x as SX  # noqa
S = SX.S; CFG = S.CFG['real_test']
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
B_BOOT = 2000


def load_list(L_, M):
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; N = len(M['seq'])
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in L_:
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def yawerr(Rp, Rg):
    E = Rg.T @ Rp
    return abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))


def frame_q(P, M):
    g = M['gt']; K = M['K']; N = len(g)
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf'])
    d = np.linalg.norm(P['pt'] - g[:, :3], axis=1)
    tp = np.zeros(N, bool); se = np.full(N, np.nan); ae = np.full(N, np.nan); hs = np.full(N, np.nan)
    for i in np.where(has)[0]:
        fe = np.isfinite(P['peul'][i]).all(); eul = P['peul'][i] if fe else np.zeros(3)
        if S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(P['plwh'][i] - g[i, 3:6]))
        Rg = Rot.from_euler('xyz', g[i, 6:9]).as_matrix()
        if fe:
            Rp = Rot.from_euler('xyz', eul).as_matrix(); ae[i] = S.sym(Rp, Rg); hs[i] = min(yawerr(Rp @ RZ[q], Rg) for q in range(4))
        else:
            ae[i] = 90.0
    return dict(has=has, conf=P['conf'], d=d, tp=tp, se=se, ae=ae, hs=hs)


RS = np.linspace(0, 1, 101); RS = RS[RS > 0.1]


def fast_ap(sc, hit, n_gt):
    if not len(sc):
        return 0.0
    o = np.argsort(-sc); t = hit[o].astype(float)
    ctp = np.cumsum(t); cfp = np.cumsum(1 - t); rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1)
    smax = np.maximum.accumulate(prec[::-1])[::-1]; k = np.searchsorted(rec, RS, side='left')
    p = np.where(k < len(rec), smax[np.minimum(k, len(rec) - 1)], 0.0)
    return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)


def stats(F, idx):
    h = F['has'][idx]; n = len(idx)
    ap = 100 * np.mean([fast_ap(F['conf'][idx][h], F['d'][idx][h] <= th, n) for th in CFG['thr']])
    t = idx[F['tp'][idx]]
    ATE, ASE, AOE = F['d'][t].mean(), F['se'][t].mean(), F['ae'][t].mean(); mdr = len(t) / n
    uds = 100 * (0.5 * ap / 100 + 0.5 * np.mean([1 - min(1, ATE / CFG['a']), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr]))
    return dict(UDS=uds, AP=ap, ATE=ATE, ASE=ASE, AOE=AOE, mDR=100 * mdr, yaw_med=float(np.nanmedian(F['hs'][t])), yaw_mean=float(np.nanmean(F['hs'][t])),
                n_tp=int(len(t)))


def build_c(sp, with_rk):
    """(c) / (c'): det1 boxes, student rotation by the deploy rule; depth unchanged; (c') adds the deploy height h = r_k (l+w)/2 (f = 0)."""
    D = pickle.load(open(X + '/recs/det1.pkl', 'rb')); sbar = np.asarray(D['size'], np.float64)
    DEP = json.load(open(X + '/deploy_pcmd.json')); cc = np.asarray(DEP['col_centres']); rk = {int(k): v for k, v in DEP['r_k'].items()}
    U = pickle.load(open('%s/ref/inf_fin_det1_%s.pkl' % (X, sp), 'rb'))
    KP = pickle.load(open('%s/kps_%s.pkl' % (X, {'val': 'val1', 'test': 'te1'}[sp]), 'rb'))
    KPtr = pickle.load(open(X + '/kps_tr1.pkl', 'rb')); rk_all = float(np.median(list(KPtr.values())))
    ZL = CX.zL_table({'val': 'val1', 'test': 'te1'}[sp], '%s/ou/read_%s_stu.pkl' % (X, {'val': 'val1', 'test': 'te1'}[sp]))
    out = []; nr = 0
    for r in D['recs'][sp]:
        if r['box'] is None:
            continue
        k = (r['seq'], r['frame']); q = np.asarray(r['box'], np.float64).copy()
        if with_rk:
            u = U.get(k); c = CX.cls_of(cc, u['lum']) if u is not None else -1
            l, wd = sbar[0], sbar[1]
            h = rk[c] * (l + wd) / 2 if c >= 0 else KP.get(k, rk_all) * (l + wd) / 2
            q[3:6] = [l, wd, h]
        z = ZL.get(k)
        if z is not None:
            q[6:9] = Rot.from_matrix(z['R']).as_euler('xyz'); nr += 1
        out.append(dict(seq=r['seq'], frame=r['frame'], boxes9d=q.reshape(1, 9), scores=np.array([r['conf']])))
    return out, nr


if __name__ == '__main__':
    res = {}
    for sp in ('val', 'test'):
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); seq = np.array([str(s) for s in M['seq']]); N = len(seq)
        D1 = pickle.load(open(X + '/recs/det1.pkl', 'rb'))['recs'][sp]
        det1 = [dict(seq=r['seq'], frame=r['frame'], boxes9d=np.asarray(r['box'], np.float64).reshape(1, 9), scores=np.array([r['conf']])) for r in D1 if r['box'] is not None]
        cL, nrc = build_c(sp, False); cpL, _ = build_c(sp, True)
        for nm, L_ in (('c_students', cL), ('cp_students_rk', cpL), ('b_det1', det1)):
            pickle.dump(L_, open('%s/%s_%s.pkl' % (OUT, nm, sp), 'wb'))
        arms = {'(a) sim-only detector': pickle.load(open('%s/preds/srconly_%s.pkl' % (X, sp), 'rb')),
                '(b) + self-trained detector (R_det)': det1,
                '(c) + orientation students': cL,
                "(c') (c) + LiDAR height ratio r_k": cpL,
                '(d) + depth students (no fusion)': pickle.load(open('%s/preds/pcmd_nofuse_%s.pkl' % (X, sp), 'rb')),
                '(e) + test-time fusion (PST)': pickle.load(open('%s/preds/pcmd_fused_%s.pkl' % (X, sp), 'rb'))}
        files = {'(b) + self-trained detector (R_det)': '%s/b_det1_%s.pkl' % (OUT, sp), '(c) + orientation students': '%s/c_students_%s.pkl' % (OUT, sp),
                 "(c') (c) + LiDAR height ratio r_k": '%s/cp_students_rk_%s.pkl' % (OUT, sp)}
        PP = {nm: load_list(L_, M) for nm, L_ in arms.items()}
        # (c) rotation must equal the deployed nofuse rotation wherever the student read
        Pd = PP['(d) + depth students (no fusion)']; Pc = PP['(c) + orientation students']
        okr = np.isfinite(Pc['peul']).all(1) & np.isfinite(Pd['peul']).all(1)
        rot_dev = max(S.sym(Rot.from_euler('xyz', Pc['peul'][i]).as_matrix(), Rot.from_euler('xyz', Pd['peul'][i]).as_matrix()) for i in np.where(okr)[0])
        FF, rows = {}, {}
        allidx = np.arange(N)
        for nm, P in PP.items():
            F = frame_q(P, M); FF[nm] = F; s = stats(F, allidx)
            ref = S.metrics(P, M['gt'], M['K'], allidx, CFG)
            assert all(abs(ref[k] - s[k]) < 1e-9 for k in ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR')), (nm, ref, s)
            if nm in files:   # also via the chain's scorer entry point
                sm = SX.score_meta(files[nm], '%s/cmpeval/%s_meta.npz' % (G, sp))['all']; assert abs(sm['UDS'] - s['UDS']) < 1e-9
            rows[nm] = s
        names = list(PP.keys())
        pairs = [(names[1], names[0]), (names[2], names[1]), (names[4], names[2]), (names[3], names[2]), (names[4], names[3]), (names[5], names[4]), (names[5], names[0])]
        us = np.unique(seq); idx = {u: np.where(seq == u)[0] for u in us}
        rng = np.random.default_rng(0); acc = {p: {k: [] for k in ('UDS', 'AOE', 'yaw_med', 'AP', 'ATE')} for p in pairs}
        for _ in range(B_BOOT):
            ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
            st = {nm: stats(FF[nm], ii) for nm in names}
            for a, b in pairs:
                for k in acc[(a, b)]:
                    acc[(a, b)][k].append(st[a][k] - st[b][k])
        boot = {}
        for (a, b), A in acc.items():
            boot['%s  minus  %s' % (a, b)] = {k: dict(point=float(rows[a][k] - rows[b][k]), mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                                                         hi=float(np.percentile(v, 97.5)), frac_pos=float(np.mean(np.array(v) > 0))) for k, v in A.items()}
        res[sp] = dict(rows=rows, boot=boot, n_seq=int(len(us)), n_frames=int(N), n_rot_student_c=nrc, max_rot_dev_c_vs_nofuse_deg=float(rot_dev))
        print(sp, json.dumps(rows, default=lambda x: round(float(x), 3)), flush=True)
        print(json.dumps({k: {m: [round(v['point'], 2), round(v['lo'], 2), round(v['hi'], 2)] for m, v in d.items()} for k, d in boot.items()}, ensure_ascii=False), flush=True)
    json.dump(res, open(OUT + '/buildup.json', 'w', encoding='utf-8'), indent=1, default=float, ensure_ascii=False)
