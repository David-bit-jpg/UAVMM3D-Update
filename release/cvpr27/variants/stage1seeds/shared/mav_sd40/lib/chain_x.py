# [review1113 copy of srcx/lib/chain_x.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/mav_sd40/lib] sys.path -> redirected sx; code unchanged
# -*- coding: utf-8 -*-
"""[outdoor->srcx] label-free glue of the final PCMD chain (copies of the rules in wf23/fusetrain/build_targets.py, nolabel/fuse_eval_nl.py,
wf21 final chain_eval / MODULES §6). Reads NO real labels.
  targets <w>     fused stage-3 pseudo-targets on the refiner's real crops (ref/real_d0):
                  delta_pl = log(z_pl / z_det0) where z_pl = Module-3 pseudo-label depth; crop used only if the crop detection agrees with the
                  pseudo-label within 14.3% of its distance ([outdoor] indoor 0.5 m at the ~3.5 m MAV6D median distance);
                  delta_new = delta_pl + w * (log z_L - log z_pl - b_k),  z_L = corner-PnP depth of the orientation student (+ mirror) on the
                  final detector's train crops (used if that detection agrees with the pseudo-label within the same 14.3%);
                  b_k = per colour class median(log z_L / z_pl) on unlabeled train; colour class from the stage-2 refiner silhouette brightness.
  deploy <w> <out_prefix>   constants (colour centres by 1-D two-means, c_k, r_k on unlabeled train; b_k on unlabeled val images; s_bar from the
                  detector run) and single-frame deployment on val/test -> prediction pickles for score.py
Inputs are fixed file names produced by run_outdoor.sh."""
import os, sys, pickle, json
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/mav_sd40/lib'); import sx  # noqa  [srcx] per source/target constants
O = sx.O
V = O
import ori_x as OO  # noqa
from scipy.spatial.transform import Rotation as Rot  # noqa
GUARD = float(np.log(1.4)); AGREE = sx.T['agree_rel']; ALPHA = 0.5


def two_means(v):
    if sx.NOIR:
        return np.zeros(2)
    c = np.percentile(v, [25, 75]).astype(float)
    for _ in range(100):
        a = np.abs(v[:, None] - c[None]).argmin(1)
        c2 = np.array([v[a == k].mean() for k in range(2)])
        if np.allclose(c2, c):
            break
        c = c2
    return np.sort(c)


def cls_of(cc, lum):
    if sx.NOIR:   # [11-12] no IR: the colour class comes from the IR-supervised silhouette -> one class for every crop
        return 0
    return int(np.argmin(np.abs(cc - lum))) if lum is not None else -1


def zL_table(name, reads):
    """per (seq, frame): corner-PnP depth and rotation of the student ensemble (orig + mirror) on the crops of set <name>"""
    I = pickle.load(open('%s/oc_%s.pkl' % (OO.W, name), 'rb')); lwh = np.asarray(I['size'], np.float64)
    R = pickle.load(open(reads, 'rb')); out = {}
    for i, f in enumerate(I['info']):
        s = OO.ensemble(R['reads'], R['order'], i, R['cells'][i], f['K_crop'], lwh)
        if s is not None and s[1][2] > 0:
            out[(f['seq'], f['frame'])] = dict(z=float(s[1][2]), R=s[0])
    return out


def cmd_targets(w):
    ix = pickle.load(open('%s/train/index_pseudo_%s.pkl' % (sx.REAL, sx.PSEUDO_TAG), 'rb'))
    ps = {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64)[0] for m in ix['metas'] if m is not None}
    D1 = {(r['seq'], r['frame']): r for r in pickle.load(open(O + '/recs/det1.pkl', 'rb'))['recs']['train'] if r['box'] is not None}
    U2 = pickle.load(open(V + '/ref/inf_st2_det1_train.pkl', 'rb'))
    cc = two_means(np.array([u['lum'] for u in U2.values() if u['lum'] is not None]))
    ZL = zL_table('tr1', V + '/ou/read_tr1_stu.pkl')
    dd = {}
    for k, z in ZL.items():
        if k in ps and k in D1 and np.linalg.norm(D1[k]['box'][:3] - ps[k][:3]) < AGREE * np.linalg.norm(ps[k][:3]):
            dd[k] = float(np.log(z['z'] / ps[k][2]))
    cl = lambda k: cls_of(cc, U2[k]['lum']) if k in U2 else -1
    B = {}
    for c in (0, 1):
        v = [x for k, x in dd.items() if cl(k) == c]
        if v:
            B[c] = float(np.median(v))
    meta = pickle.load(open(O + '/ref/real_d0_meta.pkl', 'rb'))
    dp = np.full(len(meta), np.nan, np.float32); nf = 0
    D0 = {(r['seq'], r['frame']): r for r in pickle.load(open(O + '/recs/det0.pkl', 'rb'))['recs']['train'] if r['box'] is not None}
    for i, m in enumerate(meta):
        k = (m['seq'], m['frame'])
        if k not in ps:
            continue
        if np.linalg.norm(D0[k]['box'][:3] - ps[k][:3]) >= AGREE * np.linalg.norm(ps[k][:3]):
            continue
        d = np.log(ps[k][2] / m['z_det'])
        if k in dd and cl(k) in B:
            d += w * (dd[k] - B[cl(k)]); nf += 1
        dp[i] = d
    np.save(V + '/ref/dp_fused.npy', dp)
    info = dict(w=w, col_centres=cc.tolist(), b_train=B, n_zL=len(ZL), n_dd=len(dd), n_crops=len(meta), n_target=int(np.isfinite(dp).sum()), n_fused=nf)
    json.dump(info, open(V + '/targets_info.json', 'w'), indent=1); print('targets', info)


def cmd_deploy(w, prefix):
    D = pickle.load(open(O + '/recs/det1.pkl', 'rb')); sbar = np.asarray(D['size'], np.float64)
    U = {sp: pickle.load(open(V + '/ref/inf_fin_det1_%s.pkl' % sp, 'rb')) for sp in ('train', 'val', 'test')}
    KP = {sp: pickle.load(open(O + '/kps_%s.pkl' % sp, 'rb')) for sp in ('tr1', 'val1', 'te1')}
    ZL = {'val': zL_table('val1', V + '/ou/read_val1_stu.pkl'), 'test': zL_table('te1', V + '/ou/read_te1_stu.pkl')}
    tr = [u for u in U['train'].values()]
    cc = two_means(np.array([u['lum'] for u in tr if u['lum'] is not None]))
    C = {}
    for c in (0, 1):
        v = np.array([u['d'] for u in tr if cls_of(cc, u['lum']) == c and abs(u['d']) <= GUARD])
        C[c] = float(np.median(v))
    allv = np.array([u['d'] for u in tr if abs(u['d']) <= GUARD]); C[-1] = float(np.median(allv))
    rk = {}
    for c in (0, 1):
        v = [KP['tr1'][k] for k, u in U['train'].items() if cls_of(cc, u['lum']) == c and k in KP['tr1']]
        rk[c] = float(np.median(v))
    rk_all = float(np.median(list(KP['tr1'].values())))

    def base(sp, r):
        """single-frame chain before the deployment fusion: refiner correction, size coupling, height ratio"""
        k = (r['seq'], r['frame']); q = np.asarray(r['box'], np.float64).copy(); u = U[sp].get(k)
        c = cls_of(cc, u['lum']) if u is not None else -1
        f = 0.0
        if u is not None and abs(u['d']) <= GUARD:
            f = u['d'] - C[c]
        q[:3] *= np.exp(f)
        l, wd = sbar[0] * np.exp(ALPHA * f), sbar[1] * np.exp(ALPHA * f)
        kpk = {'val': 'val1', 'test': 'te1'}[sp]
        h = rk[c] * (l + wd) / 2 if c >= 0 else KP[kpk].get(k, rk_all) * (l + wd) / 2
        q[3:6] = [l, wd, h]
        return q, c
    Bv = {}
    for r in D['recs']['val']:
        if r['box'] is None:
            continue
        q, c = base('val', r); z = ZL['val'].get((r['seq'], r['frame']))
        if z is not None and c >= 0:
            Bv.setdefault(c, []).append(np.log(z['z'] / q[2]))
    Bv = {c: float(np.median(v)) for c, v in Bv.items()}
    res = dict(w=w, s_bar=sbar.tolist(), col_centres=cc.tolist(), c_k=C, r_k=rk, b_k_val_images=Bv, alpha=ALPHA)
    for sp in ('val', 'test'):
        for tag, ww in (('fused', w), ('nofuse', 0.0)):
            P = []; nf = nr = 0
            for r in D['recs'][sp]:
                if r['box'] is None:
                    continue
                q, c = base(sp, r); z = ZL[sp].get((r['seq'], r['frame']))
                if z is not None:
                    q[6:9] = Rot.from_matrix(z['R']).as_euler('xyz'); nr += 1
                if ww > 0 and z is not None and c in Bv:
                    lz = (1 - ww) * np.log(q[2]) + ww * (np.log(z['z']) - Bv[c])
                    q[:3] *= np.exp(lz) / q[2]; nf += 1
                P.append(dict(seq=r['seq'], frame=r['frame'], boxes9d=q.reshape(1, 9), scores=np.array([r['conf']])))
            fn = '%s/preds/%s_%s_%s.pkl' % (V, prefix, tag, sp)
            pickle.dump(P, open(fn, 'wb')); res['%s_%s' % (tag, sp)] = dict(n=len(P), n_fused=nf, n_rot_student=nr, file=fn)
    json.dump(res, open('%s/deploy_%s.json' % (V, prefix), 'w'), indent=1, default=float); print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    if sys.argv[1] == 'targets':
        cmd_targets(float(sys.argv[2]))
    else:
        cmd_deploy(float(sys.argv[2]), sys.argv[3])
