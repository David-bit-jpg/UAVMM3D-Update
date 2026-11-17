# -*- coding: utf-8 -*-
"""A2: re-fuse PST's deployed (pre-fusion) distance with a second depth from another reader, with the unchanged chain_x.cmd_deploy fusion rule
and frozen omega = 0.190:  log z = (1 - w) log z_PST + w (log z_2 - b'_k),  b'_k = per PST colour class median log(z_2 / z_PST) on unlabeled val
images (re-estimated for each second-depth source, exactly as the deploy rule does for its own corner reader).
z_PST = preds/pcmd_nofuse_<sp>.pkl (refined distance, size coupling, r_k height, PST student rotation). Everything except the position is unchanged.
Labels: only for scoring (BU.stats == pack real_test metrics) and for descriptive error correlations."""
import json, pickle, sys
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA')
from pa_common import *  # noqa
import chain_x as CX  # noqa
WK = 'D:/S2MUAV3D/srcx_work'
DEP = json.load(open(X + '/deploy_pcmd.json')); W_ = DEP['w']; cc = np.asarray(DEP['col_centres'])
NM = {'val': 'val1', 'test': 'te1'}
CORNER = {   # name: (crop-set dir, student readings dir)
    'PST students (deployed, seed 0)': (WK + '/in2mav', X + '/ou'),
    'PST students, seed run +10': (WK + '/in2mav_seed2', R13 + '/sup3/run/ou'),
    'PST students, seed run +20': (WK + '/in2mav_sd20', R13 + '/seeds/mav_full_s20/ou'),
    'PST students w/o mirror loss': (R13 + '/clean3/nomir/work', R13 + '/clean3/nomir/ou'),
    'PST students w/o IR-LiDAR check': (R13 + '/sup2/nogate/work', R13 + '/sup2/nogate/ou'),
    'RGB-only students (seed 0)': (WK + '/in2mav_none', XN + '/ou'),
    'RGB-only students, seed run +10': (WK + '/in2mav_none_sd10', R13 + '/seeds/mav_none_s10/ou'),
    'RGB-only students, seed run +20': (WK + '/in2mav_none_sd20', R13 + '/seeds/mav_none_s20/ou'),
}


def z_from_preds(fn):
    return {(r['seq'], r['frame']): float(np.asarray(r['boxes9d']).reshape(-1, 9)[int(np.argmax(r['scores']))][2]) for r in P(fn)}


def z_from_recs(chain, name, sp):
    return {(r['seq'], r['frame']): float(r['box'][2]) for r in recs(chain, name)['recs'][sp] if r['box'] is not None}


def cls_map(sp):
    U = P('%s/ref/inf_fin_det1_%s.pkl' % (X, sp))
    return {k: CX.cls_of(cc, u['lum']) for k, u in U.items()}


def refuse(base, Z, CL, w=W_):
    """base: list of nofuse pred dicts; Z: (seq, frame) -> second depth (val and test dict); returns fused lists + b'_k + counts"""
    Bv = {}
    for r in base['val']:
        k = (r['seq'], r['frame']); z = Z['val'].get(k); c = CL['val'].get(k, -1)
        if z is not None and c >= 0:
            Bv.setdefault(c, []).append(np.log(z / r['boxes9d'][0, 2]))
    Bv = {c: float(np.median(v)) for c, v in Bv.items()}
    out, nf = {}, {}
    for sp in ('val', 'test'):
        L = []; n = 0
        for r in base[sp]:
            q = r['boxes9d'][0].copy(); k = (r['seq'], r['frame']); z = Z[sp].get(k); c = CL[sp].get(k, -1)
            if w > 0 and z is not None and c in Bv:
                lz = (1 - w) * np.log(q[2]) + w * (np.log(z) - Bv[c]); q[:3] *= np.exp(lz) / q[2]; n += 1
            L.append(dict(seq=r['seq'], frame=r['frame'], boxes9d=q.reshape(1, 9), scores=r['scores']))
        out[sp] = L; nf[sp] = n
    return out, Bv, nf


def main():
    base = {sp: [dict(seq=r['seq'], frame=r['frame'], boxes9d=np.asarray(r['boxes9d'], np.float64).reshape(1, 9).copy(), scores=np.asarray(r['scores']))
                 for r in P('%s/preds/pcmd_nofuse_%s.pkl' % (X, sp))] for sp in ('val', 'test')}
    CL = {sp: cls_map(sp) for sp in ('val', 'test')}
    SRC = {}
    for nm, (wd, od) in CORNER.items():
        SRC[nm] = {sp: zl_table('%s/oc_%s.pkl' % (wd, NM[sp]), '%s/read_%s_stu.pkl' % (od, NM[sp]),
                                tag=nm.replace(' ', '_').replace(',', '').replace('+', 'p').replace('/', '') + '_' + sp) for sp in ('val', 'test')}
        SRC[nm] = {sp: {k: v['z'] for k, v in SRC[nm][sp].items()} for sp in SRC[nm]}
    SRC['control: sim-only detector depth (det0)'] = {sp: z_from_recs(X, 'det0', sp) for sp in ('val', 'test')}
    SRC['control: PST self-trained detector depth (det1, before refiner)'] = {sp: z_from_recs(X, 'det1', sp) for sp in ('val', 'test')}
    SRC['control: RGB-only chain final depth (its pcmd_nofuse)'] = {sp: z_from_preds('%s/preds/pcmd_nofuse_%s.pkl' % (XN, sp)) for sp in ('val', 'test')}
    SRC['control: RGB-only chain self-trained detector depth (its det1)'] = {sp: z_from_recs(XN, 'det1', sp) for sp in ('val', 'test')}
    res = dict(w=W_, sources={}, check={})
    arms = {'PST without fusion (w = 0)': {sp: base[sp] for sp in base}}
    for nm, Z in SRC.items():
        fl, Bv, nf = refuse(base, Z, CL)
        arms['fused with: ' + nm] = fl; res['sources'][nm] = dict(b_k=Bv, n_fused=nf, n_z={sp: len(Z[sp]) for sp in Z})
    # reproduction: the deployed corner reader must give exactly pcmd_fused
    for sp in ('val', 'test'):
        ref = {(r['seq'], r['frame']): np.asarray(r['boxes9d']).reshape(-1, 9)[0] for r in P('%s/preds/pcmd_fused_%s.pkl' % (X, sp))}
        mine = arms['fused with: PST students (deployed, seed 0)'][sp]
        res['check'][sp] = float(max(np.max(np.abs(ref[(r['seq'], r['frame'])] - r['boxes9d'][0])) for r in mine))
    print('reproduction max |diff|', res['check'], 'b_k', res['sources']['PST students (deployed, seed 0)']['b_k'], DEP['b_k_val_images'], flush=True)
    assert max(res['check'].values()) < 1e-6
    # scoring + paired sequence bootstrap (B = 2000, seed 0), plus descriptive error stats of each second depth on val/test
    KEYS = ('UDS', 'AP', 'ATE')
    for sp in ('val', 'test'):
        M = meta(sp); seq = np.array([str(s) for s in M['seq']]); N = len(seq)
        key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
        FF = {nm: BU.frame_q(BU.load_list(L[sp], M), M) for nm, L in arms.items()}
        rows = {nm: BU.stats(F, np.arange(N)) for nm, F in FF.items()}
        ref = 'PST without fusion (w = 0)'; pst = 'fused with: PST students (deployed, seed 0)'
        us = np.unique(seq); idx = {u: np.where(seq == u)[0] for u in us}; rng = np.random.default_rng(0)
        acc = {nm: {k: [] for k in KEYS} for nm in arms}; accp = {nm: {k: [] for k in KEYS} for nm in arms}
        for _ in range(BU.B_BOOT):
            ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
            st = {nm: BU.stats(FF[nm], ii) for nm in arms}
            for nm in arms:
                for k in KEYS:
                    acc[nm][k].append(st[nm][k] - st[ref][k]); accp[nm][k].append(st[nm][k] - st[pst][k])
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        # error stats of the second depth itself after its own b'_k (GT used only here, descriptive): spread, corr. with PST pre-fusion error
        gz = M['gt'][:, 2]; e_base = np.full(N, np.nan)
        for r in base[sp]:
            i = key.get((str(r['seq']), str(r['frame'])))
            if i is not None:
                e_base[i] = np.log(r['boxes9d'][0, 2] / gz[i])
        desc = {}
        for nm, Z in SRC.items():
            Bv = res['sources'][nm]['b_k']; e2 = np.full(N, np.nan)
            for r in base[sp]:
                k = (r['seq'], r['frame']); i = key.get((str(k[0]), str(k[1]))); c = CL[sp].get(k, -1)
                if i is not None and k in Z[sp] and c in Bv:
                    e2[i] = np.log(Z[sp][k] / gz[i]) - Bv[c]
            ok = np.isfinite(e2) & np.isfinite(e_base) & (np.abs(e_base) < 0.5) & (np.abs(e2) < 0.5)   # gross misdetections out (descriptive only)
            r_ = float(np.corrcoef(e2[ok], e_base[ok])[0, 1])
            s1 = 1.4826 * np.median(np.abs(e_base[ok] - np.median(e_base[ok]))); s2 = 1.4826 * np.median(np.abs(e2[ok] - np.median(e2[ok])))
            # inverse-variance weight this source would deserve on these frames (oracle, descriptive): w* = (s1^2 - r s1 s2) / (s1^2 + s2^2 - 2 r s1 s2)
            wst = (s1 ** 2 - r_ * s1 * s2) / (s1 ** 2 + s2 ** 2 - 2 * r_ * s1 * s2)
            desc[nm] = dict(n=int(ok.sum()), bias_after_bk=float(np.median(e2[ok])), spread=float(s2), spread_pst=float(s1), corr_with_pst=r_, w_oracle=float(wst))
        res[sp] = dict(rows=rows, desc=desc,
                       vs_nofuse={nm: {k: dict(point=float(rows[nm][k] - rows[ref][k]), ci=ci(acc[nm][k]), frac_pos=float(np.mean(np.array(acc[nm][k]) > 0))) for k in KEYS} for nm in arms},
                       vs_pst={nm: {k: dict(point=float(rows[nm][k] - rows[pst][k]), ci=ci(accp[nm][k])) for k in KEYS} for nm in arms})
        for nm in arms:
            v = res[sp]['vs_nofuse'][nm]['UDS']; vp = res[sp]['vs_pst'][nm]['UDS']; d = desc.get(nm.replace('fused with: ', ''), {})
            print('%-5s %-70s UDS %.2f  gain %+.2f [%+.2f,%+.2f]  vsPST %+.2f [%+.2f,%+.2f]  corr %.2f spread %.3f w* %.2f' % (
                sp, nm, rows[nm]['UDS'], v['point'], *v['ci'], vp['point'], *vp['ci'], d.get('corr_with_pst', np.nan), d.get('spread', np.nan), d.get('w_oracle', np.nan)), flush=True)
    json.dump(res, open(PA + '/a2_second_depth.json', 'w', encoding='utf-8'), indent=1, default=float)


if __name__ == '__main__':
    main()
