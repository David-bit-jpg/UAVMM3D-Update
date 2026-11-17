# -*- coding: utf-8 -*-
"""Panel A shared helpers (CPU only, read-only on chain files). Labels are read only for evaluation."""
import os, sys, json, pickle, importlib.util
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('SX_SRC', 'in'); os.environ.setdefault('SX_TGT', 'mav'); os.environ.setdefault('SX_TEACH', 'full')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'; G = GP.rsplit('/', 1)[0]
X = GP + '/srcx/in2mav'; XN = GP + '/srcx/in2mav_none'; R13 = GP + '/review1113'
PA = R13 + '/panelA'; CACHE = PA + '/cache'
os.makedirs(CACHE, exist_ok=True)
REAL = 'E:/mmcache/mav6d_cn_obj'
LAM, CONF, GUARD, AGREE = 0.75, 0.5, float(np.log(1.4)), 0.5 / 3.5
spec = importlib.util.spec_from_file_location('buildup', R13 + '/clean2/buildup/buildup.py')
BU = importlib.util.module_from_spec(spec); spec.loader.exec_module(BU)   # imports chain_x / ori_x read-only (sx dirs already exist)
S = BU.S; CFG = BU.CFG
import ori_x as OO  # noqa  (same module object chain_x uses)


def P(fn):
    return pickle.load(open(fn, 'rb'))


def zl_table(oc_pkl, reads_pkl, tag=None):
    """= chain_x.zL_table, with the crop-set file given explicitly (so another chain's students can be read). Cached."""
    cf = '%s/zl_%s.pkl' % (CACHE, tag) if tag else None
    if cf and os.path.exists(cf):
        return P(cf)
    I = P(oc_pkl); lwh = np.asarray(I['size'], np.float64); R = P(reads_pkl); out = {}
    for i, f in enumerate(I['info']):
        s = OO.ensemble(R['reads'], R['order'], i, R['cells'][i], f['K_crop'], lwh)
        if s is not None and s[1][2] > 0:
            out[(f['seq'], f['frame'])] = dict(z=float(s[1][2]), R=s[0])
    if cf:
        pickle.dump(out, open(cf, 'wb'))
    return out


def gt_index(split):
    I = P('%s/%s/index.pkl' % (REAL, split))
    return {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64)[0] for m in I['metas'] if m is not None and len(m['boxes9d'])}


def meta(split):
    return np.load('%s/cmpeval/%s_meta.npz' % (G, split))


def recs(chain, name):
    return P('%s/recs/%s.pkl' % (chain, name))


def avg(U):   # = srcx/lib/avg_inf.py ('d' only)
    return {k: float(np.mean([u[k]['d'] for u in U])) for k in U[0] if all(k in u for u in U)}


def pseudo_table(det0_recs, dh, cbar):
    """make_pseudo_x.py rule: conf >= 0.5; t *= exp(d - LAM cbar) if |d| <= log 1.4"""
    T = {}
    for r in det0_recs:
        if r['box'] is None or r['conf'] < CONF:
            continue
        p = np.asarray(r['box'], np.float64).copy(); d = dh.get((r['seq'], r['frame']))
        if d is not None and abs(d) <= GUARD:
            p[:3] *= np.exp(d - LAM * cbar)
        T[(r['seq'], r['frame'])] = p
    return T


def cbar_of(det0_train, dh):
    d = np.array([dh[(r['seq'], r['frame'])] for r in det0_train if (r['seq'], r['frame']) in dh])
    return float(np.median(d[np.abs(d) <= GUARD]))


def relstats(e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]
    return dict(n=int(len(e)), med_abs_rel=float(np.median(np.abs(np.expm1(e)))) if len(e) else np.nan,
                med_log=float(np.median(e)) if len(e) else np.nan, mean_abs_log=float(np.mean(np.abs(e))) if len(e) else np.nan,
                frac_gt25=float(np.mean(np.abs(np.expm1(e)) > 0.25)) if len(e) else np.nan)


def fmt_ci(p, lo, hi, d=2, pct=False):
    f = (lambda x: '%+.*f' % (d, 100 * x)) if pct else (lambda x: '%+.*f' % (d, x))
    return '%s [%s, %s]' % (f(p), f(lo), f(hi))
