# -*- coding: utf-8 -*-
"""A1 (position pseudo-label quality stage by stage, train frames) and A8 (copy analysis without the 1 m filter, held-out val).
Clean MAV6D chain srcx/in2mav (PST) and the RGB-only control srcx/in2mav_none. CPU only; existing chain files only; GT read only to evaluate.
Error e = log(z / z_gt) (z = camera depth of the box centre; the chain scales t uniformly, so this equals the range log-ratio)."""
import json, sys
import numpy as np
from scipy.stats import spearmanr, theilslopes
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA')
from pa_common import *  # noqa
WK = 'D:/S2MUAV3D/srcx_work'
CH = {'PST': dict(X=X, W=WK + '/in2mav', noir=False), 'RGB-only': dict(X=XN, W=WK + '/in2mav_none', noir=True)}
rng = np.random.default_rng(0); B = 2000
res = {}


def cls_fn(dep, noir):
    cc = np.asarray(dep['col_centres'])
    if noir:
        return lambda lum: 0
    return lambda lum: int(np.argmin(np.abs(cc - lum))) if lum is not None else -1


def deploy_on(ch, sp, det1_recs, zl):
    """chain_x.cmd_deploy base() + fusion applied to the frames of split sp (train allowed: diagnostic of the deployed function on its training frames);
    constants exactly as deployed (deploy_pcmd.json: c_k on unlabeled train, b'_k on unlabeled val images, omega)."""
    dep = json.load(open(ch['X'] + '/deploy_pcmd.json')); cl = cls_fn(dep, ch['noir'])
    C = {int(k): v for k, v in dep['c_k'].items()}; Bv = {int(k): v for k, v in dep['b_k_val_images'].items()}; w = dep['w']
    U = P('%s/ref/inf_fin_det1_%s.pkl' % (ch['X'], sp))
    zr, zf = {}, {}
    for r in det1_recs:
        if r['box'] is None:
            continue
        k = (r['seq'], r['frame']); z = float(r['box'][2]); u = U.get(k); c = cl(u['lum']) if u is not None else -1
        if u is not None and abs(u['d']) <= GUARD:
            z *= np.exp(u['d'] - C[c])
        zr[k] = z; zz = zl.get(k)
        zf[k] = float(np.exp((1 - w) * np.log(z) + w * (np.log(zz['z']) - Bv[c]))) if (zz is not None and c in Bv) else z
    return zr, zf


def err(Z, GT, keys):
    return np.array([np.log(Z[k] / GT[k][2]) if (k in Z and Z[k] is not None and k in GT and np.isfinite(GT[k][2])) else np.nan for k in keys])


# ------------------------------------------------------------------ A1: train frames
GTtr = gt_index('train'); D0 = recs(X, 'det0'); D0tr = D0['recs']['train']
assert all((a['box'] is None) == (b['box'] is None) for a, b in zip(D0tr, recs(XN, 'det0')['recs']['train']))   # identical sim-only detectors
allk = [(r['seq'], r['frame']) for r in D0tr if r['box'] is not None]
keptk = [(r['seq'], r['frame']) for r in D0tr if r['box'] is not None and r['conf'] >= CONF]
z0 = {(r['seq'], r['frame']): float(r['box'][2]) for r in D0tr if r['box'] is not None}
box0 = {(r['seq'], r['frame']): np.asarray(r['box'], np.float64) for r in D0tr if r['box'] is not None}
far = {k: (k in GTtr and np.linalg.norm(box0[k][:3] - GTtr[k][:3]) >= 1.0) for k in allk}
A1 = dict(n_train_dets=len(allk), n_kept=len(keptk), frac_kept_far_from_gt_1m=float(np.mean([far[k] for k in keptk])), arms={})
STG = {}
for nm, ch in CH.items():
    U2 = {k: v['d'] for k, v in P(ch['X'] + '/ref/inf_st2_det0_train.pkl').items()}
    cbar = cbar_of(D0tr, U2)
    dk = np.array([U2[k] for k in keptk if k in U2]); cbar_kept = float(np.median(dk[np.abs(dk) <= GUARD]))
    dd = np.array([U2[k] for k in allk if k in U2 and k not in set(keptk)]);
    PL = pseudo_table(D0tr, U2, cbar); PLk = pseudo_table(D0tr, U2, cbar_kept)
    # check vs the index the students trained on
    IX = P('%s/train/index_pseudo_srcx_%s.pkl' % (REAL, 'in2mav' if nm == 'PST' else 'in2mav_none'))
    ixz = {(IX['metas'][int(i)]['seq'], IX['metas'][int(i)]['frame']): float(IX['metas'][int(i)]['boxes9d'][0][2]) for i in IX['valid_idx']}
    chk = max(abs(np.log(ixz[k] / PL[k][2])) for k in ixz)
    assert set(ixz) == set(PL) and chk < 1e-5, (nm, chk)
    zfull = {k: z0[k] * np.exp(U2[k]) if (k in U2 and abs(U2[k]) <= GUARD) else z0[k] for k in keptk}
    # stage-3 refiner targets (fused) on the real d0 crops; the IR-only part = pseudo label itself
    meta_ = P(ch['X'] + '/ref/real_d0_meta.pkl'); dp = np.load(ch['X'] + '/ref/dp_fused.npy')
    zt = {}; gate_drop = []; gate_keep = []
    for i, m in enumerate(meta_):
        k = (m['seq'], m['frame'])
        if np.isfinite(dp[i]):
            zt[k] = m['z_det'] * np.exp(dp[i]); gate_keep.append(k)
        else:
            gate_drop.append(k)
    # recompute the gate to make sure the NaNs are the gate (det0 box vs pseudo box > 14.3 % of distance)
    regate = [k for k in keptk if np.linalg.norm(box0[k][:3] - PL[k][:3]) >= AGREE * np.linalg.norm(PL[k][:3])]
    assert set(regate) == set(gate_drop), (nm, len(regate), len(gate_drop))
    # students on train frames
    D1 = recs(ch['X'], 'det1')['recs']['train']; z1 = {(r['seq'], r['frame']): float(r['box'][2]) for r in D1 if r['box'] is not None}
    zl = zl_table(ch['W'] + '/oc_tr1.pkl', ch['X'] + '/ou/read_tr1_stu.pkl', tag='%s_tr1' % nm)
    zr, zf = deploy_on(ch, 'train', D1, zl)
    STG[nm] = dict(det0=z0, refiner_full=zfull, PL=PL and {k: v[2] for k, v in PL.items()}, PL_cbar_kept={k: v[2] for k, v in PLk.items()},
                   target3=zt, det1=z1, det1_ref=zr, det1_ref_fused=zf)
    rows = {}
    for st, Z in STG[nm].items():
        for setname, ks in (('kept (conf>=0.5)', keptk), ('all train dets', allk)):
            if st in ('PL', 'PL_cbar_kept', 'refiner_full', 'target3') and setname != 'kept (conf>=0.5)':
                continue
            ks2 = [k for k in ks if k in Z]
            rows['%s | %s' % (st, setname)] = relstats(err(Z, GTtr, ks2))
            rows['%s | %s | near GT (det0 <1 m)' % (st, setname)] = relstats(err(Z, GTtr, [k for k in ks2 if not far[k]]))
    # selection story: delta-hat on kept vs dropped (conf < 0.5) frames
    sel = dict(c_bar_all=cbar, c_bar_kept=cbar_kept, delta_med_kept=float(np.median(dk[np.abs(dk) <= GUARD])), delta_med_dropped=float(np.median(dd[np.abs(dd) <= GUARD])),
               retained_offset_on_kept=float(np.median(dk[np.abs(dk) <= GUARD]) - LAM * cbar), intended=float((1 - LAM) * cbar_kept))
    # agreement gate: errors of crops dropped vs kept
    gate = {}
    for gname, ks in (('dropped by 14.3% gate', gate_drop), ('kept by gate', gate_keep)):
        gate[gname] = dict(n=len(ks), det0=relstats(err(z0, GTtr, ks)), PL=relstats(err(STG[nm]['PL'], GTtr, ks)),
                           delta_hat_med=float(np.median([U2[k] for k in ks if k in U2])),
                           PL_better_than_det0=float(np.nanmean(np.abs(err(STG[nm]['PL'], GTtr, ks)) < np.abs(err(z0, GTtr, ks)))),
                           far_1m=float(np.mean([far[k] for k in ks])))
    # second gate (corner term needs det1 vs pseudo box agreement): crops with and without the fusion term
    tinfo = json.load(open(ch['X'] + '/targets_info.json'))
    A1['arms'][nm] = dict(stages=rows, selection=sel, gate=gate, targets_info=tinfo, index_check_max_abs_logz=float(chk))
    print(nm, json.dumps(sel), flush=True)
    for gname, g in gate.items():
        print('  gate', gname, g['n'], 'det0 %.3f/%+.3f' % (g['det0']['med_abs_rel'], g['det0']['med_log']), 'PL %.3f/%+.3f' % (g['PL']['med_abs_rel'], g['PL']['med_log']),
              'dhat %.3f PLbetter %.2f far %.3f' % (g['delta_hat_med'], g['PL_better_than_det0'], g['far_1m']), flush=True)
    for k, v in rows.items():
        if 'near GT' not in k:
            print('  %-50s n %5d  med|rel| %5.1f%%  med log %+6.1f%%  >25%% %.3f' % (k, v['n'], 100 * v['med_abs_rel'], 100 * v['med_log'], v['frac_gt25']), flush=True)
res['A1'] = A1

# ---- A1 mediator, train (in-sample): per-sequence paired bootstrap on the kept frames (identical frame set for both arms)
def med_abs(e):
    e = e[np.isfinite(e)]
    return float(np.median(np.abs(np.expm1(e)))) if len(e) else np.nan


def mediator(keys, EA, EB, seqs, label, n_boot=B):
    """EA/EB: dict name -> (errors arm PST, errors arm RGB) on keys. Pooled differences of median |rel err| (PST - RGB) with a paired sequence
    bootstrap; plus the across-sequence association of per-sequence differences (Pearson / Spearman / OLS slope) with bootstrap CIs."""
    us = np.unique(seqs); idx = {u: np.where(seqs == u)[0] for u in us}
    pt = {n: med_abs(EA[n]) - med_abs(EB[n]) for n in EA}
    per = {n: np.array([med_abs(EA[n][idx[u]]) - med_abs(EB[n][idx[u]]) for u in us]) for n in EA}
    acc = {n: [] for n in EA}; assoc = []
    nl = list(EA)
    for _ in range(n_boot):
        pick = rng.choice(us, len(us)); ii = np.concatenate([idx[u] for u in pick])
        for n in EA:
            acc[n].append(med_abs(EA[n][ii]) - med_abs(EB[n][ii]))
        a, b = per[nl[0]][np.searchsorted(us, pick)], per[nl[1]][np.searchsorted(us, pick)]
        ok = np.isfinite(a) & np.isfinite(b)
        assoc.append((np.corrcoef(a[ok], b[ok])[0, 1], np.polyfit(a[ok], b[ok], 1)[0]))
    a, b = per[nl[0]], per[nl[1]]; ok = np.isfinite(a) & np.isfinite(b); assoc = np.array(assoc)
    out = dict(label=label, n_seq=int(len(us)), n_frames=int(len(keys)),
               pooled={n: dict(point=pt[n], ci=[float(np.nanpercentile(acc[n], 2.5)), float(np.nanpercentile(acc[n], 97.5))],
                               frac_lt0=float(np.mean(np.array(acc[n]) < 0))) for n in EA},
               across_seq=dict(x=nl[0], y=nl[1], pearson=float(np.corrcoef(a[ok], b[ok])[0, 1]), spearman=float(spearmanr(a[ok], b[ok])[0]),
                               slope=float(np.polyfit(a[ok], b[ok], 1)[0]), pearson_ci=[float(np.nanpercentile(assoc[:, 0], 2.5)), float(np.nanpercentile(assoc[:, 0], 97.5))],
                               slope_ci=[float(np.nanpercentile(assoc[:, 1], 2.5)), float(np.nanpercentile(assoc[:, 1], 97.5))],
                               seqs_both_better=int(np.sum((a[ok] < 0) & (b[ok] < 0))), seqs_pl_better=int(np.sum(a[ok] < 0)), seqs_stu_better=int(np.sum(b[ok] < 0)), n=int(ok.sum())))
    return out


kk = [k for k in keptk if k in GTtr and np.isfinite(GTtr[k][2])]
sq = np.array([k[0] for k in kk])
MED = {}
for stu in ('det1', 'det1_ref', 'det1_ref_fused'):
    MED['train: PL vs student %s' % stu] = mediator(kk, {'PL': err(STG['PST']['PL'], GTtr, kk), stu: err(STG['PST'][stu], GTtr, kk)},
                                                    {'PL': err(STG['RGB-only']['PL'], GTtr, kk), stu: err(STG['RGB-only'][stu], GTtr, kk)}, sq,
                                                    'train kept frames (in-sample): PST - RGB-only median |rel err|')
# frame-level association on train (sequence-demeaned): does a frame whose PST label is better than its RGB-only label also get a better PST student?
def dm(x, s):
    y = x.copy()
    for u in np.unique(s):
        m = s == u; y[m] = x[m] - np.nanmean(x[m])
    return y


for stu in ('det1', 'det1_ref_fused'):
    a = np.abs(err(STG['PST']['PL'], GTtr, kk)) - np.abs(err(STG['RGB-only']['PL'], GTtr, kk))
    b = np.abs(err(STG['PST'][stu], GTtr, kk)) - np.abs(err(STG['RGB-only'][stu], GTtr, kk))
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 0.5) & (np.abs(b) < 0.5)
    MED['train frame-level %s' % stu] = dict(n=int(ok.sum()), spearman=float(spearmanr(a[ok], b[ok])[0]), pearson_demeaned=float(np.corrcoef(dm(a[ok], sq[ok]), dm(b[ok], sq[ok]))[0, 1]),
                                             slope_demeaned=float(np.polyfit(dm(a[ok], sq[ok]), dm(b[ok], sq[ok]), 1)[0]))

# ---- val: student side (held out), PST label side (N1clean held-out PL reconstruction). RGB-only val labels need its stage-2 refiner on val: not available.
GTva = gt_index('val'); D0va = D0['recs']['val']
Uva = avg([P('%s/N1clean/st2_s%d_det0_val.pkl' % (R13, j)) for j in (5, 6)])
PLva = {k: v[2] for k, v in pseudo_table(D0va, Uva, A1['arms']['PST']['selection']['c_bar_all']).items()}
kv = [k for k in PLva if k in GTva and np.isfinite(GTva[k][2])]; sqv = np.array([k[0] for k in kv])
VAL = {}
for nm, ch in CH.items():
    D1 = recs(ch['X'], 'det1')['recs']['val']; z1 = {(r['seq'], r['frame']): float(r['box'][2]) for r in D1 if r['box'] is not None}
    zl = zl_table(ch['W'] + '/oc_val1.pkl', ch['X'] + '/ou/read_val1_stu.pkl', tag='%s_val1' % nm)
    zr, zf = deploy_on(ch, 'val', D1, zl)
    # check the deployed function against the chain's own fused val predictions
    ref = {(r['seq'], r['frame']): float(np.asarray(r['boxes9d']).reshape(-1, 9)[0, 2]) for r in P(ch['X'] + '/preds/pcmd_fused_val.pkl')}
    VAL[nm] = dict(det1=z1, det1_ref=zr, det1_ref_fused=zf, chk=float(max(abs(np.log(zf[k] / ref[k])) for k in ref)))
    assert VAL[nm]['chk'] < 1e-6, (nm, VAL[nm]['chk'])
MED['val (held out): students only, PL-conf frames'] = mediator(kv, {'det1': err(VAL['PST']['det1'], GTva, kv), 'det1_ref_fused': err(VAL['PST']['det1_ref_fused'], GTva, kv)},
                                                                 {'det1': err(VAL['RGB-only']['det1'], GTva, kv), 'det1_ref_fused': err(VAL['RGB-only']['det1_ref_fused'], GTva, kv)},
                                                                 sqv, 'val det0-conf>=0.5 frames: PST - RGB-only student error (no RGB-only val labels)')
MED['val PST label quality'] = dict(PL=relstats(err(PLva, GTva, kv)), det0=relstats(err({k: float(r['box'][2]) for r in D0va if r['box'] is not None for k in [(r['seq'], r['frame'])]}, GTva, kv)))
res['mediator'] = MED
for k, v in MED.items():
    print(k, json.dumps(v, default=lambda x: round(float(x), 4))[:900], flush=True)

# ------------------------------------------------------------------ A8: copy analysis on held-out val WITHOUT the 1 m filter
z0v = {(r['seq'], r['frame']): float(r['box'][2]) for r in D0va if r['box'] is not None}
air = np.array([k[0].split('/')[0] for k in kv])


def copy_stats(sel, filt_1m=False):
    ks = [k for k, s in zip(kv, sel) if s]
    S_ = VAL['PST']['det1']
    rows = []
    for k in ks:
        if k not in S_:
            continue
        g = GTva[k]
        if filt_1m:
            b1 = [r for r in recs(X, 'det1')['recs']['val'] if False]  # placeholder (filter applied below via positions)
        rows.append((np.log(S_[k] / g[2]), np.log(PLva[k] / g[2]), np.log(z0v[k] / g[2]), k[0]))
    es, ep, e0, s = map(np.array, zip(*rows)); es, ep, e0 = es.astype(float), ep.astype(float), e0.astype(float)
    return es, ep, e0, s


# positions for the optional 1 m filter (to reproduce N1clean)
P0 = {(r['seq'], r['frame']): np.asarray(r['box'], np.float64) for r in D0va if r['box'] is not None}
P1 = {(r['seq'], r['frame']): np.asarray(r['box'], np.float64) for r in recs(X, 'det1')['recs']['val'] if r['box'] is not None}


def pl_pos(k):
    return P0[k][:3] * (PLva[k] / P0[k][2])


def boot_ci(fn, es, ep, e0, s, nb=B):
    us = np.unique(s); idx = {u: np.where(s == u)[0] for u in us}; v = []
    for _ in range(nb):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        v.append(fn(es[ii], ep[ii], e0[ii], s[ii]))
    v = np.array(v)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if v.ndim == 1 else [[float(np.nanpercentile(v[:, j], q)) for q in (2.5, 97.5)] for j in range(v.shape[1])]


F = {
    'gap student-PL (med |rel|)': lambda es, ep, e0, s: float(np.median(np.abs(np.expm1(es - ep)))),
    'gap student-det0 (med |rel|)': lambda es, ep, e0, s: float(np.median(np.abs(np.expm1(es - e0)))),
    'student-GT (med |rel|)': lambda es, ep, e0, s: float(np.median(np.abs(np.expm1(es)))),
    'PL-GT (med |rel|)': lambda es, ep, e0, s: float(np.median(np.abs(np.expm1(ep)))),
    'det0-GT (med |rel|)': lambda es, ep, e0, s: float(np.median(np.abs(np.expm1(e0)))),
    'bias student / PL / det0 (med log)': lambda es, ep, e0, s: [float(np.median(es)), float(np.median(ep)), float(np.median(e0))],
    'OLS slope student on PL': lambda es, ep, e0, s: float(np.polyfit(ep, es, 1)[0]),
    'OLS slope student on det0': lambda es, ep, e0, s: float(np.polyfit(e0, es, 1)[0]),
    'Theil-Sen slope student on PL': lambda es, ep, e0, s: float(theilslopes(es, ep)[0]),
    'Theil-Sen slope student on det0': lambda es, ep, e0, s: float(theilslopes(es, e0)[0]),
    'multiple OLS student ~ PL + det0 (coef PL, coef det0)': lambda es, ep, e0, s: np.linalg.lstsq(np.c_[ep, e0, np.ones_like(ep)], es, rcond=None)[0][:2].tolist(),
    'Pearson student~PL': lambda es, ep, e0, s: float(np.corrcoef(es, ep)[0, 1]),
    'Spearman student~PL': lambda es, ep, e0, s: float(spearmanr(es, ep)[0]),
    'Spearman student~det0': lambda es, ep, e0, s: float(spearmanr(es, e0)[0]),
}
A8 = {}
for filt in (False, True):
    for grp in ('all', 'phantom4', 'mavic2'):
        sel = np.ones(len(kv), bool) if grp == 'all' else (air == grp)
        if filt:
            sel = sel & np.array([k in P1 and np.linalg.norm(P1[k][:3] - GTva[k][:3]) < 1 and np.linalg.norm(pl_pos(k) - GTva[k][:3]) < 1 for k in kv])
        es, ep, e0, s = copy_stats(sel)
        out = dict(n=int(len(es)), n_seq=int(len(np.unique(s))), n_far_1m=int(sum(1 for k, q in zip(kv, sel) if q and k in P1 and np.linalg.norm(P1[k][:3] - GTva[k][:3]) >= 1)))
        for nm_, fn in F.items():
            pt = fn(es, ep, e0, s)
            ci = boot_ci(fn, es, ep, e0, s, nb=500 if 'Theil' in nm_ else B) if out['n_seq'] >= 3 else None
            out[nm_] = dict(point=pt, ci=ci)
        A8['%s | %s' % ('1 m filter (N1clean)' if filt else 'no filter', grp)] = out
        print('A8', filt, grp, out['n'], out['n_seq'], out['n_far_1m'], {k: v['point'] for k, v in out.items() if isinstance(v, dict)}, flush=True)
res['A8'] = A8
json.dump(res, open(PA + '/a1_a8_labels.json', 'w', encoding='utf-8'), indent=1, default=float)
print('done')
