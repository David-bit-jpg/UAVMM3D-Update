# -*- coding: utf-8 -*-
"""N1clean: does the self-trained detector (det1) copy its pseudo labels on the CLEAN MAV6D chain (srcx/in2mav)?
Pseudo label = srcx/lib/make_pseudo_x.py rule: det0 box (conf >= 0.5), t *= exp(delta_hat - 0.75 * c_bar) if |delta_hat| <= log 1.4,
delta_hat = mean of refiner stage-2 seeds 5, 6 (avg_inf.py), c_bar = TRAIN constant (median delta_hat over all train crops, |.| <= log 1.4).
Val: held-out reconstruction (det0 val recs + new stage-2 val inference, train c_bar). Train: reconstruction checked against index_pseudo_srcx_in2mav.pkl.
Error e = log(z / z_gt). Frames: det0 conf >= 0.5 and both PL and student boxes within 1 m of GT (GT used only to score).
Same statistics code as review1113/N1/n1_analyze.py."""
import sys, json, pickle
import numpy as np
from scipy.stats import spearmanr
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav'; N = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1clean'
REAL = 'E:/mmcache/mav6d_cn_obj'
LAM, CONF, GUARD = 0.75, 0.5, float(np.log(1.4))
rng = np.random.default_rng(0)


def pear(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def clipc(a, b):
    a = np.clip(a - np.median(a), -.4, .4); b = np.clip(b - np.median(b), -.4, .4); return pear(a, b)


def dm(x, seq):
    y = x.copy()
    for q in np.unique(seq):
        m = seq == q; y[m] = x[m] - x[m].mean()
    return y


def stats(es, ep, seq, name):
    r = dict(name=name, n=int(len(es)), n_seq=int(len(np.unique(seq))),
             pearson=pear(es, ep), pearson_clip=clipc(es, ep), spearman=float(spearmanr(es, ep)[0]),
             pearson_demean=pear(dm(es, seq), dm(ep, seq)), pearson_demean_clip=clipc(dm(es, seq), dm(ep, seq)),
             med_rel_student_gt=float(np.median(np.abs(np.exp(es) - 1))),
             med_rel_pl_gt=float(np.median(np.abs(np.exp(ep) - 1))),
             med_rel_student_pl=float(np.median(np.abs(np.exp(es - ep) - 1))),
             bias_student=float(np.median(es)), bias_pl=float(np.median(ep)),
             frac_between_seq_var_pl=float(1 - np.var(dm(ep, seq)) / np.var(ep)),
             frac_between_seq_var_student=float(1 - np.var(dm(es, seq)) / np.var(es)))
    qs = np.unique(seq); idx = {q: np.where(seq == q)[0] for q in qs}
    if len(qs) >= 5:
        B1, B2 = [], []
        for _ in range(2000):
            pick = rng.choice(qs, len(qs))
            a = np.concatenate([es[idx[q]] for q in pick]); b = np.concatenate([ep[idx[q]] for q in pick])
            lab = np.concatenate([np.full(len(idx[q]), t) for t, q in enumerate(pick)])
            B1.append(pear(a, b)); B2.append(pear(dm(a, lab), dm(b, lab)))
        r['pearson_ci'] = [float(np.percentile(B1, 2.5)), float(np.percentile(B1, 97.5))]
        r['pearson_demean_ci'] = [float(np.percentile(B2, 2.5)), float(np.percentile(B2, 97.5))]
    return r


def avg(U):   # = srcx/lib/avg_inf.py
    R = {}
    for k in U[0]:
        if not all(k in u for u in U):
            continue
        R[k] = float(np.mean([u[k]['d'] for u in U]))
    return R


def pl_table(recs, dh, cbar):
    T = {}; ncor = 0
    for r in recs:
        if r['box'] is None or r['conf'] < CONF:
            continue
        p = np.asarray(r['box'], np.float64).copy(); d = dh.get((r['seq'], r['frame']))
        if d is not None and abs(d) <= GUARD:
            p[:3] *= np.exp(d - LAM * cbar); ncor += 1
        T[(r['seq'], r['frame'])] = p
    return T, ncor


def gt_table(split):
    I = pickle.load(open('%s/%s/index.pkl' % (REAL, split), 'rb'))
    return {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64)[0] for m in I['metas'] if m is not None and len(m['boxes9d'])}


def collect(T, S, GT):
    c = dict(teacher_conf_ok=len(T)); rows = []
    for k, t in T.items():
        if k not in GT or k not in S or S[k] is None:
            continue
        g = GT[k]; s = S[k]
        if not np.isfinite(g[2]) or np.linalg.norm(t[:3] - g[:3]) >= 1.0 or np.linalg.norm(s[:3] - g[:3]) >= 1.0:
            continue
        rows.append((np.log(s[2] / g[2]), np.log(t[2] / g[2]), k[0]))
    c['both_within_1m'] = len(rows)
    es, ep, sq = map(np.array, zip(*rows))
    return es.astype(float), ep.astype(float), sq.astype(str), c


R = {}
D0 = pickle.load(open(X + '/recs/det0.pkl', 'rb'))['recs']; D1 = pickle.load(open(X + '/recs/det1.pkl', 'rb'))['recs']
# ---- train constant c_bar (exactly make_pseudo_x.py, from the chain's own averaged stage-2 train inference)
Utr = pickle.load(open(X + '/ref/inf_st2_det0_train.pkl', 'rb'))
dtr = np.array([Utr[(r['seq'], r['frame'])]['d'] for r in D0['train'] if (r['seq'], r['frame']) in Utr])
cbar = float(np.median(dtr[np.abs(dtr) <= GUARD])); R['c_bar_train'] = cbar
Ttr, ncor_tr = pl_table(D0['train'], {k: v['d'] for k, v in Utr.items()}, cbar)
# ---- reproduction check vs the index the students actually trained on
IX = pickle.load(open(REAL + '/train/index_pseudo_srcx_in2mav.pkl', 'rb'))
ixk = {(IX['metas'][int(i)]['seq'], IX['metas'][int(i)]['frame']): np.asarray(IX['metas'][int(i)]['boxes9d'], np.float64)[0] for i in IX['valid_idx']}
dz = [abs(np.log(ixk[k][2] / Ttr[k][2])) for k in ixk if k in Ttr]
dt = [np.max(np.abs(ixk[k][:3] - Ttr[k][:3])) for k in ixk if k in Ttr]
R['check_index_pseudo_srcx_in2mav'] = dict(n_index=len(ixk), n_recon=len(Ttr), same_keys=set(ixk) == set(Ttr), n_corrected=ncor_tr,
                                           max_abs_logz_diff=float(np.max(dz)), max_abs_t_diff_m=float(np.max(dt)), index_c_bar=IX['pseudo']['c_bar'])
# ---- batch-4 re-inference reproduces the chain's stage-2 train inference
chk = {}
for j in (5, 6):
    A = pickle.load(open('%s/chk_s%d_train300.pkl' % (N, j), 'rb')); B = pickle.load(open('%s/ref/st2_s%d_det0_train.pkl' % (X, j), 'rb'))
    chk['s%d' % j] = dict(n=len(A), max_abs_d_diff=float(max(abs(A[k]['d'] - B[k]['d']) for k in A)), keys_subset=all(k in B for k in A))
R['check_reinfer_train300'] = chk
# ---- val: held-out reconstruction
Uva = avg([pickle.load(open('%s/st2_s%d_det0_val.pkl' % (N, j), 'rb')) for j in (5, 6)])
Tva, ncor_va = pl_table(D0['val'], Uva, cbar)
dva = np.array(list(Uva.values()))
R['val_refiner'] = dict(n_crops=len(Uva), n_val=len(D0['val']), n_pl=len(Tva), n_corrected=ncor_va,
                        delta_median_val=float(np.median(dva[np.abs(dva) <= GUARD])), frac_in_guard=float(np.mean(np.abs(dva) <= GUARD)))
GTtr, GTva = gt_table('train'), gt_table('val')
S1 = {sp: {(r['seq'], r['frame']): (None if r['box'] is None else np.asarray(r['box'], np.float64)) for r in D1[sp]} for sp in ('train', 'val')}
S0 = {sp: {(r['seq'], r['frame']): (None if r['box'] is None else np.asarray(r['box'], np.float64)) for r in D0[sp]} for sp in ('train', 'val')}
for nm, T, S, GT, lab in (('det1_val', Tva, S1['val'], GTva, 'student det1 vs pseudo label / val held-out sequences'),
                          ('det1_train', Ttr, S1['train'], GTtr, 'student det1 vs pseudo label / train (fitted) frames'),
                          ('val_ctrl_student_vs_det0raw', {k: S0['val'][k] for k in Tva}, S1['val'], GTva, 'control: student vs raw det0 (before refiner) / val'),
                          ('val_ctrl_det0raw_vs_pl', Tva, S0['val'], GTva, 'control: raw det0 vs pseudo label / val'),
                          ('train_ctrl_student_vs_det0raw', {k: S0['train'][k] for k in Ttr}, S1['train'], GTtr, 'control: student vs raw det0 / train')):
    a, b, s_, cc = collect(T, S, GT)
    R[nm] = dict(stats(a, b, s_, lab), counts=cc)
trs = {r['seq'] for r in D0['train']}; vas = {r['seq'] for r in D0['val']}
R['seq_overlap'] = dict(n_train_seq=len(trs), n_val_seq=len(vas), shared_seq=len(trs & vas))
json.dump(R, open(N + '/n1c_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in R.items():
    if isinstance(v, dict) and 'pearson' in v:
        print('%-32s n=%5d seq=%3d r=%.3f [%s] sp=%.3f | demean r=%.3f [%s] | |S-GT| %.1f%% |PL-GT| %.1f%% |S-PL| %.1f%% | bias S %+.1f PL %+.1f | btw-seq var PL %.2f' % (
            k, v['n'], v['n_seq'], v['pearson'], ','.join('%.3f' % x for x in v.get('pearson_ci', [])), v['spearman'], v['pearson_demean'],
            ','.join('%.3f' % x for x in v.get('pearson_demean_ci', [])), 100 * v['med_rel_student_gt'], 100 * v['med_rel_pl_gt'],
            100 * v['med_rel_student_pl'], 100 * v['bias_student'], 100 * v['bias_pl'], v['frac_between_seq_var_pl']), v['counts'])
    else:
        print(k, v)
