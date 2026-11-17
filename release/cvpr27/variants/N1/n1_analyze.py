# -*- coding: utf-8 -*-
"""N1: does the self-trained detector (student) copy its pseudo labels frame by frame?
Held-out sequences (MAV6D val, never trained on) vs training frames. GT is used for diagnosis only; nothing is selected with it.
Error = e = log(z / z_gt) (the pseudo-label correction scales the whole translation, so z ratio = distance ratio).
Frame set: teacher top-confidence detection conf >= 0.5 (same rule as the pseudo labels), and both teacher and student
top-confidence detections within 1 m of GT (same drone, no false positive; same as the greedy 1 m protocol)."""
import os, sys, json, pickle
import numpy as np
from scipy.stats import spearmanr
ROOT = 'E:/Open3DUAVDet'; G = ROOT + '/output/camnorm/geo1004'; N1 = G + '/gproto/review1113/N1'
sys.path.insert(0, ROOT); sys.path.insert(0, G)
from st2combo_analyze import causal_global, GUARD
rng = np.random.default_rng(0)


def top(pred, conf):
    pred = np.asarray(pred, np.float64).reshape(-1, 9); conf = np.asarray(conf, np.float64).reshape(-1)
    if len(conf) == 0:
        return None, -1.0, -1
    j = int(np.argmax(conf)); return pred[j], float(conf[j]), j


def pear(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def clipc(a, b):   # same as old check_r2 metric: subtract median, clip to +-0.4
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
    if len(qs) >= 5:   # sequence-level cluster bootstrap
        B1, B2 = [], []
        for _ in range(2000):
            pick = rng.choice(qs, len(qs))
            a = np.concatenate([es[idx[q]] for q in pick]); b = np.concatenate([ep[idx[q]] for q in pick])
            lab = np.concatenate([np.full(len(idx[q]), t) for t, q in enumerate(pick)])
            B1.append(pear(a, b)); B2.append(pear(dm(a, lab), dm(b, lab)))
        r['pearson_ci'] = [float(np.percentile(B1, 2.5)), float(np.percentile(B1, 97.5))]
        r['pearson_demean_ci'] = [float(np.percentile(B2, 2.5)), float(np.percentile(B2, 97.5))]
    return r


def collect(T, S, GT, conf_min=0.5):
    keys = [k for k in T if T[k][1] >= conf_min and k in GT]
    c = dict(teacher_conf_ok=len(keys))
    rows = []
    for k in keys:
        g = GT[k]; t = T[k][0]; s = S.get(k, (None, -1))[0]
        if s is None or not np.isfinite(g[2]):
            continue
        if np.linalg.norm(t[:3] - g[:3]) >= 1.0 or np.linalg.norm(s[:3] - g[:3]) >= 1.0:
            continue
        rows.append((np.log(s[2] / g[2]), np.log(t[2] / g[2]), k[0]))
    c['both_within_1m'] = len(rows)
    es, ep, sq = map(np.array, zip(*rows))
    return es.astype(float), ep.astype(float), sq.astype(str), c


R = {}
# ---------------- final label-free chain nl075 ----------------
D = pickle.load(open(ROOT + '/output/camnorm/st/real_train_preds_st2nb.pkl', 'rb'))['frames']
DI = pickle.load(open(G + '/st2m/delta_IR_nb.pkl', 'rb'))['train']
d0 = np.array([np.asarray(e['d']).reshape(-1)[0] if len(np.asarray(e['d']).reshape(-1)) else np.nan for e in DI])
CG_NL = float(np.median(d0[np.isfinite(d0) & (np.abs(d0) <= GUARD)]))   # train constant, exactly as make_nolabel_pseudo.py
LAM = 0.75
R['cg_train'] = CG_NL


def pl_box(f, d):
    p, cf, j = top(f['pred'], f['conf'])
    if p is None:
        return None, -1.0
    p = p.copy(); dd = np.asarray(d).reshape(-1); dj = dd[j] if len(dd) else np.nan
    if np.isfinite(dj) and abs(dj) <= GUARD:
        p[:3] *= np.exp(dj - LAM * CG_NL)
    return p, cf


Ttr, GTtr = {}, {}
for f, e in zip(D, DI):
    k = (f['seq'], f['frame']); Ttr[k] = pl_box(f, e['d']); GTtr[k] = np.asarray(f['gt'], np.float64)
IX = pickle.load(open('E:/mmcache/mav6d_cn_obj/train/index_pseudo_nl075.pkl', 'rb'))
dz = []
for i in IX['valid_idx']:
    m = IX['metas'][int(i)]; k = (m['seq'], m['frame']); dz.append(abs(np.log(float(m['boxes9d'][0][2]) / Ttr[k][0][2])))
R['check_index_pseudo_nl075'] = dict(n=len(dz), max_abs_logz_diff=float(np.max(dz)))
Str = {(r['seq_id'], r['frame_id']): top(r['pred'], r['conf'])[:2] for r in pickle.load(open(G + '/gproto/nolabel/nl075/recs/nl075_train.pkl', 'rb'))['records']}
es, ep, sq, c = collect(Ttr, Str, GTtr)
R['nl075_train'] = dict(stats(es, ep, sq, 'final detector nl075 (3-seed ens) / train frames'), counts=c)

V = pickle.load(open(N1 + '/real_val_preds_st2nb.pkl', 'rb'))['frames']
VD = pickle.load(open(N1 + '/val_deltas.pkl', 'rb'))
assert [(f['seq'], f['frame']) for f in V] == VD['nb']['keys']
Tva, GTva = {}, {}
for f, e in zip(V, VD['nb']['unc']):
    k = (f['seq'], f['frame']); Tva[k] = pl_box(f, e['d']); GTva[k] = np.asarray(f['gt'], np.float64)
Sva = {(r['seq_id'], r['frame_id']): top(r['pred'], r['conf'])[:2] for r in pickle.load(open(G + '/nl075_ens/ens_val.pkl', 'rb'))['records']}
es, ep, sq, c = collect(Tva, Sva, GTva)
R['nl075_val'] = dict(stats(es, ep, sq, 'final detector nl075 (3-seed ens) / val held-out sequences'), counts=c)
# controls: shared difficulty
T0 = {(f['seq'], f['frame']): top(f['pred'], f['conf'])[:2] for f in V}
T0tr = {(f['seq'], f['frame']): top(f['pred'], f['conf'])[:2] for f in D}
for nm, TT, SS, GG, lab in (('nl075_val_ctrl_ST2raw_vs_pl', Tva, T0, GTva, 'control: teacher raw detector (ST2) vs pseudo label / val'),
                            ('nl075_val_ctrl_student_vs_ST2raw', T0, Sva, GTva, 'control: student vs teacher raw detector (ST2) / val'),
                            ('nl075_train_ctrl_student_vs_ST2raw', T0tr, Str, GTtr, 'control: student vs teacher raw detector (ST2) / train')):
    a, b, s_, cc_ = collect(TT, SS, GG)
    R[nm] = dict(stats(a, b, s_, lab), counts=cc_)

# ---------------- 11-07 wa (pseudo labels st2plus) ----------------
FV = pickle.load(open(G + '/final/ens_val.pkl', 'rb'))['records']
assert [(r['seq_id'], r['frame_id']) for r in FV] == VD['bg']['keys']
recs = [dict(seq_id=r['seq_id'], frame_id=r['frame_id'], pred=np.asarray(r['pred'], np.float64).reshape(-1, 9), conf=np.asarray(r['conf']).reshape(-1)) for r in FV]
pr = [(np.asarray(e['d']).reshape(-1), np.asarray(e['lb']).reshape(-1)) for e in VD['bg']['unc']]
dv = np.concatenate([x[0] for x in pr]); c0v = float(np.median(dv[np.isfinite(dv) & (np.abs(dv) <= GUARD)]))
cc = causal_global(recs, pr, c0v)
Twv = {}
for r, (d, _), ci in zip(recs, pr, cc):
    p, cf, j = top(r['pred'], r['conf'])
    if p is None:
        Twv[(r['seq_id'], r['frame_id'])] = (None, -1.0); continue
    p = p.copy()
    if np.isfinite(d[j]) and abs(d[j]) <= GUARD:
        p[:3] *= np.exp(d[j] - 0.75 * ci[j])
    Twv[(r['seq_id'], r['frame_id'])] = (p, cf)
GTv2 = {(r['seq_id'], r['frame_id']): np.asarray(r['gt'], np.float64).reshape(-1, 9)[0] for r in FV}
for nm, path, lab in (('wa_val', '/wa_ens/ens_val.pkl', '11-07 wa (3-seed ens, temporal bg) / val held-out sequences'),
                      ('wa_val_nobg', '/wanb_ens/ens_val.pkl', '11-07 wa (3-seed ens, no temporal bg) / val held-out sequences')):
    SS = {(r['seq_id'], r['frame_id']): top(r['pred'], r['conf'])[:2] for r in pickle.load(open(G + path, 'rb'))['records']}
    a, b, s_, cc_ = collect(Twv, SS, GTv2)
    R[nm] = dict(stats(a, b, s_, lab), counts=cc_, c0_val=c0v)

# train frames: reproduce diag_trainfit.py (per-seed readout W_k_z vs pseudo-label box z, same 11,910 frames)
A = dict(np.load(G + '/gproto/real_feats_train.npz', allow_pickle=True)); B = dict(np.load(G + '/gproto/real_feats_train_wa.npz', allow_pickle=True))
I = pickle.load(open('E:/mmcache/mav6d_cn_obj/train/index.pkl', 'rb'))
gt = {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64)[0] for m in I['metas'] if m}
zg = np.array([gt.get((s, f), np.full(9, np.nan))[2] for s, f in zip(A['seq'], A['frame'])])
zp = A['box'][:, 2].astype(np.float64); sqA = np.array([str(s) for s in A['seq']])
ok = np.isfinite(zg); ep = np.log(zp / zg)[ok]
for k in range(3):
    es = np.log(B['W%d_z' % k].astype(np.float64) / zg)[ok]; m = np.isfinite(es)
    R['wa_train_seed%d' % k] = stats(es[m], ep[m], sqA[ok][m], '11-07 wa seed %d / train frames (diag_trainfit repro)' % k)
esm = np.mean([np.log(B['W%d_z' % k].astype(np.float64) / zg) for k in range(3)], 0)[ok]; m = np.isfinite(esm)
R['wa_train_mean3'] = stats(esm[m], ep[m], sqA[ok][m], '11-07 wa 3-seed log-mean / train frames')

# scene overlap between val sequences and training sequences (seq id = type/scene/seq)
trs = {f['seq'] for f in D}; vas = {f['seq'] for f in V}
R['seq_overlap'] = dict(n_train_seq=len(trs), n_val_seq=len(vas), shared_seq=len(trs & vas),
                        val_scenes=sorted({'/'.join(s.split('/')[:2]) for s in vas}),
                        val_scenes_also_in_train=sorted({'/'.join(s.split('/')[:2]) for s in vas} & {'/'.join(s.split('/')[:2]) for s in trs}))
json.dump(R, open(N1 + '/n1_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in R.items():
    if isinstance(v, dict) and 'pearson' in v:
        print('%-36s n=%5d seq=%3d r=%.3f [%s] clip=%.3f sp=%.3f | demean r=%.3f [%s] clip=%.3f | |S-GT| %.1f%% |PL-GT| %.1f%% |S-PL| %.1f%% | bias S %+.1f PL %+.1f | btw-seq var PL %.2f S %.2f' % (
            k, v['n'], v['n_seq'], v['pearson'], ','.join('%.2f' % x for x in v.get('pearson_ci', [])), v['pearson_clip'], v['spearman'], v['pearson_demean'],
            ','.join('%.2f' % x for x in v.get('pearson_demean_ci', [])), v['pearson_demean_clip'], 100 * v['med_rel_student_gt'], 100 * v['med_rel_pl_gt'],
            100 * v['med_rel_student_pl'], 100 * v['bias_student'], 100 * v['bias_pl'], v['frac_between_seq_var_pl'], v['frac_between_seq_var_student']))
        if 'counts' in v:
            print('    counts', v['counts'])
    else:
        print(k, v)
