# -*- coding: utf-8 -*-
"""[clean item 3] Height-source swap on the clean-protocol MAV6D chain (srcx/in2mav, test UDS 60.81). CPU only, reuses the chain's predictions.
Only h changes: h := r * mean(l, w); l, w, position, rotation, confidence unchanged (= review1113/height variant (a)). In the srcx chain the
corner PnP uses the s_bar box (oc_*.pkl 'size'), never r_k, so (a) is the whole effect of the height source.
Arms: PST r_k (deployed: per brightness class, median of height-network readings on unlabeled train; class -1 = per-frame reading)
      vs constant aspect of the chain's own s_bar  vs  sim-box constant 0.400 (A3 (a): mean h/((l+w)/2) over all indoor-sim train boxes, 0.3999).
Per-frame quantities / AP / UDS by A3/b_swap.frame_q + stats (vectorised, asserted equal to cmpeval/paper_table.metrics = pack real_test).
Sequence bootstrap of differences (alternative - r_k), B = 2000, seed 0; same resampled sequences for both arms. Labels read only for scoring."""
import os, sys, json, pickle, importlib.util
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
G = 'E:/Open3DUAVDet/output/camnorm/geo1004'; X = G + '/gproto/srcx/in2mav'
spec = importlib.util.spec_from_file_location('b_swap', G + '/gproto/review1113/A3/b_swap.py')
BS = importlib.util.module_from_spec(spec); spec.loader.exec_module(BS)
PT = BS.PT
DEP = json.load(open(X + '/deploy_pcmd.json')); SB = np.asarray(DEP['s_bar']); RK = DEP['r_k']
C_SBAR = float(SB[2] / SB[:2].mean())
C_SIM = float(json.load(open(G + '/gproto/review1113/A3/b_sim_aspect.json'))['a_per_object_mean'])
ALT = {'s_bar constant aspect': C_SBAR, 'sim-box constant': C_SIM}
B_BOOT = 2000


def load(fn, M):
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; N = len(M['seq'])
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(fn, 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores']).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


if __name__ == '__main__':
    res = dict(r_k=RK, s_bar=SB.tolist(), alternatives=ALT, splits={})
    for sp in ('val', 'test'):
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); valid = np.where(M['n_gt'] > 0)[0]
        gt = M['gt']; seq = np.array([str(s) for s in M['seq']])
        typ = np.where(np.isclose(gt[:, 5], 0.12, atol=0.01), 'Mavic 2', 'Phantom 4')
        R = load('%s/preds/pcmd_fused_%s.pkl' % (X, sp), M)
        ratio = R['plwh'][:, 2] / R['plwh'][:, :2].mean(1); fin = np.isfinite(ratio)
        cls = np.where(np.isclose(ratio, RK['0'], atol=1e-4), 0, np.where(np.isclose(ratio, RK['1'], atol=1e-4), 1, -1))
        V = {'PST: LiDAR height ratio r_k': R}
        for nm, r in ALT.items():
            A = {k: v.copy() for k, v in R.items()}; A['plwh'][:, 2] = r * A['plwh'][:, :2].mean(1); V[nm] = A
        vm = np.zeros(len(gt), bool); vm[valid] = True
        groups = {'all': vm, 'Mavic 2': vm & (typ == 'Mavic 2'), 'Phantom 4': vm & (typ == 'Phantom 4')}
        FF, rows = {}, {}
        for nm, P in V.items():
            m = PT.metrics(P, M, valid); F = BS.frame_q(P, M, valid); FF[nm] = F
            chk = BS.stats(F, valid); assert abs(chk['UDS'] - m['UDS']) < 1e-9 and abs(chk['ASE'] - m['ASE']) < 1e-9, (nm, chk, m)
            row = dict(UDS=m['UDS'], AP=m['AP'], ATE=m['ATE'], ASE=m['ASE'], AOE=m['AOE'], mDR=m['mDR'])
            for gname, gm in groups.items():
                ix = np.where(gm)[0]; s = BS.stats(F, ix)
                row[gname] = dict(ASE=s['ASE'], dh=s['dh'], dhr=s['dhr'], n_tp=s['n_tp'], UDS_sub=s['UDS'])
            rows[nm] = row
        rng = np.random.default_rng(0); boot = {}
        for gname, gm in groups.items():
            us = np.unique(seq[gm]); idx = {u: np.where(gm & (seq == u))[0] for u in us}
            acc = {nm: {k: [] for k in ('ASE', 'dh', 'dhr', 'UDS')} for nm in ALT}
            for _ in range(B_BOOT):
                ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
                s0 = BS.stats(FF['PST: LiDAR height ratio r_k'], ii)
                for nm in ALT:
                    s1 = BS.stats(FF[nm], ii)
                    for k in acc[nm]:
                        acc[nm][k].append(s1[k] - s0[k])
            boot[gname] = {nm: {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in a.items()} for nm, a in acc.items()}
            boot[gname]['n_seq'] = int(len(us))
        res['splits'][sp] = dict(rows=rows, boot=boot, n_class={int(c): int(((cls == c) & vm & fin).sum()) for c in (0, 1, -1)},
                                 class_by_type={t: {int(c): float(np.mean(cls[vm & (typ == t) & fin] == c)) for c in (0, 1, -1)} for t in ('Mavic 2', 'Phantom 4')})
        print(sp, json.dumps(res['splits'][sp], default=lambda x: round(float(x), 3), ensure_ascii=False)[:2500], flush=True)
    json.dump(res, open(OUT + '/height_clean.json', 'w', encoding='utf-8'), indent=1, default=float, ensure_ascii=False)
