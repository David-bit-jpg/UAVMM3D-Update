# -*- coding: utf-8 -*-
"""[corrupt_1117] tables + paired bootstrap (CPU only, existing prediction pickles only).
  python boot.py antifs|mav
Point metrics = pack scorer D:/UAV3D_baselines_pack/scripts/score.py metrics() on the target's sealed test meta (every cell is also checked
against the run's own score json). Paired difference PST - arm per condition, same resample for both:
  antifs (RealDrone3D): circular moving blocks of L = 176 test frames within each sequence (frames ordered by frame number), B = 2000
  mav (MAV6D): resample whole sequences with replacement, B = 2000
95% percentile interval. Per-frame quantities use the pack's own box2d / iou / sym; AP / UDS recomputed per resample (checked == pack)."""
import os, sys, json, pickle, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as Rot
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
R = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/corrupt_1117'
TGT = sys.argv[1]
if TGT == 'antifs':
    META = 'D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz'; c = S.CFG['outdoor_test']; SPL = 'antifs_test'
    ARMS = [('PST (IR+LiDAR)', 'out2antifs'), ('IR only', 'out2antifs_ir'), ('LiDAR only', 'out2antifs_lid'), ('RGB only', 'out2antifs_none')]
else:
    META = 'E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval/test_meta.npz'; c = S.CFG['real_test']; SPL = 'real_test'
    ARMS = [('PST (IR+LiDAR)', 'in2mav'), ('IR only', 'in2mav_ir'), ('LiDAR only', 'in2mav_lid'), ('RGB only', 'in2mav_none')]
CORR = ['lowlight', 'noise', 'blur', 'down', 'haze']
CONDS = ['clean'] + ['%s_s%d' % (k, s) for s in (1, 2) for k in CORR]
M = np.load(META); g = M['gt']; K = M['K']; N = len(g)
seq = np.array([str(s) for s in M['seq']]); frm = [str(f) for f in M['frame']]
key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
B = 2000; L = 176


def load(fn):
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(fn, 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def per_frame(P):
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf'])
    d = np.linalg.norm(P['pt'] - g[:, :3], axis=1)
    if c['rel']:
        d = d / np.linalg.norm(g[:, :3], axis=1)
    tp = np.zeros(N, bool); se = np.full(N, np.nan); ae = np.full(N, np.nan)
    for i in np.where(has)[0]:
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        if S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(P['plwh'][i] - g[i, 3:6]))
        ae[i] = S.sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if np.isfinite(P['peul'][i]).all() else 90.0
    return dict(has=has, conf=P['conf'], d=d, tp=tp, se=se, ae=ae)


RS = np.linspace(0, 1, 101); RS = RS[RS > 0.1]


def fast_ap(sc, hit, n_gt):
    if not len(sc):
        return 0.0
    o = np.argsort(-sc, kind='stable'); t = hit[o].astype(float)
    ctp = np.cumsum(t); cfp = np.cumsum(1 - t); rec = ctp / max(n_gt, 1); prec = ctp / np.maximum(ctp + cfp, 1)
    smax = np.maximum.accumulate(prec[::-1])[::-1]
    k = np.searchsorted(rec, RS, side='left')
    p = np.where(k < len(rec), smax[np.minimum(k, len(rec) - 1)], 0.0)
    return float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9)


def fast(F, idx):
    h = F['has'][idx]; n = len(idx)
    sc = F['conf'][idx][h]; d = F['d'][idx][h]
    ap = 100 * np.mean([fast_ap(sc, d <= th, n) for th in c['thr']])
    t = F['tp'][idx]
    if not t.any():
        return dict(AP=ap, UDS=50 * ap / 100, ATE=np.nan, AOE=np.nan, mDR=0.0)
    ATE = F['d'][idx][t].mean(); ASE = F['se'][idx][t].mean(); AOE = F['ae'][idx][t].mean(); mdr = t.sum() / n
    tps = [1 - min(1, ATE / c['a']), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr]
    return dict(AP=ap, UDS=100 * (0.5 * ap / 100 + 0.5 * np.mean(tps)), ATE=ATE, AOE=AOE, mDR=100 * mdr)


def groups():
    out = []
    for s in sorted(np.unique(seq)):
        ix = np.where(seq == s)[0]
        out.append(ix[sorted(range(len(ix)), key=lambda j: (len(frm[ix[j]]), frm[ix[j]]))])
    return out


G = groups()


def resample(rng):
    if TGT == 'antifs':   # circular moving blocks within each sequence
        o = []
        for ix in G:
            n = len(ix); Lg = min(L, n); nb = int(np.ceil(n / Lg))
            st = rng.integers(0, n, nb); pos = (st[:, None] + np.arange(Lg)[None, :]) % n
            o.append(ix[pos.ravel()[:n]])
        return np.concatenate(o)
    return np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])


if __name__ == '__main__':
    FF, PT = {}, {}
    for _, a in ARMS:
        for cd in CONDS:
            fn = '%s/%s/%s/preds/pcmd_fused_test.pkl' % (R, a, cd)
            if not os.path.exists(fn + '.score_%s.json' % SPL):
                continue
            P = load(fn); ref = S.metrics(P, g, K, np.arange(N), c); st = json.load(open(fn + '.score_%s.json' % SPL))['all']
            assert all(abs(ref[m] - st[m]) < 1e-9 or (np.isnan(ref[m]) and np.isnan(st[m])) for m in ref), (a, cd, ref, st)
            F = per_frame(P); fa = fast(F, np.arange(N))
            assert abs(fa['AP'] - ref['AP']) < 1e-9 and abs(fa['UDS'] - ref['UDS']) < 1e-9, (a, cd, fa, ref)
            FF[(a, cd)] = F; PT[(a, cd)] = ref
    rng0 = np.random.default_rng(7); idx0 = resample(rng0)   # check fast == pack on a resample with duplicates
    for (a, cd), F in list(FF.items())[:4]:
        ref = S.metrics(load('%s/%s/%s/preds/pcmd_fused_test.pkl' % (R, a, cd)), g, K, idx0, c); fa = fast(F, idx0)
        assert abs(fa['AP'] - ref['AP']) < 1e-9 and abs(fa['UDS'] - ref['UDS']) < 1e-9, ('resample', a, cd, fa, ref)
    pst = ARMS[0][1]; DIF = {}
    for cd in CONDS:
        if (pst, cd) not in FF:
            continue
        others = [a for _, a in ARMS[1:] if (a, cd) in FF]
        if not others:
            continue
        rng = np.random.default_rng(2026)
        bs = {a: {'UDS': [], 'AP': []} for a in others}
        for _ in range(B):
            ix = resample(rng); p = fast(FF[(pst, cd)], ix)
            for a in others:
                q = fast(FF[(a, cd)], ix)
                for m in ('UDS', 'AP'):
                    bs[a][m].append(p[m] - q[m])
        for a in others:
            DIF[(a, cd)] = {m: dict(d=PT[(pst, cd)][m] - PT[(a, cd)][m], lo=float(np.percentile(bs[a][m], 2.5)), hi=float(np.percentile(bs[a][m], 97.5)))
                            for m in ('UDS', 'AP')}
    # ---- markdown
    name = dict(ARMS); inv = {a: n for n, a in ARMS}
    lab = lambda cd: 'clean' if cd == 'clean' else '%s s%s' % tuple(cd.rsplit('_s', 1))
    unit = 'rel.' if c['rel'] else 'm'
    lines = ['| condition | ' + ' | '.join(inv[a] for _, a in ARMS) + ' |', '|---|' + '---|' * len(ARMS)]
    for cd in CONDS:
        if not any((a, cd) in PT for _, a in ARMS):
            continue
        cells = []
        for _, a in ARMS:
            v = PT.get((a, cd))
            cells.append('—' if v is None else '**%.2f** (AP %.2f / ATE %.3f / AOE %.1f / mDR %.1f)' % (v['UDS'], v['AP'], v['ATE'], v['AOE'], v['mDR']))
        lines.append('| %s | %s |' % (lab(cd), ' | '.join(cells)))
    lines += ['', 'Cells: **UDS** (3D AP / ATE (%s) / AOE (deg) / mDR (%%)).' % unit, '',
              'Paired difference PST minus arm, UDS [95%% CI] and AP [95%% CI] (%s, B = %d):' % ('circular blocks L = 176 frames within sequences' if TGT == 'antifs' else 'sequence resampling', B), '',
              '| condition | ' + ' | '.join('PST − %s' % inv[a] for _, a in ARMS[1:]) + ' |', '|---|' + '---|' * (len(ARMS) - 1)]
    for cd in CONDS:
        if not any((a, cd) in DIF for _, a in ARMS[1:]):
            continue
        cells = []
        for _, a in ARMS[1:]:
            v = DIF.get((a, cd))
            if v is None:
                cells.append('—'); continue
            u, p = v['UDS'], v['AP']
            star = '' if u['lo'] <= 0 <= u['hi'] else ' *'
            cells.append('UDS %+.2f [%+.2f, %+.2f]%s; AP %+.2f [%+.2f, %+.2f]' % (u['d'], u['lo'], u['hi'], star, p['d'], p['lo'], p['hi']))
        lines.append('| %s | %s |' % (lab(cd), ' | '.join(cells)))
    lines += ['', '`*` = UDS interval excludes 0.']
    open('%s/table_%s.md' % (R, TGT), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    json.dump(dict(points={'%s|%s' % k: v for k, v in PT.items()}, diffs={'%s|%s' % k: v for k, v in DIF.items()}, B=B, L=L if TGT == 'antifs' else None),
              open('%s/boot_%s.json' % (R, TGT), 'w'), indent=1, default=float)
    print('\n'.join(lines))
