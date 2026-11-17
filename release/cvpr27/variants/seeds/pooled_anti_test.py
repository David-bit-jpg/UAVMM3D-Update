# 11-14: AntiUAV3D test, PST vs RGB-only teachers, 3 runs, pooled paired moving-block bootstrap (L=176, B=2000); helpers copied from cpu5.py
# -*- coding: utf-8 -*-
import os, sys, json, pickle, hashlib, importlib.util
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scipy.spatial.transform import Rotation as Rot
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'; G = GP.rsplit('/', 1)[0]; R = GP + '/review1113'; X = GP + '/srcx'
B = 2000


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
os.environ.update(SX_SRC='in', SX_TGT='mav', SX_TEACH='full')
BU = mod('buildup', R + '/clean2/buildup/buildup.py')
SP = mod('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py'); CO = SP.CFG['outdoor_test']
def anti_meta(sp):
    if sp == 'test':
        M = np.load('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz')
        return dict(gt=M['gt'], K=M['K'], seq=np.array([str(s) for s in M['seq']]), frame=np.array([str(f) for f in M['frame']]))
    V = pickle.load(open('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/val.pkl', 'rb'))
    return dict(gt=np.array([r['box9d_ours'] for r in V], np.float64), K=np.array([r['K'] for r in V], np.float64),
                seq=np.array([str(r['seq']) for r in V]), frame=np.array([str(r['frame']) for r in V]))


def a_load(L_, M):
    key = {(s, f): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; N = len(M['seq'])
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in L_:
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def a_frame(P, M):
    g, K = M['gt'], M['K']; N = len(g)
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf'])
    d = np.linalg.norm(P['pt'] - g[:, :3], axis=1) / np.linalg.norm(g[:, :3], axis=1)
    tp = np.zeros(N, bool); se = np.full(N, np.nan); ae = np.full(N, np.nan)
    for i in np.where(has)[0]:
        fe = np.isfinite(P['peul'][i]).all(); eul = P['peul'][i] if fe else np.zeros(3)
        if SP.iou(SP.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), SP.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(P['plwh'][i] - g[i, 3:6]))
        ae[i] = SP.sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if fe else 90.0
    return dict(has=has, conf=P['conf'], d=d, tp=tp, se=se, ae=ae)


def a_stats(F, idx):
    h = F['has'][idx]; n = len(idx)
    ap = 100 * np.mean([BU.fast_ap(F['conf'][idx][h], F['d'][idx][h] <= th, n) for th in CO['thr']])
    t = idx[F['tp'][idx]]
    if not len(t):
        return dict(UDS=50 * ap / 100, AP=ap, ATE=np.nan, ASE=np.nan, AOE=np.nan, mDR=0.0)
    ATE, ASE, AOE = F['d'][t].mean(), F['se'][t].mean(), F['ae'][t].mean(); mdr = len(t) / n
    uds = 100 * (0.5 * ap / 100 + 0.5 * np.mean([1 - min(1, ATE / CO['a']), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr]))
    return dict(UDS=uds, AP=ap, ATE=ATE, ASE=ASE, AOE=AOE, mDR=100 * mdr)


def groups_of(M):
    seq = M['seq']; frm = np.array([int(f.split('.')[0]) for f in M['frame']]); gr = []
    for s in sorted(np.unique(seq)):
        ix = np.where(seq == s)[0]; gr.append(ix[np.argsort(frm[ix], kind='stable')])
    return gr


def block_indices(rng, groups, L):   # = review1113/A3/a_block_boot.py
    out = []
    for ix in groups:
        n = len(ix); Lg = min(L, n); nb = int(np.ceil(n / Lg))
        st = rng.integers(0, n, nb); pos = (st[:, None] + np.arange(Lg)[None, :]) % n
        out.append(ix[pos.ravel()[:n]])
    return np.concatenate(out)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
RUNS = {'PST1': X + '/out2antifs', 'RGB1': X + '/out2antifs_none', 'PST2': R + '/seeds/antifs_full_s10', 'RGB2': R + '/seeds/antifs_none_s10',
        'PST3': R + '/seeds/antifs_full_s20', 'RGB3': R + '/seeds/antifs_none_s20'}
PAIRS = [('PST1', 'RGB1'), ('PST2', 'RGB2'), ('PST3', 'RGB3')]
M = anti_meta('test')
FF = {k: a_frame(a_load(pickle.load(open(v + '/preds/pcmd_fused_test.pkl', 'rb')), M), M) for k, v in RUNS.items()}
N = len(M['gt']); allix = np.arange(N)
pt = {k: a_stats(F, allix) for k, F in FF.items()}
for k in RUNS: print(k, {m: round(float(v), 3) for m, v in pt[k].items()})
gr = groups_of(M); rng = np.random.default_rng(0); D = []; DK = {m: [] for m in ('AP', 'ATE', 'AOE', 'ASE')}
for b in range(2000):
    ii = block_indices(rng, gr, 176); st = {k: a_stats(F, ii) for k, F in FF.items()}
    D.append(np.mean([st[a]['UDS'] - st[c]['UDS'] for a, c in PAIRS]))
    for m in DK: DK[m].append(np.mean([st[a][m] - st[c][m] for a, c in PAIRS]))
pp = float(np.mean([pt[a]['UDS'] - pt[c]['UDS'] for a, c in PAIRS]))
print('per-run diffs', [round(float(pt[a]['UDS'] - pt[c]['UDS']), 2) for a, c in PAIRS])
print('PST mean %.2f +- %.2f | RGB mean %.2f +- %.2f' % (np.mean([pt[a]['UDS'] for a, _ in PAIRS]), np.std([pt[a]['UDS'] for a, _ in PAIRS], ddof=1), np.mean([pt[c]['UDS'] for _, c in PAIRS]), np.std([pt[c]['UDS'] for _, c in PAIRS], ddof=1)))
print('pooled UDS diff %.2f CI %s' % (pp, np.round(ci(D), 2)))
for m in DK: print('pooled %s diff %.3f CI %s' % (m, float(np.mean([pt[a][m] - pt[c][m] for a, c in PAIRS])), np.round(ci(DK[m]), 3)))
json.dump(dict(pt={k: {m: float(v) for m, v in d.items()} for k, d in pt.items()}, pooled=pp, ci=ci(D)), open(R + '/seeds/pooled_anti_test.json', 'w'), indent=1)
