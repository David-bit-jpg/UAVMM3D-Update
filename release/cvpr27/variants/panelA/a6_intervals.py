# -*- coding: utf-8 -*-
"""A6: missing intervals. CPU only, existing predictions only; labels only through the scorers.
 (1) MAV6D paired sequence bootstrap (B = 2000, seed 0; = clean3/boot12.py) for no-mirror-loss students (clean3/nomir) and no IR-LiDAR check
     (sup2/nogate) vs PST (srcx/in2mav), val and test, fused and nofuse.
 (2) AntiUAV3D v2 test: moving-block bootstrap (= A3/a_block_boot.py) at L = 50 / 100 / 176 / 300 for PST - RGB-only and PST - sim-only.
 (3) Band interaction on AntiUAV3D test: PST - RGB-only per GT-range band (<20, 20-40, 40-60, >=60 m), bootstrap of the between-band differences
     (L = 176, whole-sequence block resample then restricted to the band, as orient_dist C2), two-sided bootstrap p, Holm over the 6 band pairs."""
import json, sys, pickle, importlib.util
import numpy as np
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA')
from pa_common import *  # noqa
res = {}
# ---------------------------------------------------------------- (1)
ARMS = {'PST': X + '/preds/pcmd_fused_%s.pkl', 'PST_nofuse': X + '/preds/pcmd_nofuse_%s.pkl',
        'nomir': R13 + '/clean3/nomir/preds/nomir_fused_%s.pkl', 'nomir_nofuse': R13 + '/clean3/nomir/preds/nomir_nofuse_%s.pkl',
        'nogate': R13 + '/sup2/nogate/preds/nogate_fused_%s.pkl', 'nogate_nofuse': R13 + '/sup2/nogate/preds/nogate_nofuse_%s.pkl'}
PAIRS = [('PST', 'nomir'), ('PST', 'nogate'), ('PST_nofuse', 'nomir_nofuse'), ('PST_nofuse', 'nogate_nofuse')]
KEYS = ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR', 'yaw_med')
for sp in ('val', 'test'):
    M = meta(sp); seq = np.array([str(s) for s in M['seq']]); N = len(seq)
    FF, rows = {}, {}
    for nm, p in ARMS.items():
        Pl = BU.load_list(P(p % sp), M); F = BU.frame_q(Pl, M); s = BU.stats(F, np.arange(N))
        ref = S.metrics(Pl, M['gt'], M['K'], np.arange(N), CFG)
        assert all(abs(ref[k] - s[k]) < 1e-9 for k in ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR')), (nm, ref, s)
        FF[nm] = F; rows[nm] = s
    us = np.unique(seq); idx = {u: np.where(seq == u)[0] for u in us}
    rng = np.random.default_rng(0); acc = {p: {k: [] for k in KEYS} for p in PAIRS}
    for _ in range(BU.B_BOOT):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        st = {nm: BU.stats(FF[nm], ii) for nm in ARMS}
        for a, b in PAIRS:
            for k in KEYS:
                acc[(a, b)][k].append(st[a][k] - st[b][k])
    perseq = {'%s-%s' % (a, b): int(sum(BU.stats(FF[a], idx[u])['UDS'] > BU.stats(FF[b], idx[u])['UDS'] for u in us)) for a, b in PAIRS}
    boot = {'%s - %s' % (a, b): {k: dict(point=float(rows[a][k] - rows[b][k]), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                                          frac_pos=float(np.mean(np.array(v) > 0))) for k, v in A.items()} for (a, b), A in acc.items()}
    res['mav_' + sp] = dict(rows=rows, boot=boot, n_seq=int(len(us)), seqs_where_a_beats_b_UDS=perseq)
    print(sp, json.dumps({k: {m: [round(v['point'], 2), round(v['lo'], 2), round(v['hi'], 2)] for m, v in d.items()} for k, d in boot.items()}), perseq, flush=True)

# ---------------------------------------------------------------- (2) and (3)
spec = importlib.util.spec_from_file_location('abb', R13 + '/A3/a_block_boot.py')
AB = importlib.util.module_from_spec(spec); spec.loader.exec_module(AB)
FF = {}
for nm, fn in AB.ARMS.items():
    Pl = AB.load(fn); FF[nm] = AB.per_frame(Pl)
    ref = S.metrics(Pl, AB.g, AB.K, np.arange(AB.N), AB.c); fa, fu = AB.fast_metrics(FF[nm], np.arange(AB.N))
    assert abs(fu - ref['UDS']) < 1e-9 and abs(fa - ref['AP']) < 1e-9
point = {nm: AB.fast_metrics(F, np.arange(AB.N)) for nm, F in FF.items()}
groups = []
for s in sorted(np.unique(AB.seq)):
    ix = np.where(AB.seq == s)[0]; groups.append(ix[np.argsort(AB.frm[ix], kind='stable')])
pairs = [('PST', 'Sim-only'), ('PST', 'RGB-only teachers')]
res['anti_blocklen'] = {}
for L in (50, 100, 176, 300):
    rng = np.random.default_rng(0); vals = {nm: np.zeros((AB.B, 2)) for nm in FF}
    for b in range(AB.B):
        ii = AB.block_indices(rng, groups, L)
        for nm, F in FF.items():
            vals[nm][b] = AB.fast_metrics(F, ii)
    r = {}
    for a, bb in pairs:
        dd = vals[a] - vals[bb]
        r['%s - %s' % (a, bb)] = {m: dict(point=point[a][j] - point[bb][j], ci=[float(np.percentile(dd[:, j], 2.5)), float(np.percentile(dd[:, j], 97.5))],
                                         sd=float(dd[:, j].std()), frac_le0=float(np.mean(dd[:, j] <= 0))) for j, m in enumerate(('AP', 'UDS'))}
    res['anti_blocklen'][str(L)] = r
    print('L', L, json.dumps(r, default=lambda x: round(float(x), 3)), flush=True)

# (3) band interaction
rng_ = np.linalg.norm(AB.g[:, :3], axis=1)
BANDS = [('<20', 0, 20), ('20-40', 20, 40), ('40-60', 40, 60), ('>=60', 60, 1e9)]
inb = {b: (rng_ >= lo) & (rng_ < hi) for b, lo, hi in BANDS}


def band_metrics(F, ii, b):
    jj = ii[inb[b][ii]]
    ap, uds = AB.fast_metrics(F, jj)
    t = jj[F['tp'][jj]]
    return uds, ap, (F['d'][t].mean() if len(t) else np.nan)


ptb = {b: {nm: band_metrics(FF[nm], np.arange(AB.N), b) for nm in ('PST', 'RGB-only teachers')} for b, _, _ in BANDS}
L = 176; rng = np.random.default_rng(0)
D = {b: np.zeros((AB.B, 3)) for b, _, _ in BANDS}
for i in range(AB.B):
    ii = AB.block_indices(rng, groups, L)
    for b, _, _ in BANDS:
        D[b][i] = np.subtract(band_metrics(FF['PST'], ii, b), band_metrics(FF['RGB-only teachers'], ii, b))
bands_out = {}
for b, _, _ in BANDS:
    pt = np.subtract(ptb[b]['PST'], ptb[b]['RGB-only teachers'])
    bands_out[b] = {m: dict(point=float(pt[j]), ci=[float(np.nanpercentile(D[b][:, j], 2.5)), float(np.nanpercentile(D[b][:, j], 97.5))]) for j, m in enumerate(('UDS', 'AP', 'ATE_rel'))}
    bands_out[b]['n'] = int(inb[b].sum())
inter = {}
names = [b for b, _, _ in BANDS]
for j, m in enumerate(('UDS', 'AP', 'ATE_rel')):
    rows = []
    for x in range(4):
        for y in range(x + 1, 4):
            a, bb = names[x], names[y]
            dd = D[a][:, j] - D[bb][:, j]; dd = dd[np.isfinite(dd)]
            pt = bands_out[a][m]['point'] - bands_out[bb][m]['point']
            p = float(min(1.0, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0))))
            rows.append(dict(pair='%s minus %s' % (a, bb), point=float(pt), ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], p_boot=p))
    # Holm over the 6 pairs
    o = np.argsort([r['p_boot'] for r in rows]); mx = 0.0
    for rank, k in enumerate(o):
        adj = min(1.0, (6 - rank) * rows[k]['p_boot']); mx = max(mx, adj); rows[k]['p_holm'] = mx
    inter[m] = rows
res['anti_bands'] = dict(L=L, B=AB.B, bands=bands_out, interaction=inter, note='p from 2*min(P(d<=0),P(d>=0)) over B=2000 block resamples; Holm across the 6 pairs, per metric')
print(json.dumps(res['anti_bands'], indent=1, default=lambda x: round(float(x), 4)), flush=True)
json.dump(res, open(PA + '/a6_intervals.json', 'w', encoding='utf-8'), indent=1, default=float)
print('done')
