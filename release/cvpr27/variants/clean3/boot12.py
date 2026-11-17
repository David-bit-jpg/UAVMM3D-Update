# -*- coding: utf-8 -*-
"""[clean3 items 1-2] CPU only, existing predictions only. Reuses review1113/clean2/buildup/buildup.py (load_list, frame_q, stats: asserted
== pack score.py metrics) for paired sequence bootstrap (B = 2000, seed 0; same resampled sequences for both arms, AP recomputed per resample).
 1. PST (srcx/in2mav pcmd_fused) vs omega=0 IR-only depth targets (review1113/clean2/w0 w0_fused), val + test.
 2. RGB-only teachers (srcx/in2mav_none pcmd_fused): full rows val/test; PST - RGB-only bootstrap (UDS, AOE, + others)."""
import os, sys, json, pickle, importlib.util
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'; G = GP.rsplit('/', 1)[0]
spec = importlib.util.spec_from_file_location('buildup', GP + '/review1113/clean2/buildup/buildup.py')
BU = importlib.util.module_from_spec(spec); spec.loader.exec_module(BU)
S = BU.S; CFG = BU.CFG
ARMS = {'PST': GP + '/srcx/in2mav/preds/pcmd_fused_%s.pkl', 'w0': GP + '/review1113/clean2/w0/preds/w0_fused_%s.pkl',
        'w0_nofuse': GP + '/review1113/clean2/w0/preds/w0_nofuse_%s.pkl', 'PST_nofuse': GP + '/srcx/in2mav/preds/pcmd_nofuse_%s.pkl',
        'RGBonly': GP + '/srcx/in2mav_none/preds/pcmd_fused_%s.pkl'}
PAIRS = [('PST', 'w0'), ('PST_nofuse', 'w0_nofuse'), ('PST', 'RGBonly')]
KEYS = ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR', 'yaw_med')
res = {}
for sp in ('val', 'test'):
    M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); seq = np.array([str(s) for s in M['seq']]); N = len(seq)
    FF, rows = {}, {}
    for nm, p in ARMS.items():
        P = BU.load_list(pickle.load(open(p % sp, 'rb')), M); F = BU.frame_q(P, M); s = BU.stats(F, np.arange(N))
        ref = S.metrics(P, M['gt'], M['K'], np.arange(N), CFG)
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
    # per-sequence UDS difference sign (descriptive)
    perseq = {'%s-%s' % (a, b): int(sum(BU.stats(FF[a], idx[u])['UDS'] > BU.stats(FF[b], idx[u])['UDS'] for u in us)) for a, b in PAIRS}
    boot = {'%s - %s' % (a, b): {k: dict(point=float(rows[a][k] - rows[b][k]), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                                          frac_pos=float(np.mean(np.array(v) > 0))) for k, v in A.items()} for (a, b), A in acc.items()}
    res[sp] = dict(rows=rows, boot=boot, n_seq=int(len(us)), n_frames=int(N), seqs_where_a_beats_b_UDS=perseq)
    print(sp, len(us), 'seqs', json.dumps({k: {m: round(float(v[m]), 3) for m in KEYS} for k, v in rows.items()}), flush=True)
    print(json.dumps({k: {m: [round(v['point'], 2), round(v['lo'], 2), round(v['hi'], 2)] for m, v in d.items()} for k, d in boot.items()}), flush=True)
    print(perseq, flush=True)
json.dump(res, open(OUT + '/boot12.json', 'w', encoding='utf-8'), indent=1, default=float)
