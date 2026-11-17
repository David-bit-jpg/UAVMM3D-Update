# [review1113 copy of srcx/lib/score_x.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2b/mav_s0/lib] sys.path -> redirected sx; code unchanged
# -*- coding: utf-8 -*-
"""[outdoor] score a prediction pickle with EXACTLY D:/UAV3D_baselines_pack/scripts/score.py (imported, its score() function;
we do not run its __main__, which would write into the read-only pack).
  python score_out.py pred.pkl outdoor_test      -> official one-shot test scoring (reads sealed GT meta in the pack)
  python score_out.py pred.pkl val               -> sanity on the unsealed val split (labels/val.pkl box9d_ours), same metric code/rules
Writes <pred>.score_<split>.json."""
import importlib.util, json, os, pickle, sys
import numpy as np
spec = importlib.util.spec_from_file_location('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sx   # 11-12: target-specific label paths (anti = v1 subset, antifs = v2 verified-intrinsics subset)


RULES = sx.T.get('rules', 'outdoor_test') if sx.T else 'outdoor_test'   # [11-13] mav = 'real_test' (indoor abs. thresholds, a = 1 m)


def score_meta(pred, meta):
    """= pack score.score() with the GT meta file given explicitly (same metric code; rules of the target: outdoor_test, or real_test for mav)."""
    M = np.load(meta)
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    N = len(M['seq']); P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(pred, 'rb')):
        i = key.get((str(r['seq']), str(r['frame'])))
        b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    c = S.CFG[RULES]; g = M['gt']; dist = np.linalg.norm(g[:, :3], axis=1) if c['rel'] else g[:, 2]   # bins as pack score()
    out = {'all': S.metrics(P, g, M['K'], np.arange(N), c)}
    for lo, hi in c['bins']:
        out['%g-%g' % (lo, hi)] = S.metrics(P, g, M['K'], np.where((dist >= lo) & (dist < hi))[0], c)
    return out


def score_val(pred):
    if str(sx.T['val_labels']).endswith('.npz'):   # [11-13] mav: val GT meta in the same npz format (cmpeval/val_meta.npz)
        return score_meta(pred, sx.T['val_labels'])
    V = pickle.load(open(sx.T['val_labels'], 'rb'))
    key = {(r['seq'], r['frame']): i for i, r in enumerate(V)}
    N = len(V); g = np.array([r['box9d_ours'] for r in V], np.float64); K = np.array([r['K'] for r in V], np.float64)
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(pred, 'rb')):
        i = key.get((str(r['seq']), str(r['frame'])))
        b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    c = S.CFG['outdoor_test']; dist = np.linalg.norm(g[:, :3], axis=1)
    out = {'all': S.metrics(P, g, K, np.arange(N), c)}
    for lo, hi in c['bins']:
        out['%g-%g' % (lo, hi)] = S.metrics(P, g, K, np.where((dist >= lo) & (dist < hi))[0], c)
    return out


if __name__ == '__main__':
    pred, split = sys.argv[1], sys.argv[2]
    res = S.score(pred, split) if split == 'outdoor_test' else (score_meta(pred, sx.T['gt_meta']) if split == sx.T['score_split'] else score_val(pred))
    json.dump(res, open(sys.argv[3] if len(sys.argv) > 3 else pred + '.score_%s.json' % split, 'w'), indent=1, default=float)
    for k, v in res.items():
        print('%-8s ' % k + '  '.join('%s %.3f' % (m, x) for m, x in v.items()))
