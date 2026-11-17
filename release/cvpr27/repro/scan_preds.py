# Discovery helper: score every candidate test-prediction pickle with the pack scorer (top-1 box, sealed TEST meta).
# Output: scan_preds.json {path: {target, n_rows, sha256, size, mtime, AP, ATE, ASE, AOE, mDR, UDS}}
import os, sys, glob, json, pickle, hashlib, time
import numpy as np
sys.path.insert(0, 'D:/UAV3D_baselines_pack/scripts'); import score as S

META = {'mav': ('D:/UAV3D_baselines_pack/data/gt/real_test_meta.npz', 'real_test'),
        'anti': ('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz', 'outdoor_test')}
MM = {k: np.load(v[0], allow_pickle=True) for k, v in META.items()}
KEYS = {k: {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))} for k, M in MM.items()}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def score_rows(rows, tgt, sbar=None):
    M, key = MM[tgt], KEYS[tgt]; N = len(key)
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in rows:
        i = key.get((str(r['seq']), str(r['frame'])))
        b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); s = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b): continue
        j = int(np.argmax(s)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], (b[j, 3:6] if sbar is None else sbar), b[j, 6:9], s[j]
    return S.metrics(P, M['gt'], M['K'], np.arange(N), S.CFG[META[tgt][1]])


def target_of(rows):
    best = None
    for t, key in KEYS.items():
        n = sum((str(r['seq']), str(r['frame'])) in key for r in rows[:2000])
        if n > 0 and (best is None or n > best[1]): best = (t, n)
    return best[0] if best else None


if __name__ == '__main__':
    out = sys.argv[1]; files = [l.strip() for l in open(sys.argv[2], encoding='utf-8') if l.strip()]
    res = json.load(open(out, encoding='utf-8')) if os.path.exists(out) else {}
    for p in files:
        p = p.replace('\\', '/')
        if p in res: continue
        try:
            rows = pickle.load(open(p, 'rb'))
            if isinstance(rows, dict) or not len(rows) or 'boxes9d' not in rows[0]:
                res[p] = {'skip': 'not a prediction list'}; continue
            t = target_of(rows)
            if t is None: res[p] = {'skip': 'no test keys'}; continue
            m = score_rows(rows, t)
            res[p] = dict(target=t, n_rows=len(rows), sha256=sha(p), size=os.path.getsize(p),
                          mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p))), **{k: float(v) for k, v in m.items()})
            print(p, t, '%.2f' % m['UDS'], flush=True)
        except Exception as e:
            res[p] = {'skip': repr(e)[:200]}
        json.dump(res, open(out, 'w', encoding='utf-8'), indent=1)
