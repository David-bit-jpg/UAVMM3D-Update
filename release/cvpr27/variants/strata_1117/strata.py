# -*- coding: utf-8 -*-
"""[11-17] Per-stratum benefit of privileged modalities (IR, LiDAR) from EXISTING test predictions. CPU only, no training.
Definitions: README.md section 1 (written before any per-stratum number). Usage:
  D:/Miniconda3/envs/city/python.exe strata.py            (re-run picks up in2mav_ir / in2mav_lid once their preds exist)
Writes only into this directory: results.json, feat_<target>.npz (cached image features), README.md section 2."""
import os, sys, json, pickle, glob, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, 'D:/UAV3D_baselines_pack/scripts'); import score as S

OUT = os.path.dirname(os.path.abspath(__file__))
SRCX = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx'
B, SEED, L = 2000, 0, 176
TG = {
    'RealDrone3D': dict(meta='D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz', cfg='outdoor_test',
                        index='D:/mmcache/outdoor_real_fs/test/index.pkl', boot='block',
                        arms=[('PST', SRCX + '/out2antifs/preds/pcmd_fused_test.pkl'), ('IR', SRCX + '/out2antifs_ir/preds/pcmd_fused_test.pkl'),
                              ('LiDAR', SRCX + '/out2antifs_lid/preds/pcmd_fused_test.pkl'), ('none', SRCX + '/out2antifs_none/preds/pcmd_fused_test.pkl'),
                              ('E1', 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s0/preds/e1_fused_test.pkl')]),
    'MAV6D': dict(meta='E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval/test_meta.npz', cfg='real_test',
                  index='E:/mmcache/mav6d_cn_obj/test/index.pkl', boot='seq',
                  arms=[('PST', SRCX + '/in2mav/preds/pcmd_fused_test.pkl'), ('IR', SRCX + '/in2mav_ir/preds/pcmd_fused_test.pkl'),
                        ('LiDAR', SRCX + '/in2mav_lid/preds/pcmd_fused_test.pkl'), ('none', SRCX + '/in2mav_none/preds/pcmd_fused_test.pkl')]),
}
PAIRS = [('IR', 'none'), ('LiDAR', 'none'), ('PST', 'IR'), ('PST', 'LiDAR'), ('PST', 'none'), ('PST', 'E1')]
MK = ('AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS')


def load_P(p, M):
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; N = len(key)
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(p, 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], float).reshape(-1, 9); s = np.asarray(r['scores'], float).reshape(-1)
        if i is None or not len(b): continue
        j = int(np.argmax(s)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], s[j]
    return P


def per_frame(P, g, K, c):
    """per-frame pieces of S.metrics (same code path, frame by frame)."""
    N = len(g); pt, pl, pe, cf = P['pt'], P['plwh'], P['peul'], P['conf']
    has = np.isfinite(pt).all(1) & np.isfinite(cf)
    d = np.linalg.norm(pt - g[:, :3], axis=1)
    if c['rel']: d = d / np.linalg.norm(g[:, :3], axis=1)
    tp = np.zeros(N, bool); se = np.full(N, np.nan); ae = np.full(N, np.nan)
    for i in np.where(has)[0]:
        eul = pe[i] if np.isfinite(pe[i]).all() else np.zeros(3)
        if S.iou(S.box2d(pt[i], pl[i], eul, K[i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1: continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(pl[i] - g[i, 3:6]))
        ae[i] = S.sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if np.isfinite(pe[i]).all() else 90.0
    return dict(has=has, cf=np.where(has, cf, -np.inf), d=d, tp=tp, se=se, ae=ae)


RS = np.linspace(0, 1, 101); RS = RS[RS > 0.1]


def fast(F, idx, c):
    """vectorised S.metrics on an index multiset."""
    n = len(idx)
    h = idx[F['has'][idx]]
    if len(h):
        o = np.argsort(-F['cf'][h], kind='stable'); dd = F['d'][h][o]; k = np.arange(1, len(h) + 1)
        aps = []
        for th in c['thr']:
            ctp = np.cumsum(dd <= th); rec = ctp / max(n, 1); prec = ctp / k
            smax = np.maximum.accumulate(prec[::-1])[::-1]
            j = np.searchsorted(rec, RS, 'left'); p = np.where(j < len(h), smax[np.minimum(j, len(h) - 1)], 0.0)
            aps.append(float(np.mean(np.clip(p - 0.1, 0, None)) / 0.9))
        ap = 100 * np.mean(aps)
    else:
        ap = 0.0
    t = idx[F['tp'][idx]]
    if len(t):
        ATE, ASE, AOE = F['d'][t].mean(), F['se'][t].mean(), F['ae'][t].mean(); mdr = len(t) / max(n, 1)
        tps = [1 - min(1, ATE / c['a']), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr]
    else:
        ATE = ASE = AOE = np.nan; mdr = 0.0; tps = [0, 0, 0, 0]
    return dict(AOE=AOE, ATE=ATE, ASE=ASE, AP=ap, UDS=100 * (0.5 * ap / 100 + 0.5 * np.mean(tps)), mDR=100 * mdr)


def features(name, M, cfg):
    """GT-derived stratum variables; brightness from raw images (cached)."""
    fn = os.path.join(OUT, 'feat_%s.npz' % name)
    if os.path.exists(fn): return dict(np.load(fn))
    I = pickle.load(open(cfg['index'], 'rb')); km = {(d['seq'], d['frame']): d for d in I['metas']}
    metas = [km[(str(s), str(f))] for s, f in zip(M['seq'], M['frame'])]
    g = M['gt']; N = len(g); t = g[:, :3]; dist = np.linalg.norm(t, axis=1)
    R = Rot.from_euler('xyz', g[:, 6:9]).as_matrix()                       # (N,3,3) body->camera
    v = -t / dist[:, None]; vb = np.einsum('nji,nj->ni', R, v)             # view vector in body frame
    elev = np.degrees(np.arccos(np.clip(vb[:, 2], -1, 1)))
    a = np.degrees(np.abs(np.arctan2(vb[:, 1], vb[:, 0]))) % 180; yaw = np.minimum(a, 180 - a)
    boxes = np.array([S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], np.asarray(metas[i]['K_raw'], float)) for i in range(N)])
    width = boxes[:, 2] - boxes[:, 0]
    if name == 'MAV6D':
        os.environ['SX_SRC'] = 'in'; os.environ['SX_TGT'] = 'mav'
        sys.path.insert(0, SRCX + '/lib'); import sx; rd = sx.imread_raw
    else:
        import cv2; rd = lambda m: cv2.imread(m['raw_path'], cv2.IMREAD_COLOR)

    def bright(i):
        import cv2
        img = rd(metas[i]); gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); H, W = gr.shape
        x0, y0, x1, y1 = boxes[i]; cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        def rc(a, b, c_, d_):
            return int(np.clip(np.floor(a), 0, W)), int(np.clip(np.floor(b), 0, H)), int(np.clip(np.ceil(c_), 0, W)), int(np.clip(np.ceil(d_), 0, H))
        bx = rc(x0, y0, x1, y1); ox = rc(cx - w, cy - h, cx + w, cy + h)
        inner = gr[bx[1]:bx[3], bx[0]:bx[2]]; outer = gr[ox[1]:ox[3], ox[0]:ox[2]]
        if inner.size == 0 or outer.size <= inner.size: return np.nan
        ring = (outer.sum(dtype=np.float64) - inner.sum(dtype=np.float64)) / (outer.size - inner.size)
        return float(inner.mean() - ring)
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex: con = np.array(list(ex.map(bright, range(N))))
    print(name, 'brightness read %d frames in %.0f s' % (N, time.time() - t0), flush=True)
    F = dict(dist=dist, width=width, contrast=con, elev=elev, yaw=yaw, cls=np.asarray(M['cls']).astype(str))
    np.savez(fn, **F); return F


def strata(F):
    def terc(x, names):
        q = np.nanpercentile(x, [100 / 3, 200 / 3]); b = np.digitize(x, q)
        return [('%s (%s)' % (names[k], ['<%.3g' % q[0], '%.3g-%.3g' % (q[0], q[1]), '>=%.3g' % q[1]][k]), np.where(b == k)[0]) for k in range(3)]
    out = [('all', [('all', np.arange(len(F['dist'])))])]
    out.append(('dist', terc(F['dist'], ['near', 'mid', 'far'])))
    out.append(('width', terc(F['width'], ['tiny', 'medium', 'large'])))
    c = F['contrast']
    out.append(('bright', [('dark-on-bright (c<0)', np.where(c < 0)[0]), ('bright-on-dark (c>=0)', np.where(c >= 0)[0])]))
    out.append(('|contrast|', terc(np.abs(c), ['low', 'mid', 'high'])))
    out.append(('elev', terc(F['elev'], ['top-down', 'mid', 'edge-on/below'])))
    y = F['yaw']
    out.append(('yaw', [('frontal (<30)', np.where(y < 30)[0]), ('oblique (30-60)', np.where((y >= 30) & (y < 60))[0]), ('side (>=60)', np.where(y >= 60)[0])]))
    out.append(('type', [(k, np.where(F['cls'] == k)[0]) for k in sorted(set(F['cls']))]))
    return out


def resamples(M, mode, rng):
    N = len(M['seq'])
    if mode == 'block':
        nb = int(np.ceil(N / L))
        for _ in range(B):
            st = rng.integers(0, N, nb); yield ((st[:, None] + np.arange(L)[None]) % N).reshape(-1)[:N]
    else:
        seqs = np.asarray(M['seq']); us = sorted(set(seqs)); ix = {s: np.where(seqs == s)[0] for s in us}
        for _ in range(B):
            yield np.concatenate([ix[us[k]] for k in rng.integers(0, len(us), len(us))])


def run(name, cfg):
    M = np.load(cfg['meta']); c = S.CFG[cfg['cfg']]; g, K = M['gt'], M['K']
    arms = [(a, p) for a, p in cfg['arms'] if os.path.exists(p)]
    print(name, 'arms:', [a for a, _ in arms], flush=True)
    Ps = {a: load_P(p, M) for a, p in arms}; Fs = {a: per_frame(Ps[a], g, K, c) for a in Ps}
    feat = features(name, M, cfg); ST = strata(feat)
    pairs = [(x, y) for x, y in PAIRS if x in Ps and y in Ps]
    res = {'arms': {a: p for a, p in arms}, 'n_frames': int(len(g)), 'strata': {}}
    # point estimates with S.metrics + exactness check of the fast version
    flat = [(s, b, idx) for s, bins in ST for b, idx in bins]
    for s, b, idx in flat:
        e = res['strata'].setdefault(s, {}).setdefault(b, {'n': int(len(idx)), 'arms': {}, 'diff': {}})
        for a in Ps:
            m = S.metrics(Ps[a], g, K, idx, c); f = fast(Fs[a], idx, c)
            for k in MK:
                assert (np.isnan(m[k]) and np.isnan(f[k])) or abs(m[k] - f[k]) < 1e-9, (name, s, b, a, k, m[k], f[k])
            e['arms'][a] = {k: float(m[k]) for k in MK}
    # paired bootstrap
    rng = np.random.default_rng(SEED); masks = [np.zeros(len(g), bool) for _ in flat]
    for mk, (_, _, idx) in zip(masks, flat): mk[idx] = True
    D = {(i, p): [] for i in range(len(flat)) for p in pairs}
    t0 = time.time()
    for r in resamples(M, cfg['boot'], rng):
        for i, mk in enumerate(masks):
            rr = r[mk[r]]
            mm = {a: fast(Fs[a], rr, c) for a in Ps}
            for x, y in pairs: D[(i, (x, y))].append([mm[x][k] - mm[y][k] for k in MK])
    print(name, 'bootstrap %.0f s' % (time.time() - t0), flush=True)
    for i, (s, b, idx) in enumerate(flat):
        e = res['strata'][s][b]
        for x, y in pairs:
            arr = np.array(D[(i, (x, y))]); pe = e['arms']
            e['diff']['%s-%s' % (x, y)] = {k: dict(est=float(pe[x][k] - pe[y][k]), lo=float(np.nanpercentile(arr[:, j], 2.5)), hi=float(np.nanpercentile(arr[:, j], 97.5)),
                                                   n_nan=int(np.isnan(arr[:, j]).sum())) for j, k in enumerate(MK)}
    return res, pairs


def fmt_d(d, k, nd):
    lo, hi = d[k]['lo'], d[k]['hi']; star = '**' if (lo > 0 or hi < 0) else ''
    return '%s%+.*f [%+.*f, %+.*f]%s' % (star, nd, d[k]['est'], nd, lo, nd, hi, star)


def table(name, res, pairs):
    arms = list(res['arms']); u = 'rel.' if name == 'RealDrone3D' else 'm'; nd_t = 3
    L_ = ['### %s (%d test frames; arms: %s)' % (name, res['n_frames'], ', '.join(arms)), '',
          'Per-arm scores (UDS / AOE deg / ATE %s):' % u, '',
          '| stratum | bin | n | ' + ' | '.join(arms) + ' |', '|' + '---|' * (3 + len(arms))]
    for s, bins in res['strata'].items():
        for b, e in bins.items():
            L_.append('| %s | %s | %d | ' % (s, b, e['n']) + ' | '.join('%.2f / %.2f / %.3f' % (e['arms'][a]['UDS'], e['arms'][a]['AOE'], e['arms'][a]['ATE']) for a in arms) + ' |')
    for k, nd, lab in (('UDS', 2, 'UDS (higher better)'), ('AOE', 2, 'AOE deg (lower better)'), ('ATE', nd_t, 'ATE %s (lower better)' % u)):
        L_ += ['', 'Paired difference A-B, %s, 95%% bootstrap CI; ** = CI excludes 0:' % lab, '',
               '| stratum | bin | ' + ' | '.join('%s-%s' % p for p in pairs) + ' |', '|' + '---|' * (2 + len(pairs))]
        for s, bins in res['strata'].items():
            for b, e in bins.items():
                L_.append('| %s | %s | ' % (s, b) + ' | '.join(fmt_d(e['diff']['%s-%s' % p], k, nd) for p in pairs) + ' |')
    return L_ + ['']


if __name__ == '__main__':
    R = {}; lines = []
    for name, cfg in TG.items():
        res, pairs = run(name, cfg); R[name] = res; lines += table(name, res, pairs)
    json.dump(R, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    rd = open(os.path.join(OUT, 'README.md'), encoding='utf-8').read().split('## 2. Results')[0]
    open(os.path.join(OUT, 'README.md'), 'w', encoding='utf-8').write(rd + '## 2. Results\n(generated by strata.py %s)\n\n' % time.strftime('%m-%d %H:%M') + '\n'.join(lines))
    print('done')
