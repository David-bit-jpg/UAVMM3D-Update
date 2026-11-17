# -*- coding: utf-8 -*-
"""实例原型 v2：原始分辨率 1920×1080；相机固定 → 每片段时间中位数当背景（不看标签），|帧 − 背景| = 无人机剪影；
第 t 帧剪影当原型，到 t+k 帧按连续尺度仿射重采样做匹配（不再取整像素）→ log 尺度比。裁窗中心/大小来自检测器（部署 5 集成），真值只评测。"""
import pickle, os, sys
import numpy as np, cv2
G = '@@ROOT@@/output/camnorm/geo1004'
L = float(np.linalg.norm([0.3495, 0.3487, 0.1996])); BODY = 0.35
KS = (1, 2, 3, 5, 10)
COARSE = np.exp(np.linspace(-0.15, 0.15, 21))
def warp_crop(img, c, half, s):
    """以 c 为中心、把半径 half*s 的区域缩放到 2*half 见方（s>1 = 目标在这帧更大）"""
    n = int(2 * half); a = 1.0 / s
    M = np.array([[a, 0, half - a * c[0]], [0, a, half - a * c[1]]], np.float32)
    return cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
def ncc_at(T, img, c, half, s, srch):
    W = warp_crop(img, c, half + srch * s, s)            # 多裁一圈做平移搜索
    r = cv2.matchTemplate(W, T, cv2.TM_CCOEFF_NORMED)
    return float(r.max())
def match(Fa, ca, ha, Fb, cb, hb):
    if ha < 8: return np.nan, np.nan
    T = warp_crop(Fa, ca, ha, 1.0)
    if T.std() < 1e-3: return np.nan, np.nan
    srch = int(0.3 * ha) + 4; base = hb / ha
    sc = [ncc_at(T, Fb, cb, ha, base * s, srch) for s in COARSE]
    j = int(np.argmax(sc))
    fine = base * COARSE[j] * np.exp(np.linspace(-0.015, 0.015, 13))
    sf = [ncc_at(T, Fb, cb, ha, s, srch) for s in fine]
    i = int(np.clip(np.argmax(sf), 1, len(fine) - 2)); y0, y1, y2 = sf[i - 1], sf[i], sf[i + 1]
    den = y0 - 2 * y1 + y2; off = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
    ls = np.log(fine[i]) + np.clip(off, -1, 1) * (np.log(fine[1]) - np.log(fine[0]))
    return ls, max(sf)
W_ID, W_N = int(sys.argv[1]), int(sys.argv[2])
R = '@@MMCACHE@@/mav6d_cn_obj/train/'
P0 = pickle.load(open(R + os.environ.get('INST_IDX', 'index_pseudo_st2plus.pkl'), 'rb'))   # 11-06 无教师消融可换源
V = [int(i) for i in P0['valid_idx']]
dump = pickle.load(open(P0['pseudo']['dump'], 'rb'))['frames']
gtd = {(str(f['seq']), str(f['frame'])): np.asarray(f['gt'], np.float64).reshape(-1, 9)[0] for f in dump}
seqs = {}
for i in V:
    m = P0['metas'][i]; seqs.setdefault(m['seq'], []).append(i)
S = []
for si, (s, ids) in enumerate(sorted(seqs.items())):
    if si % W_N != W_ID: continue
    ids = sorted(ids, key=lambda i: int(P0['metas'][i]['frame'].split('.')[0])); n = len(ids)
    cls, sess, sid = s.split('/')
    path = lambda i: '@@MAV6D@@/%s/JPEGImages/%s/%s/%s' % (cls, sess, sid, P0['metas'][i]['frame'])
    Kr = np.asarray(P0['metas'][ids[0]]['K_raw'], np.float64)
    samp = [cv2.imread(path(ids[j]), 0) for j in np.linspace(0, n - 1, min(n, 61)).astype(int)]
    bg = np.median(np.stack(samp), 0).astype(np.float32)
    info = []
    for i in ids:
        p = np.asarray(P0['metas'][i]['boxes9d'], np.float64)[0]; g = gtd.get((s, P0['metas'][i]['frame']))
        c = np.array([Kr[0, 0] * p[0] / p[2] + Kr[0, 2], Kr[1, 1] * p[1] / p[2] + Kr[1, 2]])
        info.append((c, 0.8 * Kr[0, 0] * BODY / p[2], np.log(np.linalg.norm(p[:3])), np.log(np.linalg.norm(g[:3])) if g is not None else np.nan))
    F = {}
    def fg(t):
        if t not in F:
            F[t] = cv2.GaussianBlur(np.abs(cv2.imread(path(ids[t]), 0).astype(np.float32) - bg), (0, 0), 1.0)
        return F[t]
    pairs = []
    for t in range(n):
        for k in KS:
            if t + k < n:
                ls, q = match(fg(t), info[t][0], info[t][1], fg(t + k), info[t + k][0], info[t + k][1])
                if np.isfinite(ls): pairs.append((t, t + k, ls, q))
        F.pop(t - 1, None)
    S.append((s, np.array([x[2] for x in info]), np.array([x[3] for x in info]), pairs, ids))
    print('train', s, n, '帧', len(pairs), '对', flush=True)
pickle.dump(S, open(os.environ.get('INST_PAIRS', 'pairs2_train') + '_w%d.pkl' % W_ID, 'wb'))
