# -*- coding: utf-8 -*-
"""[cleanfix A1] Window contamination of the CLEAN background banks (real train labels used ONLY for this post-hoc measurement).
Replica of gproto/wf24/diag_randwin.py + wf21/strict/diag_bank.py on the clean MAV6D chain srcx/in2mav:
  * replays the exact window-selection rules and RNG sequence of srcx/lib/refiner_x.py cmd_bank (det0 recs, conf>=.5, width 6-170 input px,
    sqrt(3)w square moved 1.8w in one of 8 directions, rng seed 0, keep <= 8000) and srcx/lib/ori_x.py cmd_bank (det1 recs, conf>=.6,
    window (1920/z)x(1080/z), z = clip(95/w2d, 1, 4), uniform position avoiding the 20 %-enlarged detected box, <= 20 tries, rng seed 0)
    -- no image is read; the replayed window count is checked against the saved bank sizes (4976 / 4420);
  * contamination of a window = max over GT UAVs of (area of the GT 2D box (projected 8 corners, raw K) inside the window / GT box area);
  * random baseline (= diag_randwin.py): same frames, same window size, uniform position in the frame (no avoidance), rng seed 0.
Reports share of windows with >=10 % / >=50 % of a UAV, by airframe."""
import pickle, json, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav'
OUT = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/A1'
TW, TH, WSC, ZMAX = 1920, 1080, 1920.0 / 512.0, 4.0
S0 = float(np.sqrt(3.0))
P8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
idx = pickle.load(open('E:/mmcache/mav6d_cn_obj/train/index.pkl', 'rb'))
M = {(m['seq'], m['frame']): m for m in idx['metas'] if m}


def bb(box, K):
    c = (P8 * box[3:6]) @ Rot.from_euler('xyz', box[6:9]).as_matrix().T + box[:3]
    q = (K @ c.T).T
    if (q[:, 2] <= 1e-3).any():
        return None
    uv = q[:, :2] / q[:, 2:3]
    return uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()


def frac(m, x0, y0, w, h):   # GT share inside window [x0,x0+w]x[y0,y0+h] (raw px)
    K = np.asarray(m['K_raw'], np.float64); best = 0.0
    for g in np.asarray(m['boxes9d'], np.float64).reshape(-1, 9):
        b = bb(g, K)
        if b is None:
            continue
        u0, u1, v0, v1 = b; a = max(1e-6, (u1 - u0) * (v1 - v0))
        iw = max(0.0, min(u1, x0 + w) - max(u0, x0)); ih = max(0.0, min(v1, y0 + h) - max(v0, y0))
        best = max(best, iw * ih / a)
    return best


def summ(rows):
    c = np.array([a for a, _ in rows]); f = np.array([b for _, b in rows])
    r = dict(n=len(f), ge10=100 * float((f >= .1).mean()), ge50=100 * float((f >= .5).mean()))
    for t in sorted(set(c)):
        r[t] = dict(n=int((c == t).sum()), ge10=100 * float((f[c == t] >= .1).mean()))
    return r


res = {}
rr = np.random.default_rng(0)   # random baseline RNG (as diag_randwin.py)
# ---------------- refiner (depth) bank: replay refiner_x.cmd_bank on det0
recs = pickle.load(open(X + '/recs/det0.pkl', 'rb'))['recs']['train']
rng = np.random.default_rng(0)
D = [(a, b) for a in (-1.8, 0, 1.8) for b in (-1.8, 0, 1.8) if (a, b) != (0, 0)]
win, rnd = [], []
for r in recs:
    if r['box'] is None or r['conf'] < 0.5:
        continue
    m = M[(r['seq'], r['frame'])]; K = np.asarray(m['K_raw'], np.float64)
    b = bb(r['box'], K)
    if b is None:
        continue
    w = b[1] - b[0]
    if not (6 <= w / WSC <= 170):
        continue
    cx, cy = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]); half = 0.5 * S0 * w
    for j in rng.permutation(len(D)):
        x, y = cx + D[j][0] * w, cy + D[j][1] * w
        if x - half >= 0 and y - half >= 0 and x + half <= TW and y + half <= TH:
            win.append((m, x - half, y - half, 2 * half)); break
    if 2 * half < TW - 1 and 2 * half < TH - 1:   # random baseline for every eligible frame (diag_randwin: no avoidance)
        x0 = rr.uniform(0, TW - 1 - 2 * half); y0 = rr.uniform(0, TH - 1 - 2 * half)
        rnd.append((m['cls'], frac(m, x0, y0, 2 * half, 2 * half)))
keep = np.sort(rng.choice(len(win), min(8000, len(win)), replace=False))
bank_n = np.load(X + '/ref/bank.npy', mmap_mode='r').shape[0]
assert len(keep) == bank_n, (len(keep), bank_n)
res['refiner_bank_clean'] = summ([(win[i][0]['cls'], frac(win[i][0], win[i][1], win[i][2], win[i][3], win[i][3])) for i in keep])
res['refiner_random'] = summ(rnd)
# ---------------- orientation bank: replay ori_x.cmd_bank on det1
D1 = pickle.load(open(X + '/recs/det1.pkl', 'rb'))['recs']['train']
rng = np.random.default_rng(0); win, rnd = [], []
for r in D1:
    if r['box'] is None or r['conf'] < 0.6:
        continue
    m = M[(r['seq'], r['frame'])]; Kr = np.asarray(m['K_raw'], np.float64)
    z = float(np.clip(95.0 / max(r['w2d'], 1e-3), 1.0, ZMAX)); w = TW / z; h = w * 288 / 512
    c = (P8 * r['box'][3:6]) @ Rot.from_euler('xyz', r['box'][6:9]).as_matrix().T + r['box'][:3]
    q = (Kr @ c.T).T; uv = q[:, :2] / q[:, 2:3]
    bx0, by0 = uv.min(0); bx1, by1 = uv.max(0)
    ex, ey = 0.2 * (bx1 - bx0), 0.2 * (by1 - by0); bx0, bx1, by0, by1 = bx0 - ex, bx1 + ex, by0 - ey, by1 + ey
    if w >= TW:
        continue
    for _ in range(20):
        x0, y0 = rng.uniform(0, TW - w), rng.uniform(0, TH - h)
        if x0 > bx1 or x0 + w < bx0 or y0 > by1 or y0 + h < by0:
            win.append((m['cls'], frac(m, x0, y0, w, h))); break
    x0 = rr.uniform(0, TW - w); y0 = rr.uniform(0, TH - h)
    rnd.append((m['cls'], frac(m, x0, y0, w, h)))
ob_n = np.load('D:/S2MUAV3D/srcx_work/in2mav/ou_bank.npy', mmap_mode='r').shape[0]
assert len(win) == ob_n, (len(win), ob_n)
res['orientation_bank_clean'] = summ(win)
res['orientation_random'] = summ(rnd)
print(json.dumps(res, indent=1))
json.dump(res, open(OUT + '/a1_contam.json', 'w'), indent=1)
