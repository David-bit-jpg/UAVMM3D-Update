# -*- coding: utf-8 -*-
"""wf21/strict 诊断（真值只诊断，不参与任何选择）：背景库里残留真机的比例。
按 xpaste_bank_sf.py / xpaste_bank_sf2.py 的选窗规则逐条重放（不读图，随机数序列与原脚本一致），
对每个入库窗口量"真值无人机 2D 框落在窗口视野里的面积比例"，报 ≥10% / ≥50% 的窗口占比（按机型）。
  旧库 bank_bg_sf.npy：选窗检测 = irloc_train_recs/train_P0_s3.pkl（纯仿真检测器，带热图时序背景 causal100）
  新库 bank_bg_sf2.npy：选窗检测 = wf21/strict/recs/wsfnb_train.pkl（全严格检测器，单帧）"""
import pickle, sys, json
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, '@@ROOT@@'); sys.path.insert(0, G)
from scipy.spatial.transform import Rotation as R
from uavdet3d.utils import pose_eval as pe
d = pickle.load(open('@@MMCACHE@@/mav6d_cn_obj/train/index.pkl', 'rb'))
key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(d['metas'])}
H, W = 288, 512          # 缓存图分辨率（K_in 与之对应）
S0 = float(np.sqrt(1.2 * 2.5))
HALF = 0.5 * S0
DIRS = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]


def box2d(g, K):
    c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
    q = (K @ c.T).T
    if (q[:, 2] <= 1e-3).any():
        return None
    uv = q[:, :2] / q[:, 2:3]
    return uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()


def replay(recs):
    rng = np.random.default_rng(0)
    win = []
    for r in recs:
        if len(r['pred']) == 0:
            continue
        pi = int(np.argmax(r['conf']))
        if r['conf'][pi] < 0.5:
            continue
        g = r['pred'][pi]
        if g[2] <= 1e-3:
            continue
        i = key[(r['seq_id'], r['frame_id'])]
        K = np.asarray(d['metas'][i]['K_in'], np.float64)
        b = box2d(g, K)
        if b is None:
            continue
        u0, u1, v0, v1 = b; w = u1 - u0
        if not (6.0 <= w <= 170.0):
            continue
        cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        ok = None
        for k in rng.permutation(len(DIRS)):
            a, bb = DIRS[k]
            x, y = cx + 1.8 * w * a, cy + 1.8 * w * bb
            if x - HALF * w >= 0 and x + HALF * w <= W - 1 and y - HALF * w >= 0 and y + HALF * w <= H - 1:
                ok = (x, y); break
        if ok is None:
            continue
        gt = np.asarray(r['gt'], np.float64).reshape(-1, 9)
        win.append((r['seq_id'], ok[0], ok[1], w, gt, K))
    sel = np.sort(rng.choice(len(win), min(8000, len(win)), replace=False))
    return [win[i] for i in sel]


def frac(win):
    out = []
    for s, x, y, w, gt, K in win:
        h = HALF * w
        best = 0.0
        for g in gt:
            b = box2d(g, K)
            if b is None:
                continue
            u0, u1, v0, v1 = b
            a = max(1e-6, (u1 - u0) * (v1 - v0))
            iw = max(0.0, min(u1, x + h) - max(u0, x - h)); ih = max(0.0, min(v1, y + h) - max(v0, y - h))
            best = max(best, iw * ih / a)
        out.append((s.split('/')[0], best))
    return out


res = {}
for name, path in (('旧库 sf（P0_s3 带时序背景检测选窗）', G + '/irloc_train_recs/train_P0_s3.pkl'),
                   ('新库 sf2（wsfnb 单帧检测选窗）', G + '/gproto/wf21/strict/recs/wsfnb_train.pkl')):
    recs = pickle.load(open(path, 'rb'))['records']
    fr = frac(replay(recs))
    ty = np.array([a for a, _ in fr]); v = np.array([b for _, b in fr])
    row = dict(n=len(v), ge10=100 * float(np.mean(v >= 0.1)), ge50=100 * float(np.mean(v >= 0.5)))
    for t in ('mavic2', 'phantom4'):
        m = ty == t
        row[t] = dict(n=int(m.sum()), ge10=100 * float(np.mean(v[m] >= 0.1)), ge50=100 * float(np.mean(v[m] >= 0.5)))
    res[name] = row
    print(name, json.dumps(row, ensure_ascii=False), flush=True)
json.dump(res, open(G + '/gproto/wf21/strict/diag_bank.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
