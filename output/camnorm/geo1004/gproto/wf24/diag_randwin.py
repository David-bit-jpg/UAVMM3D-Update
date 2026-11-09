# -*- coding: utf-8 -*-
"""11-09 信息泄露复核（真值只诊断）：如果贴图背景窗"不避开"真实无人机、只在同一帧里随机取窗，窗里会有多少真机？
与现做法（用模型自己的单帧检测避开）对比。窗口大小与两份背景库相同：测距库 √3·w 见方；朝向库 (512/z)×(288/z)，z = clip(95/w, 1, 4)。坐标为 512×288 缓存图。"""
import pickle, sys, json
import numpy as np
sys.path.insert(0, '@@ROOT@@/output/camnorm/geo1004/gproto/wf21/strict')
import diag_bank as DB
recs = pickle.load(open('@@ROOT@@/output/camnorm/geo1004/gproto/wf21/strict/recs/wsfnb_train.pkl', 'rb'))['records']
H, W = DB.H, DB.W
rng = np.random.default_rng(0)
out = {'dist': [], 'ori': []}
for r in recs:
    if len(r['pred']) == 0:
        continue
    pi = int(np.argmax(r['conf']))
    if r['conf'][pi] < 0.5:
        continue
    g = r['pred'][pi]
    i = DB.key[(r['seq_id'], r['frame_id'])]
    K = np.asarray(DB.d['metas'][i]['K_in'], np.float64)
    b = DB.box2d(g, K) if g[2] > 1e-3 else None
    if b is None:
        continue
    w = b[1] - b[0]
    if not (6.0 <= w <= 170.0):
        continue
    gt = np.asarray(r['gt'], np.float64).reshape(-1, 9)
    cls = r['seq_id'].split('/')[0]
    z = float(np.clip(95.0 / w, 1.0, 4.0))
    for kind, ww, hh in (('dist', 2 * DB.HALF * w, 2 * DB.HALF * w), ('ori', W / z, H / z)):
        if ww >= W - 1 or hh >= H - 1:
            continue
        x0 = rng.uniform(0, W - 1 - ww); y0 = rng.uniform(0, H - 1 - hh)
        best = 0.0
        for gg in gt:
            bb = DB.box2d(gg, K)
            if bb is None:
                continue
            u0, u1, v0, v1 = bb
            a = max(1e-6, (u1 - u0) * (v1 - v0))
            iw = max(0.0, min(u1, x0 + ww) - max(u0, x0)); ih = max(0.0, min(v1, y0 + hh) - max(v0, y0))
            best = max(best, iw * ih / a)
        out[kind].append((cls, best))
res = {}
for kind, v in out.items():
    c = np.array([x for x, _ in v]); f = np.array([y for _, y in v])
    res[kind] = dict(n=len(v), ge10=float((f >= .1).mean()), ge50=float((f >= .5).mean()),
                     mavic2_ge10=float((f[c == 'mavic2'] >= .1).mean()), phantom4_ge10=float((f[c == 'phantom4'] >= .1).mean()))
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open('@@ROOT@@/output/camnorm/geo1004/gproto/wf24/diag_randwin.json', 'w'), indent=1)
