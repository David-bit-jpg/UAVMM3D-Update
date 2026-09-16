# -*- coding: utf-8 -*-
"""从 UE 蓝图导出（assets_v2）逐机型量真实渲染尺寸，与官方规格比：包围盒、网格外廓、电机轴距、机身高度。"""
import json, os
import numpy as np
from scipy.spatial.distance import pdist, squareform
BASE = 'E:/Open3DUAVDet/output/camnorm/annot_audit/assets_v2'
d = json.load(open(os.path.join(BASE, 'assets_v2.json'), encoding='utf-8'))
SPEC = {  # 官方：轴距 mm / 外形 mm（口径见备注）
    'DJI-phantom4': (350, None, '轴距 350'),
    'DJI-mavic-mini': (213, (245, 290, 55), '对角 213；带桨展开 245x290x55'),
    'DJI-avata2': (None, (185, 212, 64), '185x212x64'),
    'Matrice-600-Pro': (1133, (1668, 1518, 727), '轴距 1133；带桨 1668x1518x727'),
    'm210-rtk': (643, (887, 880, 408), '轴距 643；展开 887x880x408'),
    'matrix-300-RTK': (895, (810, 670, 430), '轴距 895；展开(不含桨) 810x670x430'),
    'drone-unk3': (None, None, '型号不明'),
}
for m, rec in d['models'].items():
    V = []
    for c in rec['mesh_components']:
        if c.get('visible') is not True or 'pCamera' in c['name']:
            continue
        for s in c['sections']:
            v = np.fromfile(s['files'] + '.v.f32', np.float32).reshape(-1, 3)
            V.append(v)
    V = np.concatenate(V).astype(np.float64) / 100.0     # cm -> m，actor 系
    box = rec['boxes'][0]; bl = 2 * np.array(box['unscaled_extent_cm']) * np.array(box['world_scale']) / 100
    aabb = V.max(0) - V.min(0)
    # 桨盘高度层：z 最高 12% 的点里，按方位角分扇区找桨；每个桨取最远两点中点 = 桨毂
    z = V[:, 2]; top = V[z > np.percentile(z, 88)]
    r = np.hypot(top[:, 0], top[:, 1]); top = top[r > 0.35 * r.max()]
    ang = np.degrees(np.arctan2(top[:, 1], top[:, 0]))
    nrot = 6 if m == 'Matrice-600-Pro' else 4
    # 按角度直方图峰找桨中心方位
    h, e = np.histogram(ang, bins=72, range=(-180, 180))
    hubs = []
    order = np.argsort(-h)
    used = []
    for i in order:
        a0 = (e[i] + e[i + 1]) / 2
        if any(abs((a0 - u + 180) % 360 - 180) < 360 / nrot * 0.6 for u in used):
            continue
        used.append(a0)
        if len(used) == nrot:
            break
    for a0 in used:
        sel = top[np.abs((ang - a0 + 180) % 360 - 180) < 360 / nrot / 2]
        if len(sel) < 20:
            continue
        sub = sel[np.random.RandomState(0).choice(len(sel), min(len(sel), 1500), replace=False)]
        D = squareform(pdist(sub[:, :2]))
        i, j = np.unravel_index(np.argmax(D), D.shape)
        hubs.append(((sub[i, :2] + sub[j, :2]) / 2, D[i, j]))
    hc = np.array([h_[0] for h_ in hubs]); tip = np.array([h_[1] for h_ in hubs])
    rad = np.linalg.norm(hc, axis=1)
    wheelbase = 2 * np.median(rad) if len(hc) else np.nan
    body_h = np.percentile(z, 88) - z.min()     # 桨盘以下（含起落架）
    wb_spec, dims_spec, note = SPEC[m]
    k_wb = wheelbase * 1000 / wb_spec if wb_spec else np.nan
    print('== %-16s 包围盒 %s m | 网格外廓(含桨) %s m | 电机中心半径 %s m -> 轴距 %.3f m | 桨尖跨度中位 %.3f m | 桨盘以下高度 %.3f m' % (
        m, np.round(bl, 3), np.round(aabb, 3), np.round(rad, 3), wheelbase, np.median(tip) if len(tip) else np.nan, body_h))
    extra = ''
    if dims_spec:
        extra = ' | 外廓比（按规格口径，粗略）%s' % np.round(np.sort(aabb[:2])[::-1] * 1000 / np.sort(np.array(dims_spec[:2]))[::-1], 2).tolist()
    print('   官方 %s | 轴距比 %.2f%s' % (note, k_wb, extra))
