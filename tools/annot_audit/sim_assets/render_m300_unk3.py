# -*- coding: utf-8 -*-
"""m300（材质颜色全相同）按材质名伪彩色；unk3（单材质）方向光放大渲染。四视图 + 定量不对称统计。"""
import json, os
import cv2, numpy as np, torch
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes
BASE = 'E:/Open3DUAVDet/output/camnorm/annot_audit/assets_v2'
S = 640
dev = torch.device('cuda')
PAL = {'body': (0.35, 0.35, 0.35), 'leg': (0.6, 0.6, 0.6), 'glass': (0.1, 0.4, 1.0), 'plastic': (0.2, 0.8, 0.2), 'metal': (1.0, 0.55, 0.1),
       'light': (1.0, 0.95, 0.0), 'camera3': (0.9, 0.1, 0.9), 'cameralens': (0.9, 0.1, 0.9)}
d = json.load(open(os.path.join(BASE, 'assets_v2.json'), encoding='utf-8'))


def load(model):
    V, F, C, info = [], [], [], {}
    off = 0
    for comp in d['models'][model]['mesh_components']:
        if comp.get('visible') is not True:
            continue
        for s in comp['sections']:
            v = np.fromfile(s['files'] + '.v.f32', np.float32).reshape(-1, 3); t = np.fromfile(s['files'] + '.tri.i32', np.int32).reshape(-1, 3)
            key = next((k for k in PAL if k in str(s['slot_name']).lower()), None)
            col = np.array(PAL.get(key, (0.8, 0.8, 0.8)), np.float32)
            V.append(v); F.append(t + off); C.append(np.repeat(col[None], len(t), 0)); off += len(v)
            info.setdefault(key or str(s['slot_name']), []).append(v)
    return np.concatenate(V), np.concatenate(F), np.concatenate(C), {k: np.concatenate(v) for k, v in info.items()}


def render(V, F, C, view, ext, center=(0, 0)):
    maps = {'top': (V[:, 1], V[:, 0], V[:, 2]), 'from+X': (-V[:, 1], V[:, 2], V[:, 0]), 'from+Y': (V[:, 0], V[:, 2], V[:, 1]),
            'from-Y': (-V[:, 0], V[:, 2], -V[:, 1]), 'bottom': (V[:, 1], -V[:, 0], -V[:, 2])}
    u, w, dd = maps[view]
    u = u - center[0]; w = w - center[1]
    ndc = np.stack([-u / ext, w / ext, (dd.max() - dd) / (np.ptp(dd) + 1e-6) + 0.1], 1).astype(np.float32)
    mesh = Meshes(verts=[torch.from_numpy(ndc).to(dev)], faces=[torch.from_numpy(F.astype(np.int64)).to(dev)])
    p2f = rasterize_meshes(mesh, image_size=S, blur_radius=0.0, faces_per_pixel=1, bin_size=0)[0][0, ..., 0].cpu().numpy()
    # 方向光：用原始 actor 系法向，光从视线方向偏上方来
    tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    vd = {'top': (0, 0, 1), 'from+X': (1, 0, 0), 'from+Y': (0, 1, 0), 'from-Y': (0, -1, 0), 'bottom': (0, 0, -1)}[view]
    L = np.array(vd, float) * 0.8 + np.array([0.3, 0.2, 0.5]); L /= np.linalg.norm(L)
    shade = 0.25 + 0.75 * np.abs(n @ L)
    img = np.full((S, S, 3), 255, np.float32); m = p2f >= 0
    img[m] = C[p2f[m]] * shade[p2f[m], None] * 255
    return np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1].copy()


out = []
for model in ('matrix-300-RTK', 'drone-unk3'):
    V, F, C, info = load(model)
    ext = np.abs(V).max() * 1.05
    tiles = []
    for view, title in (('top', 'TOP up=+X right=+Y'), ('from+X', 'from +X right=-Y'), ('from+Y', 'from +Y right=+X'),
                        ('from-Y', 'from -Y right=-X'), ('bottom', 'BOTTOM up=-X right=+Y')):
        img = render(V, F, C, view, ext)
        cv2.putText(img, model + ' ' + title, (5, 18), 0, 0.5, (0, 0, 0), 1)
        tiles.append(cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA))
    out.append(np.hstack(tiles))
    print('==', model)
    for k, v in info.items():
        yp, yn = np.sum(v[:, 1] > 2), np.sum(v[:, 1] < -2)
        print('   %-14s n=%7d  y>0: %6d  y<0: %6d  mean y %.1f cm  mean z %.1f' % (k, len(v), yp, yn, v[:, 1].mean(), v[:, 2].mean()))
    # 底部（z 最低 25%、中心 |x|,|y|<机体 25%）质心 y：云台一般挂在前下方
    allv = np.concatenate(list(info.values()))
    r = np.abs(allv).max(0)
    low = allv[(allv[:, 2] < np.percentile(allv[:, 2], 25)) & (np.abs(allv[:, 0]) < 0.25 * r[0]) & (np.abs(allv[:, 1]) < 0.35 * r[1])]
    print('   bottom-center parts: n=%d mean y %.1f cm (y>0 %d / y<0 %d)' % (len(low), low[:, 1].mean(), np.sum(low[:, 1] > 0), np.sum(low[:, 1] < 0)))
cv2.imwrite(os.path.join(BASE, 'render', 'm300_unk3_detail.jpg'), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 90])
