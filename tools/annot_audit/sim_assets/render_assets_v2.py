# -*- coding: utf-8 -*-
"""离线渲染 UE 导出的无人机资产（dump_drone_assets_v2.py 的输出），每个机型一张四视图，按材质颜色着色。

坐标：UE actor 系（X 前 / Y 右 / Z 上，左手系，cm）。正交投影，z-buffer（pytorch3d rasterize_meshes）。
  顶视（从 +Z 往下看）：图上 = +X（标签前），图右 = +Y
  从 +X 看（站在标签「正前方」看向它）：图右 = -Y，图上 = +Z
  从 +Y 看（站在 actor 右侧）：图右 = +X，图上 = +Z
  从 -Y 看（站在 actor 左侧）：图右 = -X，图上 = +Z
黄色 = BoundingCheck 盒；顶视画红箭头 +X（标签机头）、绿箭头 +Y；并标出颜色鲜艳/名字带 glass、lens、light 的分段位置。

    python tools/annot_audit/sim_assets/render_assets_v2.py
"""
import json
import os

import cv2
import numpy as np
import torch
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes

BASE = 'E:/Open3DUAVDet/output/camnorm/annot_audit/assets_v2'
S = 520
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(rec):
    V, F, C, marks = [], [], [], []
    off = 0
    for comp in rec['mesh_components']:
        if comp.get('visible') is not True:
            continue
        for s in comp.get('sections', []):
            fb = s.get('files')
            if not fb or not os.path.exists(fb + '.v.f32'):
                continue
            v = np.fromfile(fb + '.v.f32', dtype=np.float32).reshape(-1, 3)
            t = np.fromfile(fb + '.tri.i32', dtype=np.int32).reshape(-1, 3)
            if len(v) == 0 or len(t) == 0:
                continue
            mat = s.get('material', {})
            par = (mat.get('vector_params') or {}).get('Param')
            col = np.array(par[:3], np.float32) if isinstance(par, list) else np.array([0.55, 0.55, 0.55], np.float32)
            name = (str(s.get('slot_name')) + ' ' + str(mat.get('path'))).lower()
            if 'prepelar' in name:
                col = np.array([0.35, 0.35, 0.35], np.float32)
            V.append(v); F.append(t + off); C.append(np.repeat(col[None], len(t), 0)); off += len(v)
            sat = col.max() - col.min()
            if sat > 0.35 or any(k in name for k in ('glass', 'lens', 'light')):
                marks.append((name.split(' ')[0][:22], v.mean(0), col, len(v)))
    return np.concatenate(V), np.concatenate(F), np.concatenate(C), marks


def render(V, F, C, view, ext):
    if view == 'top':
        u, w, d = V[:, 1], V[:, 0], V[:, 2]          # 右 +Y，上 +X，朝向观察者 = +Z
    elif view == 'from+X':
        u, w, d = -V[:, 1], V[:, 2], V[:, 0]
    elif view == 'from+Y':
        u, w, d = V[:, 0], V[:, 2], V[:, 1]
    else:  # from-Y
        u, w, d = -V[:, 0], V[:, 2], -V[:, 1]
    # pytorch3d NDC: +x 向左、+y 向上、z 越小越近
    ndc = np.stack([-u / ext, w / ext, (d.max() - d) / (np.ptp(d) + 1e-6) + 0.1], 1).astype(np.float32)
    mesh = Meshes(verts=[torch.from_numpy(ndc).to(dev)], faces=[torch.from_numpy(F.astype(np.int64)).to(dev)])
    p2f, zbuf, _, _ = rasterize_meshes(mesh, image_size=S, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=False)
    p2f = p2f[0, ..., 0].cpu().numpy()
    tri = np.stack([ndc[F[:, 0]], ndc[F[:, 1]], ndc[F[:, 2]]], 1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    shade = 0.35 + 0.65 * np.abs(nrm[:, 2])
    img = np.full((S, S, 3), 255, np.float32)
    m = p2f >= 0
    img[m] = (C[p2f[m]] * shade[p2f[m], None]) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1].copy()   # RGB -> BGR
    return img


def to_px(u, w, ext):
    return int((u / ext + 1) / 2 * (S - 1)), int((1 - (w / ext + 1) / 2) * (S - 1))


def main():
    d = json.load(open(os.path.join(BASE, 'assets_v2.json'), encoding='utf-8'))
    os.makedirs(os.path.join(BASE, 'render'), exist_ok=True)
    sheets = []
    for model, rec in d['models'].items():
        V, F, C, marks = load_model(rec)
        ext = np.abs(V).max() * 1.08
        box = rec['boxes'][0]
        he = np.array(box['unscaled_extent_cm']) * np.array(box['world_scale'])
        bc = np.array(box['world_location'])
        tiles = []
        for view, title in (('top', 'TOP (up=+X label front, right=+Y)'), ('from+X', 'from +X (label front)  right=-Y'),
                            ('from+Y', 'from +Y (actor right)  right=+X'), ('from-Y', 'from -Y (actor left)  right=-X')):
            img = render(V, F, C, view, ext)
            if view == 'top':
                x0, y0 = to_px(bc[1] - he[1], bc[0] + he[0], ext); x1, y1 = to_px(bc[1] + he[1], bc[0] - he[0], ext)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 1)
                o = to_px(0, 0, ext)
                cv2.arrowedLine(img, o, to_px(0, ext * 0.8, ext), (0, 0, 255), 2, tipLength=0.08)
                cv2.arrowedLine(img, o, to_px(ext * 0.8, 0, ext), (0, 180, 0), 2, tipLength=0.08)
                cv2.putText(img, '+X', to_px(-ext * 0.12, ext * 0.85, ext), 0, 0.6, (0, 0, 255), 2)
                cv2.putText(img, '+Y', to_px(ext * 0.82, -ext * 0.1, ext), 0, 0.6, (0, 150, 0), 2)
                for nm, cen, col, n in sorted(marks, key=lambda x: -x[3])[:6]:
                    p = to_px(cen[1], cen[0], ext)
                    cv2.circle(img, p, 6, (255, 0, 255), 2)
                    cv2.putText(img, '%s(%.2f,%.2f,%.2f)' % (nm[:14], col[0], col[1], col[2]), (p[0] + 7, p[1] - 4), 0, 0.33, (160, 0, 160), 1)
            cv2.putText(img, title, (5, 16), 0, 0.45, (0, 0, 0), 1)
            tiles.append(img)
        row = np.hstack(tiles)
        head = np.full((34, row.shape[1], 3), 40, np.uint8)
        cv2.putText(head, '%s   body relYaw %.1f  scale %s  box %s m' % (
            model, [c for c in rec['mesh_components'] if c['name'] == 'Drone Body'][0]['relative_rotation']['yaw'],
            [round(x, 4) for x in [c for c in rec['mesh_components'] if c['name'] == 'Drone Body'][0]['world_scale']],
            np.round(2 * he / 100, 3).tolist()), (8, 23), 0, 0.6, (255, 255, 255), 1)
        sheet = np.vstack([head, row])
        cv2.imwrite(os.path.join(BASE, 'render', '%s.jpg' % model), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheets.append(sheet)
        print(model, 'verts', len(V), 'tris', len(F), 'marks', [(m[0], np.round(m[1], 1).tolist()) for m in marks][:6], flush=True)


if __name__ == '__main__':
    main()
