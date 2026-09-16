# -*- coding: utf-8 -*-
"""两域机体坐标轴的高清近景核对（红 x / 绿 y / 蓝 z）：仿真从 1280x720 原图、MAV6D 从 1920x1080 原始 JPEG 取最近的若干帧，
各机型取表观最大的帧放大，人工看「红色 x 轴是否指向云台 / 机头」。

    python vis_body_axes_closeup.py --out ../output/camnorm/vis/body_axes_closeup
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
K_MAV = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.1, 533.9717], [0.0, 0.0, 1.0]])
D_MAV = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])


def proj(pts, K, D):
    uv, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(-1, 2)


def tile(img, K, D, box, title, size=360):
    Rm = R.from_euler('xyz', box[6:9]).as_matrix()
    c = (PROTO8 * box[3:6]) @ Rm.T + box[:3]
    L = 0.6 * float(max(box[3:6]))
    uv = proj(np.vstack([c, box[:3], box[:3] + Rm[:, 0] * L, box[:3] + Rm[:, 1] * L, box[:3] + Rm[:, 2] * L]), K, D)
    img = img.copy()
    for a, b in EDGES:
        cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), (0, 255, 255), 1, cv2.LINE_AA)
    o = tuple(np.int32(uv[8]))
    for k, col in ((9, (0, 0, 255)), (10, (0, 255, 0)), (11, (255, 0, 0))):
        cv2.arrowedLine(img, o, tuple(np.int32(uv[k])), col, 2, cv2.LINE_AA, tipLength=0.2)
    ext = max(np.ptp(uv[:8, 0]), np.ptp(uv[:8, 1]))
    h = int(max(40, 0.9 * ext))
    cu, cv_ = int(uv[8, 0]), int(uv[8, 1])
    pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT)
    crop = cv2.resize(pad[cv_:cv_ + 2 * h, cu:cu + 2 * h], (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.putText(crop, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per', type=int, default=4)
    ap.add_argument('--out', default='../output/camnorm/vis/body_axes_closeup')
    args = ap.parse_args()
    rows = []
    # ---- 仿真：每个机型取表观最大的若干帧（不同序列）----
    idx = pickle.load(open('E:/mmcache/indoor8cn/train/index.pkl', 'rb'))
    by_model = {}
    for i in idx['valid_idx'][::3]:
        m = idx['metas'][int(i)]
        for n, b in zip(m['names'], m['boxes9d']):
            app = 640.0 * float(max(b[3:6])) / float(b[2])
            by_model.setdefault(n, []).append((app, int(i), b.astype(np.float64)))
    for n in sorted(by_model):
        cands = sorted(by_model[n], key=lambda x: -x[0])
        seen, picks = set(), []
        for app, i, b in cands:
            seq = idx['metas'][i]['seq']
            if seq in seen:
                continue
            seen.add(seq)
            picks.append((app, i, b))
            if len(picks) == args.per:
                break
        tiles = []
        for app, i, b in picks:
            m = idx['metas'][i]
            img = cv2.imread(os.path.join('D:/data_collect', m['seq'], 'images_rgb', m['frame']))
            tiles.append(tile(img, np.asarray(m['K_raw'], np.float64), np.zeros(5), b, 'sim %s' % n))
        rows.append(np.concatenate(tiles, 1))
    # ---- MAV6D：每个机型取表观最大的若干帧（不同序列），原始带畸变 JPEG + 标定内参/畸变 ----
    midx = pickle.load(open('E:/mmcache/mav6d_cn/train/index.pkl', 'rb'))
    for cls in ('phantom4', 'mavic2'):
        cands = []
        for i in midx['valid_idx']:
            m = midx['metas'][int(i)]
            if m['cls'] != cls:
                continue
            b = m['boxes9d'][0].astype(np.float64)
            cands.append((1845.0 * 0.34 / b[2], int(i), b))
        cands.sort(key=lambda x: -x[0])
        seen, tiles = set(), []
        for app, i, b in cands:
            m = midx['metas'][i]
            if m['seq'] in seen:
                continue
            seen.add(m['seq'])
            c, scene, seq = m['seq'].split('/')
            img = cv2.imread(os.path.join('E:/MAV6D', c, 'JPEGImages', scene, seq, m['frame']))
            tiles.append(tile(img, K_MAV, D_MAV, b, 'MAV6D %s %s' % (cls, seq)))
            if len(tiles) == args.per:
                break
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out + '.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('写出', args.out + '.jpg', sheet.shape)


if __name__ == '__main__':
    main()
