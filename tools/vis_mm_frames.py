# -*- coding: utf-8 -*-
"""三模态同屏可视化：RGB | IR | LiDAR 投影，各自叠上 3D 框。按键翻页。

    y / →      下一帧            t / ←      上一帧
    Y          向后跳 20 帧       T          向前跳 20 帧
    o          切换 LiDAR 帧偏移（配置值 <-> 0），用来肉眼核对配对是否正确
    b          切换 LiDAR 面板背景（黑底 / 叠在 RGB 上）
    s          把当前拼图存成 PNG
    q / Esc    退出

每个模态都用【自己的】框文件和内参投影（boxes_rgb + rgb 内参、boxes_ir + ir 内参），
不做数据集里那个「按目标中心平移 IR」的配准 —— 这里要看的是几何真相：
三台相机横向相距约 2.3 m，近处目标在 RGB 和 IR 上有明显视差是正常的。

LiDAR 面板：点云经 lidar 外参 -> 世界 -> RGB 相机 -> 像素，按深度着色（近红远蓝），
tag==1（打在无人机上的点）另用品红大点标出。框用 boxes_rgb（同一相机）。

框颜色：绿 = 距离 <= --max-range 且整框在画面内（即筛选脚本认可的目标）；
        橙 = 其余（更远、或部分出画）。标签写 机型 与 距离。

用法：
    python tools/vis_mm_frames.py --list cfgs/subsets/mm20/near_train.txt
    python tools/vis_mm_frames.py --list ... --dump 8 --dump-dir ../output/vis_mm   # 无窗口，只存图
"""
import argparse
import os
import pickle
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_near_subset import frame_sort_key   # noqa: E402

CARLA_TO_OPENCV = np.array([[0, 1, 0, 0],
                            [0, 0, -1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float64)
GREEN, ORANGE, MAGENTA, WHITE, GRAY = (60, 220, 60), (0, 160, 255), (255, 0, 255), (255, 255, 255), (150, 150, 150)


# --------------------------------------------------------------------------- #
def read_list(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            p = ln.split()
            rows.append((p[0], p[1]))
    return rows


class SeqCache(object):
    """每条序列的内外参、帧列表只读一次。"""

    def __init__(self, root):
        self.root = root
        self.cache = {}

    def get(self, seq):
        if seq in self.cache:
            return self.cache[seq]
        base = os.path.join(self.root, seq)
        with open(os.path.join(base, 'im_info.pkl'), 'rb') as f:
            im_info = pickle.load(f)
        lidar_ext = np.eye(4)
        lidar_noise = -1.0
        p = os.path.join(base, 'lidar_radar_info.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                lr = pickle.load(f)
            lidar_ext = np.array(lr['lidars'][0]['extrinsic'], dtype=np.float64)
            # 逐序列随机的测距噪声 {0, 2.5, 3.5, 4.5} m —— 决定深度能不能信
            lidar_noise = float(lr['lidars'][0]['attributes'].get('NoiseStdDev', -1.0))
        frames = sorted([f for f in os.listdir(os.path.join(base, 'images_rgb')) if f.endswith('.png')],
                        key=frame_sort_key)
        meta = {
            'K': {m: np.array(im_info[m]['intrinsic'], dtype=np.float64) for m in ('rgb', 'ir', 'dvs')},
            'E': {m: np.array(im_info[m]['extrinsic'], dtype=np.float64) for m in ('rgb', 'ir', 'dvs')},
            'lidar_ext': lidar_ext,
            'lidar_noise': lidar_noise,
            'frames': frames,
            'index': {f: i for i, f in enumerate(frames)},
        }
        self.cache[seq] = meta
        return meta


def load_boxes(path):
    """boxes_<modal>/*.pkl -> [(name, (8,3) 该相机 OpenCV 系角点)]"""
    if not os.path.exists(path):
        return []
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    out = []
    for row in raw:
        if isinstance(row[0], str):
            name, pts = row[0], np.array(row[1:], dtype=np.float64).reshape(8, 3)
        else:
            name, pts = '?', np.array(row, dtype=np.float64).reshape(8, 3)
        out.append((name, pts))
    return out


def short_name(name):
    """BP_Drone01_Matrice-600-Pro_3 -> Matrice-600-Pro"""
    parts = name.split('_')
    return parts[2] if len(parts) >= 3 else name


def box_edges(pts):
    """不依赖角点顺序：每个角点连它在 3D 里最近的 3 个角点，正好是长方体的 12 条边。"""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    edges = set()
    for i in range(8):
        for j in np.argsort(d[i])[1:4]:
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def draw_box(img, pts_cam, K, max_range, thickness=2):
    """pts_cam: 该相机 OpenCV 系的 8 角点。返回 (是否合格, 距离)。"""
    h, w = img.shape[:2]
    c = pts_cam.mean(axis=0)
    r = float(np.linalg.norm(c))
    if np.any(pts_cam[:, 2] <= 1e-6):
        return False, r
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)).all()
    qualified = bool(inside and r <= max_range)
    if np.any(np.abs(uv) > 20 * max(w, h)):
        return qualified, r
    color = GREEN if qualified else ORANGE
    uvi = np.round(uv).astype(int)
    for i, j in box_edges(pts_cam):
        cv2.line(img, tuple(uvi[i]), tuple(uvi[j]), color, thickness, cv2.LINE_AA)
    return qualified, r


def label_box(img, pts_cam, K, text, color):
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x, y = int(uv[:, 0].min()), int(uv[:, 1].min()) - 6
    x = max(0, min(x, img.shape[1] - 200))
    y = max(14, y)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def project_lidar(npy_path, lidar_ext, cam_ext, K, w, h):
    """返回 (uv (M,2), depth (M,), tag (M,))，只含画面内、相机前方的点。"""
    if not os.path.exists(npy_path):
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    P = np.load(npy_path)
    xyz = P[:, :3].astype(np.float64)
    tag = P[:, 4]
    world = (lidar_ext @ np.c_[xyz, np.ones(len(xyz))].T).T
    cam = (np.linalg.inv(cam_ext) @ world.T).T
    cv = (CARLA_TO_OPENCV @ cam.T).T[:, :3]
    m = cv[:, 2] > 0.1
    cv, tag = cv[m], tag[m]
    uv = (K @ cv.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    ins = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[ins], cv[ins, 2], tag[ins]


def render_lidar(canvas, uv, depth, tag, max_depth=60.0):
    """按深度着色画点（TURBO: 近红远蓝），tag==1 的点品红加粗。"""
    if len(uv) == 0:
        return canvas, 0
    z = np.clip(depth / max_depth, 0, 1)
    cmap = cv2.applyColorMap((255 - (z * 255)).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    uvi = np.round(uv).astype(int)
    h, w = canvas.shape[:2]
    uvi[:, 0] = np.clip(uvi[:, 0], 0, w - 1)
    uvi[:, 1] = np.clip(uvi[:, 1], 0, h - 1)
    # 远的先画、近的后画，近点盖住远点
    order = np.argsort(-depth)
    canvas[uvi[order, 1], uvi[order, 0]] = cmap[order]
    hit = tag == 1
    for x, y in uvi[hit]:
        cv2.circle(canvas, (int(x), int(y)), 2, MAGENTA, -1)
    return canvas, int(hit.sum())


def put_header(panel, lines):
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return panel


# --------------------------------------------------------------------------- #
def compose(root, seq, frame, cache, max_range, lidar_offset, lidar_bg_rgb, scale):
    meta = cache.get(seq)
    base = os.path.join(root, seq)
    stem = os.path.splitext(frame)[0]

    rgb = cv2.imread(os.path.join(base, 'images_rgb', frame), cv2.IMREAD_COLOR)
    ir = cv2.imread(os.path.join(base, 'images_ir', frame), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(os.path.join(base, 'images_rgb', frame))
    h, w = rgb.shape[:2]
    if ir is None:
        ir = np.zeros((h, w), np.uint8)
    ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)

    # LiDAR 配对帧
    idx = meta['index'].get(frame, None)
    li = None if idx is None else min(idx + lidar_offset, len(meta['frames']) - 1)
    lidar_frame = meta['frames'][li] if li is not None else frame
    npy = os.path.join(base, 'lidar_1', os.path.splitext(lidar_frame)[0] + '.npy')
    uv, depth, tag = project_lidar(npy, meta['lidar_ext'], meta['E']['rgb'], meta['K']['rgb'], w, h)
    lidar_canvas = rgb.copy() if lidar_bg_rgb else np.zeros_like(rgb)
    if not lidar_bg_rgb:
        lidar_canvas[:] = (25, 25, 25)
    lidar_canvas, n_hit = render_lidar(lidar_canvas, uv, depth, tag)

    # 框：每个模态用自己的框文件 + 内参
    panels = {}
    n_q = 0
    for modal, img in (('rgb', rgb), ('ir', ir)):
        boxes = load_boxes(os.path.join(base, 'boxes_%s' % modal, stem + '.pkl'))
        for name, pts in boxes:
            q, r = draw_box(img, pts, meta['K'][modal], max_range)
            label_box(img, pts, meta['K'][modal], '%s %.1fm' % (short_name(name), r), GREEN if q else ORANGE)
            if modal == 'rgb' and q:
                n_q += 1
        panels[modal] = img
    for name, pts in load_boxes(os.path.join(base, 'boxes_rgb', stem + '.pkl')):
        draw_box(lidar_canvas, pts, meta['K']['rgb'], max_range, thickness=1)

    parts = seq.split('/')
    coll = parts[0]
    weather = parts[3] if len(parts) > 3 else '?'
    put_header(panels['rgb'], ['RGB   %s | %s' % (coll, weather),
                               'frame %s   qualified %d   (green<=%.0fm & in view)' % (stem, n_q, max_range)])
    put_header(panels['ir'], ['IR    (own extrinsic; parallax vs RGB is real)'])
    put_header(lidar_canvas, ['LiDAR -> RGB cam   offset %+d (frame %s)   range noise std %.1f m'
                              % (lidar_offset, os.path.splitext(lidar_frame)[0], meta['lidar_noise']),
                              '%d pts in view   %d tagged drone (magenta)' % (len(uv), n_hit)])

    tiles = [panels['rgb'], panels['ir'], lidar_canvas]
    if scale != 1.0:
        tiles = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tiles]
    sep = np.full((tiles[0].shape[0], 4, 3), 90, np.uint8)
    return np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default='cfgs/subsets/mm20/near_train.txt')
    ap.add_argument('--root', default='E:/data_collect')
    ap.add_argument('--max-range', type=float, default=20.0)
    ap.add_argument('--lidar-offset', type=int, default=6)
    ap.add_argument('--scale', type=float, default=0.5, help='每个面板的缩放，0.5 -> 三块拼起来 1920 宽')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--shuffle', action='store_true', help='随机顺序浏览，便于抽查各天气')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--filter', default=None, help='只看 seq 路径含该子串的帧，如 fog_night')
    ap.add_argument('--dump', type=int, default=0, help='无窗口：均匀抽 N 帧存 PNG 后退出')
    ap.add_argument('--dump-dir', default='../output/vis_mm')
    ap.add_argument('--lidar-bg-rgb', action='store_true', help='LiDAR 面板叠在 RGB 上（默认黑底）')
    args = ap.parse_args()

    rows = read_list(args.list)
    if args.filter:
        rows = [r for r in rows if args.filter in r[0]]
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if not rows:
        print('列表为空'); return 1
    print('共 %d 帧' % len(rows))
    cache = SeqCache(args.root)
    lidar_offset = args.lidar_offset
    lidar_bg = args.lidar_bg_rgb

    if args.dump > 0:
        os.makedirs(args.dump_dir, exist_ok=True)
        picks = np.linspace(0, len(rows) - 1, args.dump).round().astype(int)
        for k, i in enumerate(picks):
            seq, fr = rows[i]
            img = compose(args.root, seq, fr, cache, args.max_range, lidar_offset, lidar_bg, args.scale)
            op = os.path.join(args.dump_dir, '%02d_%s_%s.jpg' % (k, seq.replace('/', '_')[:60], os.path.splitext(fr)[0]))
            cv2.imwrite(op, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print('写出', op)
        return 0

    i = max(0, min(args.start, len(rows) - 1))
    win = 'RGB | IR | LiDAR   (t/y 翻页, o 偏移, b 背景, s 存图, q 退出)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        seq, fr = rows[i]
        img = compose(args.root, seq, fr, cache, args.max_range, lidar_offset, lidar_bg, args.scale)
        cv2.putText(img, '%d / %d' % (i + 1, len(rows)), (img.shape[1] - 150, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.imshow(win, img)
        k = cv2.waitKeyEx(0)
        if k in (27, ord('q')):
            break
        elif k in (ord('y'), 2555904, 32):          # y / → / space
            i = (i + 1) % len(rows)
        elif k in (ord('t'), 2424832):              # t / ←
            i = (i - 1) % len(rows)
        elif k == ord('Y'):
            i = (i + 20) % len(rows)
        elif k == ord('T'):
            i = (i - 20) % len(rows)
        elif k == ord('o'):
            lidar_offset = 0 if lidar_offset != 0 else args.lidar_offset
        elif k == ord('b'):
            lidar_bg = not lidar_bg
        elif k == ord('s'):
            os.makedirs(args.dump_dir, exist_ok=True)
            op = os.path.join(args.dump_dir, 'snap_%s_%s.jpg' % (seq.replace('/', '_')[:60], os.path.splitext(fr)[0]))
            cv2.imwrite(op, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print('存图', op)
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
