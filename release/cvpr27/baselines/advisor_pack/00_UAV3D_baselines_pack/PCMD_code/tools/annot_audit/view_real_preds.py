# -*- coding: utf-8 -*-
"""翻页看纯仿真模型在 MAV6D 真实 test 上的结果（2026-09-22）：左 = 真值，右 = 预测，两张大图并排、不叠画。

每张图画 3D 框 + 朝向：
    框边     左绿（真值）/ 右橙（预测）
    黄色面   机头那一面（机体 +X），红箭头 = 机头方向，蓝短线 = 机体朝上
    罗盘     俯视图（相机坐标，罗盘下方是相机，上方 = 远离相机），箭头 = 机头的水平朝向
    放大小窗 目标附近放大（两边窗口一样）；小窗和罗盘都放在目标的另一侧
图像 = MAV6D 原图 1920x1080 按标定去畸变（与训练缓存同一个针孔相机），原图找不到时退回 512x288 缓存图。

按键（先点一下窗口）：
    t / y        上一帧 / 下一帧（← → 也行）
    T / Y        后退 / 前进 24 帧（1 秒）
    g / h        上一段 / 下一段序列
    z            主图切换 整幅 / 放大
    s            存当前画面 PNG（output/camnorm/viewer/shots/）
    q / Esc      退出

    python annot_audit/view_real_preds.py [--cls phantom4] [--sort yaw] [--miss-only] [--start 100]
预测文件由 annot_audit/dump_real_preds.py 生成（缺省读发布版 5 种子集成 + 翻转的结果）。
"""
import argparse
import ctypes
import os
import pickle
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
from build_mav6d_cache import D_CALIB, K_CALIB   # noqa: E402

VIEWER_DIR = os.path.join(TOOLS, '..', 'output', 'camnorm', 'viewer')
PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]], dtype=np.float64)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
FRONT = [1, 2, 6, 5]                       # 机体 x = +0.5 的四个角 = 机头那一面
COL_GT = (60, 220, 60)                     # BGR
COL_PRED = (0, 150, 255)
COL_FRONT = (0, 230, 255)
COL_NOSE = (40, 40, 255)
COL_UP = (255, 160, 40)
FONT_FILE = 'C:/Windows/Fonts/msyh.ttc'


# ----------------------------------------------------------------------------- 几何
def rotm(box, seq):
    return R.from_euler(seq, box[6:9]).as_matrix()


def corners(box, seq):
    return (PROTO8 * box[3:6]) @ rotm(box, seq).T + box[:3]


def heading_deg(Rm):
    """俯视（相机 x-z 平面）里机头的方向角：0 = 远离相机，+90 = 画面右侧。"""
    n = Rm[:, 0]
    return float(np.degrees(np.arctan2(n[0], n[2])))


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def errors(p, g, seq):
    Rp, Rg = rotm(p, seq), rotm(g, seq)
    ang = float(np.degrees(R.from_matrix(Rp.T @ Rg).magnitude()))
    dyaw = wrap180(heading_deg(Rp) - heading_deg(Rg))
    tilt = float(np.degrees(np.arccos(np.clip(Rp[:, 2] @ Rg[:, 2], -1, 1))))
    return {'pos': float(np.linalg.norm(p[:3] - g[:3])), 'dz': float(p[2] - g[2]), 'ang': ang,
            'yaw': dyaw, 'yaw_fold': min(abs(dyaw), 180.0 - abs(dyaw)),
            'yaw_fold90': min(abs(dyaw) % 90.0, 90.0 - abs(dyaw) % 90.0), 'tilt': tilt}


def view_K(K, x0, y0, s):
    """把全分辨率内参换到「裁窗 (x0, y0) 起、缩放 s」的输出图上（cv2.resize 像素中心约定）。"""
    Kv = K.copy()
    Kv[0, 0] *= s
    Kv[1, 1] *= s
    Kv[0, 1] *= s
    Kv[0, 2] = (K[0, 2] - x0 + 0.5) * s - 0.5
    Kv[1, 2] = (K[1, 2] - y0 + 0.5) * s - 0.5
    return Kv


def project(Kv, P):
    P = np.atleast_2d(P)
    ok = P[:, 2] > 1e-3
    q = P @ Kv.T
    uv = q[:, :2] / np.where(ok, q[:, 2], 1.0)[:, None]
    return uv, ok


# ----------------------------------------------------------------------------- 画图
def draw_box(img, Kv, box, seq, col, thick):
    C = corners(box, seq)
    uv, ok = project(Kv, C)
    if not ok.all():
        return
    H, W = img.shape[:2]
    pts = np.round(uv * 16).astype(np.int32)            # 亚像素（shift=4）
    P = lambda k: (int(pts[k, 0]), int(pts[k, 1]))
    over = img.copy()
    cv2.fillPoly(over, [pts[FRONT]], COL_FRONT, cv2.LINE_AA, shift=4)
    cv2.addWeighted(over, 0.35, img, 0.65, 0, img)
    for i, j in EDGES:
        c = COL_FRONT if (i in FRONT and j in FRONT) else col
        cv2.line(img, P(i), P(j), c, thick, cv2.LINE_AA, shift=4)
    cv2.line(img, P(1), P(6), COL_FRONT, max(1, thick - 1), cv2.LINE_AA, shift=4)
    cv2.line(img, P(2), P(5), COL_FRONT, max(1, thick - 1), cv2.LINE_AA, shift=4)
    Rm = rotm(box, seq)
    L = float(max(box[3], box[4]))
    c0 = box[:3]
    tip = c0 + Rm[:, 0] * (0.5 * box[3] + 0.45 * L)
    up = c0 + Rm[:, 2] * (0.5 * box[5] + 0.25 * L)
    uv2, ok2 = project(Kv, np.stack([c0, tip, up]))
    if ok2.all():
        a, b, u = [(int(round(x[0])), int(round(x[1]))) for x in uv2]
        cv2.line(img, a, u, COL_UP, thick, cv2.LINE_AA)
        cv2.arrowedLine(img, a, b, (255, 255, 255), thick + 3, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(img, a, b, COL_NOSE, thick + 1, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(img, a, thick + 2, col, -1, cv2.LINE_AA)


def draw_compass(img, Rm, col, r, side='left'):
    """底部角落的俯视罗盘（放在目标的另一侧，不压机体）：下方三角 = 相机，箭头 = 机头的水平朝向。"""
    H, W = img.shape[:2]
    c = (r + 14, H - r - 14) if side == 'left' else (W - r - 14, H - r - 14)
    over = img.copy()
    cv2.circle(over, c, r + 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(over, 0.55, img, 0.45, 0, img)
    cv2.circle(img, c, r, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(img, (c[0], c[1] - r), (c[0], c[1] + r), (110, 110, 110), 1, cv2.LINE_AA)
    cv2.line(img, (c[0] - r, c[1]), (c[0] + r, c[1]), (110, 110, 110), 1, cv2.LINE_AA)
    tri = np.array([[c[0], c[1] + r - 12], [c[0] - 9, c[1] + r + 4], [c[0] + 9, c[1] + r + 4]], np.int32)
    cv2.fillPoly(img, [tri], (230, 230, 230), cv2.LINE_AA)
    n = Rm[:, 0]
    v = np.array([n[0], -n[2]])
    nh = np.linalg.norm(v)
    if nh > 1e-6:
        v = v / nh * (r - 6)
        cv2.arrowedLine(img, c, (int(c[0] + v[0]), int(c[1] + v[1])), col, 3, cv2.LINE_AA, tipLength=0.3)
    return c


def undistort_maps(K_raw, wh):
    return cv2.initUndistortRectifyMap(K_CALIB, D_CALIB, None, K_raw, (int(wh[0]), int(wh[1])), cv2.CV_32FC1)


class Source:
    """原图（去畸变，全分辨率）优先，缺图时退回 512x288 缓存。"""

    def __init__(self, mav6d_root, cache_rgb):
        self.root = mav6d_root
        self.maps = {}
        self.cache = np.load(cache_rgb, mmap_mode='r') if cache_rgb and os.path.exists(cache_rgb) else None
        self.warned = False

    def get(self, f):
        cls, scene, sq = f['seq'].split('/')
        p = os.path.join(self.root, cls, 'JPEGImages', scene, sq, f['frame'])
        img = cv2.imread(p, cv2.IMREAD_COLOR) if os.path.exists(p) else None
        if img is not None:
            key = tuple(np.round(f['K_raw'].reshape(-1), 6))
            if key not in self.maps:
                self.maps[key] = undistort_maps(f['K_raw'], f['raw_wh'])
            m1, m2 = self.maps[key]
            return cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), f['K_raw'], '原图去畸变 1920x1080'
        if self.cache is None:
            raise FileNotFoundError(p)
        if not self.warned:
            print('找不到原图（%s），改用 512x288 缓存图' % p)
            self.warned = True
        return np.ascontiguousarray(self.cache[f['cache_idx']]), f['K_in'], '缓存图 512x288'


def zoom_window(img_wh, Kfull, g, seq, aspect, scale=2.0, min_w=200):
    """以真值框为中心、宽 = 真值 2D 框长边 x scale 的裁窗（两边共用）。"""
    uv, ok = project(Kfull, corners(g, seq))
    W, H = img_wh
    if not ok.all():
        return 0, 0, W, H
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    w = max(min_w, max(x1 - x0, (y1 - y0) * aspect) * scale)
    w = min(w, W)
    h = w / aspect
    if h > H:
        h, w = H, H * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0 = float(np.clip(cx - w / 2, 0, W - w))
    Y0 = float(np.clip(cy - h / 2, 0, H - h))
    return X0, Y0, w, h


def render(img, Kfull, win, out_wh, box, seq, col, thick):
    x0, y0, w, h = win
    s = out_wh[0] / w
    if x0 == 0 and y0 == 0 and w == img.shape[1] and h == img.shape[0]:
        out = cv2.resize(img, (int(out_wh[0]), int(out_wh[1])), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    else:
        M = np.array([[s, 0, (0.5 - x0) * s - 0.5], [0, s, (0.5 - y0) * s - 0.5]], np.float64)
        out = cv2.warpAffine(img, M, (int(out_wh[0]), int(out_wh[1])),
                             flags=cv2.INTER_LINEAR if s < 1 else cv2.INTER_CUBIC, borderValue=(30, 30, 30))
    if box is not None:
        draw_box(out, view_K(Kfull, x0, y0, s), box, seq, col, thick)
    return out


# ----------------------------------------------------------------------------- 主程序
def screen_size():
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def main():
    ap = argparse.ArgumentParser(description='左真值 / 右预测，t y 翻页')
    ap.add_argument('--pkl', default=os.path.join(VIEWER_DIR, 'real_preds_ens5flip.pkl'))
    ap.add_argument('--mav6d', default='@@MAV6D@@', help='MAV6D 原图根目录（找不到就用缓存图）')
    ap.add_argument('--cls', default=None, choices=[None, 'mavic2', 'phantom4'])
    ap.add_argument('--seq', default=None, help='只看序列名含该子串的帧，如 01/0110')
    ap.add_argument('--sort', default='frame', choices=['frame', 'pos', 'yaw', 'yaw90', 'ang'],
                    help='frame = 按序列/帧号；pos/yaw/ang = 按该误差从大到小（未检出排最前）')
    ap.add_argument('--miss-only', action='store_true', help='只看未检出的帧')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--panel-width', type=int, default=0, help='单张图宽（像素），缺省按屏幕自适应')
    ap.add_argument('--export', type=int, default=0, help='不开窗口，均匀导出 N 帧 JPG（检查用）')
    ap.add_argument('--export-zoom', action='store_true', help='导出时每帧再多存一张放大版')
    ap.add_argument('--export-dir', default=os.path.join(VIEWER_DIR, 'export'))
    a = ap.parse_args()

    D = pickle.load(open(a.pkl, 'rb'))
    info, frames = D['info'], D['frames']
    seq = info.get('euler_seq', 'xyz')
    cache_rgb = os.path.join(info.get('data_path', '@@MMCACHE@@/mav6d_cn'), info.get('split', 'test'), 'rgb.npy')
    src = Source(a.mav6d, cache_rgb)
    for f in frames:
        j = f['matched']
        f['err'] = errors(f['pred'][j], f['gt'], seq) if j is not None else None
    sel = [f for f in frames if (a.cls is None or f['cls'] == a.cls) and (a.seq is None or a.seq in f['seq'])
           and (not a.miss_only or f['matched'] is None)]
    if a.sort != 'frame':
        k = {'pos': 'pos', 'yaw': 'yaw_fold', 'yaw90': 'yaw_fold90', 'ang': 'ang'}[a.sort]
        sel.sort(key=lambda f: -(f['err'][k] if f['err'] else 1e9))
    if not sel:
        raise SystemExit('筛完没有帧')
    seq_pos = {}
    for f in frames:
        seq_pos.setdefault(f['seq'], []).append(f['frame'])
    starts = [i for i in range(len(sel)) if i == 0 or sel[i]['seq'] != sel[i - 1]['seq']]

    hit = np.array([f['matched'] is not None for f in sel])
    pe = np.array([f['err']['pos'] for f in sel if f['err']])
    ye = np.array([abs(f['err']['yaw']) for f in sel if f['err']])
    yf = np.array([f['err']['yaw_fold'] for f in sel if f['err']])
    y9 = np.array([f['err']['yaw_fold90'] for f in sel if f['err']])
    print(info['pipeline'])
    print('自标定尺寸 lwh = %s m' % np.round(info['calib_lwh'], 3).tolist())
    print('当前筛选 %d 帧：检出 %.1f%%，位置误差中位 %.3f m，水平朝向误差中位 %.1f°（折 180 后 %.1f°，折 90 后 %.1f°）'
          % (len(sel), 100 * hit.mean(), np.median(pe) if len(pe) else np.nan,
             np.median(ye) if len(ye) else np.nan, np.median(yf) if len(yf) else np.nan, np.median(y9) if len(y9) else np.nan))
    print(__doc__.split('按键')[1].split('python')[0])

    sw, sh = screen_size()
    gap, head, foot = 8, 70, 30
    pw = a.panel_width or int(min(1400, (sw - 60 - gap) // 2))
    ph = int(round(pw * 9 / 16))
    if ph + head + foot > sh - 110:
        ph = sh - 110 - head - foot
        pw = int(ph * 16 / 9)
    thick = max(2, int(round(pw / 480)))
    f_big = ImageFont.truetype(FONT_FILE, max(16, pw // 50))
    f_sm = ImageFont.truetype(FONT_FILE, max(13, pw // 68))
    def compose(i, zoom):
        f = sel[i]
        img, Kfull, src_name = src.get(f)
        Hh, Ww = img.shape[:2]
        g = f['gt']
        j = f['matched']
        top = int(np.argmax(f['conf'])) if len(f['conf']) else None
        pbox = f['pred'][j] if j is not None else (f['pred'][top] if top is not None else None)
        full = (0.0, 0.0, float(Ww), float(Hh))
        zw = zoom_window((Ww, Hh), Kfull, g, seq, 16 / 9)
        main_win = zw if zoom else full
        L = render(img, Kfull, main_win, (pw, ph), g, seq, COL_GT, thick)
        Rr = render(img, Kfull, main_win, (pw, ph), pbox, seq, COL_PRED, thick)
        # 放大小窗（上）和罗盘（下）都放在目标的另一侧，免得压住机体
        ug, _ = project(view_K(Kfull, main_win[0], main_win[1], pw / main_win[2]), g[:3])
        side = 'left' if (zoom or ug[0, 0] > pw / 2) else 'right'
        if not zoom:
            iw = int(pw * 0.36)
            ih = int(iw * 9 / 16)
            ix = 6 if side == 'left' else pw - iw - 6
            for P, box, col in ((L, g, COL_GT), (Rr, pbox, COL_PRED)):
                ins = render(img, Kfull, zw, (iw, ih), box, seq, col, max(2, thick - 1))
                P[6:6 + ih, ix:ix + iw] = ins
                cv2.rectangle(P, (ix - 1, 5), (ix + iw, 6 + ih), (255, 255, 255), 1)
        r_c = max(40, pw // 16)
        draw_compass(L, rotm(g, seq), COL_GT, r_c, side)
        if pbox is not None:
            draw_compass(Rr, rotm(pbox, seq), COL_PRED, r_c, side)

        canvas = np.full((head + ph + foot, 2 * pw + gap, 3), 24, np.uint8)
        canvas[head:head + ph, :pw] = L
        canvas[head:head + ph, pw + gap:] = Rr
        im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(im)
        fr_list = seq_pos[f['seq']]
        k_in_seq = fr_list.index(f['frame']) + 1
        dist = float(np.linalg.norm(g[:3]))
        dr.text((10, 6), '真值   %s   %s   第 %d/%d 帧（本序列 %d/%d）' % (
            f['cls'], f['seq'], i + 1, len(sel), k_in_seq, len(fr_list)), font=f_big, fill=(120, 230, 120))
        dr.text((10, 38), '距离 %.2f m　　深度 %.2f m　　水平朝向 %+.0f°（0 = 远离相机，+90 = 朝画面右）　%s' % (
            dist, g[2], heading_deg(rotm(g, seq)), src_name), font=f_sm, fill=(210, 210, 210))
        x2 = pw + gap + 10
        if f['err'] is not None:
            e = f['err']
            dr.text((x2, 6), '预测（纯仿真训练）   检出　位置误差 %.2f m（深度 %+.2f m）' % (e['pos'], e['dz']),
                    font=f_big, fill=(255, 190, 90))
            dr.text((x2, 38), '全角度误差 %.0f°　　水平朝向误差 %+.0f°（折 180 后 %.0f°，折 90 后 %.0f°）　　倾斜误差 %.0f°　　置信度 %.2f' % (
                e['ang'], e['yaw'], e['yaw_fold'], e['yaw_fold90'], e['tilt'], f['conf'][j]), font=f_sm, fill=(210, 210, 210))
        elif pbox is not None:
            d = float(np.linalg.norm(pbox[:3] - g[:3]))
            dr.text((x2, 6), '预测（纯仿真训练）   未检出：1 m 内没有预测', font=f_big, fill=(255, 90, 90))
            dr.text((x2, 38), '画的是最高分预测，偏 %.2f m，置信度 %.2f' % (d, f['conf'][top]), font=f_sm, fill=(210, 210, 210))
        else:
            dr.text((x2, 6), '预测（纯仿真训练）   无输出', font=f_big, fill=(255, 90, 90))
        dr.text((10, head + ph + 5),
                't / y 上一帧 / 下一帧　T / Y ±24 帧　g / h 上 / 下一段序列　z 整幅 / 放大　s 存图　q 退出　　'
                '黄面 = 机头那一面　红箭头 = 机头方向　蓝线 = 机体朝上　罗盘 = 俯视（下方三角是相机）',
                font=f_sm, fill=(180, 180, 180))
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    shots = os.path.join(VIEWER_DIR, 'shots')
    if a.export:
        os.makedirs(a.export_dir, exist_ok=True)
        for i in np.linspace(a.start, len(sel) - 1, min(a.export, len(sel) - a.start)).astype(int):
            for zoom in (False, True) if a.export_zoom else (False,):
                p = os.path.join(a.export_dir, '%04d%s.jpg' % (i, '_zoom' if zoom else ''))
                cv2.imencode('.jpg', compose(int(i), zoom), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(p)
        print('导出到', os.path.abspath(a.export_dir))
        return

    win_name = 'MAV6D real test  |  left: GT   right: pure-sim prediction  |  t/y page, q quit'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win_name, 2 * pw + gap, ph + head + foot)

    i = int(np.clip(a.start, 0, len(sel) - 1))
    zoom = False
    while True:
        f = sel[i]
        frame_bgr = compose(i, zoom)
        cv2.imshow(win_name, frame_bgr)

        key = -1
        while key == -1:
            key = cv2.waitKeyEx(50)
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                key = 27
        if key in (27, ord('q'), ord('Q')):
            break
        if key in (ord('y'), 2555904):
            i = min(i + 1, len(sel) - 1)
        elif key in (ord('t'), 2424832):
            i = max(i - 1, 0)
        elif key == ord('Y'):
            i = min(i + 24, len(sel) - 1)
        elif key == ord('T'):
            i = max(i - 24, 0)
        elif key == ord('h'):
            nxt = [s for s in starts if s > i]
            i = nxt[0] if nxt else i
        elif key == ord('g'):
            prv = [s for s in starts if s < i]
            cur = max([s for s in starts if s <= i])
            i = prv[-1] if (i == cur and prv) else cur
        elif key in (ord('z'), ord('Z')):
            zoom = not zoom
        elif key in (ord('s'), ord('S')):
            os.makedirs(shots, exist_ok=True)
            p = os.path.join(shots, '%s_%s.png' % (f['seq'].replace('/', '_'), os.path.splitext(f['frame'])[0]))
            cv2.imencode('.png', frame_bgr)[1].tofile(p)
            print('存图', os.path.abspath(p))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
