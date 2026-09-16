# -*- coding: utf-8 -*-
"""annot_audit/measure —— 静止相机背景建模，实测两域「标签中心 vs 视觉中心」「标签框 vs 可见外形」。
不依赖任何模型，纯 CPU。只读数据集；输出到 output/camnorm/annot_audit/measure/。

  python measure_bg_offset.py --domain mav6d --max-frames 80
  python measure_bg_offset.py --domain sim   --max-frames 60

方法：同序列取时间上均匀分布的 N 帧 -> 逐像素灰度中值当背景（无人机在动，相机静止）
      前景 = |帧-背景| > thr，3x3 开 + 5x5 闭，连通块；取「质心离标签中心投影点最近」的块（面积>=min_area）
      并另存「与标签框投影(外扩 50%)相交的所有块的并集」作对照。
"""
import os, sys, json, glob, pickle, argparse, time, csv
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/measure'
MAV_ROOT = 'E:/MAV6D'
MAV_INDEX = 'E:/mmcache/mav6d_cn/test/index.pkl'
MAV_K = np.array([[1979.4, 0.3984, 976.8189], [0, 1979.1, 533.9717], [0, 0, 1]], dtype=np.float64)
MAV_D = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487], dtype=np.float64)
MAV_LWH = (0.34, 0.34, 0.23)
MAV_GEOM_OFF = np.array([-0.01, 0.01, -0.055])          # 官方 phantom4 角点范围的几何中心（机体系）
MAV_OFF_RANGE = ((-0.18, 0.16), (-0.16, 0.18), (-0.17, 0.06))
# 与 uavdet3d/datasets/mav6d/mav6d_utils.py read_truth_Rt / E:/MAV6D/util.py 完全相同的矩阵（VICON -> 相机）
CAM_FROM_VICON = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                           [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                           [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                           [0, 0, 0, 1]])
SIM_ROOT = 'D:/data_collect'
SIM_K = np.array([[640., 0, 639.5], [0, 640., 359.5], [0, 0, 1]])
SIM_PICK = {
    'Demonstration_Simple_Capture': ['DJI-phantom4', 'DJI-mavic-mini', 'Matrice-600-Pro'],
    'PowerPlant_Capture': ['DJI-phantom4', 'DJI-mavic-mini', 'DJI-phantom4_up'],
    'Warehouse_Capture': ['DJI-phantom4', 'DJI-mavic-mini', 'Matrice-600-Pro'],
    'Demonstration_Capture': ['DJI-phantom4', 'DJI-mavic-mini', 'DJI-phantom4_up'],
}
THRS = (25, 30, 40)
MAIN_THR = 30


def read_truth_Rt(labpath):
    """与 mav6d_utils.read_truth_Rt 等价：返回 R(相机<-机体), t(机体原点=VICON 原点在相机系的位置)。"""
    pose = [float(x) for x in open(labpath).read().strip().split(' ')]
    q = pose[9:]
    Rm = R.from_quat([q[-4], q[-3], q[-2], q[-1]]).as_matrix()
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = q[:3]
    Tc = CAM_FROM_VICON @ T
    return Tc[:3, :3], Tc[:3, 3]


def box_corners(lwh, off=(0, 0, 0)):
    l, w, h = lwh
    c = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return c + np.asarray(off, dtype=np.float64)


def range_corners(rng):
    (x0, x1), (y0, y1), (z0, z1) = rng
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float64)


def proj_mav(pts_cam):
    uv, _ = cv2.projectPoints(np.asarray(pts_cam, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), MAV_K, MAV_D)
    return uv.reshape(-1, 2)


def proj_sim(pts_cam):
    p = np.asarray(pts_cam, np.float64).reshape(-1, 3)
    return np.stack([SIM_K[0, 0] * p[:, 0] / p[:, 2] + SIM_K[0, 2], SIM_K[1, 1] * p[:, 1] / p[:, 2] + SIM_K[1, 2]], 1)


def bbox_of(uv):
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def blobs(diff, thr, min_area):
    m = (diff > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, connectivity=8)
    st, ce = st[1:], ce[1:]
    keep = st[:, cv2.CC_STAT_AREA] >= min_area
    return st[keep], ce[keep]


def pick_nearest(st, ce, uv):
    if len(st) == 0:
        return None
    d = np.hypot(ce[:, 0] - uv[0], ce[:, 1] - uv[1])
    i = int(np.argmin(d))
    x, y, w, h, a = st[i]
    return dict(cx=float(ce[i, 0]), cy=float(ce[i, 1]), x0=int(x), y0=int(y), w=int(w), h=int(h), area=int(a), dist=float(d[i]))


def pick_union(st, ce, lab_box, expand=0.5):
    if len(st) == 0:
        return None
    x0, y0, x1, y1 = lab_box
    ex, ey = (x1 - x0) * expand, (y1 - y0) * expand
    bx0, by0, bx1, by1 = x0 - ex, y0 - ey, x1 + ex, y1 + ey
    sx0, sy0 = st[:, 0], st[:, 1]
    sx1, sy1 = sx0 + st[:, 2], sy0 + st[:, 3]
    hit = (sx1 > bx0) & (sx0 < bx1) & (sy1 > by0) & (sy0 < by1)
    if not hit.any():
        return None
    s, c = st[hit], ce[hit]
    a = s[:, 4].astype(np.float64)
    ux0, uy0, ux1, uy1 = s[:, 0].min(), s[:, 1].min(), (s[:, 0] + s[:, 2]).max(), (s[:, 1] + s[:, 3]).max()
    return dict(cx=float((c[:, 0] * a).sum() / a.sum()), cy=float((c[:, 1] * a).sum() / a.sum()),
                x0=int(ux0), y0=int(uy0), w=int(ux1 - ux0), h=int(uy1 - uy0), area=int(a.sum()), nblob=int(hit.sum()))


def load_gray_stack(paths):
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError('cannot read ' + p)
        imgs.append(im)
    return np.stack(imgs)


def median_bg(stack):
    # uint8 上做 partition，只在中间两片上取均值，避免 float64 全量拷贝
    return np.median(stack, axis=0).astype(np.float32)


def draw_overlay(img_bgr, rec, lab_uv8, uv_lab, uv_geo, blob, union, out_path, crop=True):
    im = img_bgr.copy()
    b = bbox_of(lab_uv8).astype(int)
    cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    cv2.drawMarker(im, (int(round(uv_lab[0])), int(round(uv_lab[1]))), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
    if uv_geo is not None:
        cv2.drawMarker(im, (int(round(uv_geo[0])), int(round(uv_geo[1]))), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 1)
    if union is not None:
        cv2.rectangle(im, (union['x0'], union['y0']), (union['x0'] + union['w'], union['y0'] + union['h']), (255, 0, 255), 1)
    if blob is not None:
        cv2.rectangle(im, (blob['x0'], blob['y0']), (blob['x0'] + blob['w'], blob['y0'] + blob['h']), (0, 0, 255), 1)
        cv2.drawMarker(im, (int(round(blob['cx'])), int(round(blob['cy']))), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
    if crop:
        s = max(b[2] - b[0], b[3] - b[1]) * 1.6 + 40
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0, y0 = int(max(0, cx - s)), int(max(0, cy - s))
        x1, y1 = int(min(im.shape[1], cx + s)), int(min(im.shape[0], cy + s))
        im = im[y0:y1, x0:x1]
        if im.size and max(im.shape[:2]) < 400:
            f = 400.0 / max(im.shape[:2])
            im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    txt = 'green=label box/center  cyan=geom-center  red=nearest blob  magenta=union'
    cv2.putText(im, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, im, [cv2.IMWRITE_JPEG_QUALITY, 85])


def analyze_sequence(dom, seq, cls, frames, img_paths, per_frame_geom, proj_fn, K, min_area, max_frames, log):
    """frames: 帧名列表；per_frame_geom(i) -> dict(t, Rm, corners_cam(8,3), t_geo or None, off_corners or None)"""
    n = len(frames)
    sel = np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))
    t0 = time.time()
    stack = load_gray_stack([img_paths[i] for i in sel])
    bg = median_bg(stack)
    log('  %s: %d/%d frames, stack %s, load+median %.1fs' % (seq, len(sel), n, stack.shape, time.time() - t0))
    rows, ov = [], []
    fx, fy = K[0, 0], K[1, 1]
    for j, i in enumerate(sel):
        g = per_frame_geom(i)
        uv8 = proj_fn(g['corners_cam'])
        lab_box = bbox_of(uv8)
        w_lab, h_lab = lab_box[2] - lab_box[0], lab_box[3] - lab_box[1]
        uv_lab = proj_fn(g['t'].reshape(1, 3))[0]
        uv_geo = proj_fn(g['t_geo'].reshape(1, 3))[0] if g.get('t_geo') is not None else None
        off_box = bbox_of(proj_fn(g['off_corners'])) if g.get('off_corners') is not None else None
        body_px = max(w_lab, h_lab)
        Z = float(g['t'][2])
        diff = np.abs(stack[j].astype(np.float32) - bg)
        # 机体系偏移最小二乘用的雅可比（数值，含畸变）：d(uv)/d(o_body)，o 以米计
        Jc = []
        for k in range(3):
            e = np.zeros(3); e[k] = 0.01
            Jc.append((proj_fn((g['t'] + g['Rm'] @ e).reshape(1, 3))[0] - uv_lab) / 0.01)
        J = np.stack(Jc, 1).tolist()   # (2,3)
        for thr in THRS:
            st, ce = blobs(diff, thr, min_area)
            nb = pick_nearest(st, ce, uv_lab)
            un = pick_union(st, ce, lab_box)
            rec = dict(domain=dom, seq=seq, cls=cls, frame=frames[i], thr=thr, Z=Z,
                       u_lab=float(uv_lab[0]), v_lab=float(uv_lab[1]), w_lab=float(w_lab), h_lab=float(h_lab),
                       body_px=float(body_px), n_blobs=int(len(st)), J=J)
            if uv_geo is not None:
                rec.update(u_geo=float(uv_geo[0]), v_geo=float(uv_geo[1]))
            if off_box is not None:
                rec.update(w_off=float(off_box[2] - off_box[0]), h_off=float(off_box[3] - off_box[1]))
            if nb is None:
                rec.update(found=0, excluded=1)
            else:
                dx, dy = nb['cx'] - uv_lab[0], nb['cy'] - uv_lab[1]
                rec.update(found=1, excluded=int(nb['dist'] > body_px), dist_px=nb['dist'],
                           blob_cx=nb['cx'], blob_cy=nb['cy'], blob_w=nb['w'], blob_h=nb['h'], blob_area=nb['area'],
                           dx_px=dx, dy_px=dy, dx_m=dx * Z / fx, dy_m=dy * Z / fy,
                           vis_w_over_lab_w=nb['w'] / w_lab, vis_h_over_lab_h=nb['h'] / h_lab)
                if uv_geo is not None:
                    rec.update(dx_geo_px=nb['cx'] - uv_geo[0], dy_geo_px=nb['cy'] - uv_geo[1])
                if off_box is not None:
                    rec.update(vis_w_over_off_w=nb['w'] / (off_box[2] - off_box[0]),
                               vis_h_over_off_h=nb['h'] / (off_box[3] - off_box[1]))
            if un is not None:
                rec.update(un_cx=un['cx'], un_cy=un['cy'], un_w=un['w'], un_h=un['h'], un_area=un['area'], un_nblob=un['nblob'],
                           un_dx_px=un['cx'] - uv_lab[0], un_dy_px=un['cy'] - uv_lab[1],
                           un_w_over_lab_w=un['w'] / w_lab, un_h_over_lab_h=un['h'] / h_lab)
            rows.append(rec)
            if thr == MAIN_THR and j in (len(sel) // 4, (3 * len(sel)) // 4):
                ov.append((i, rec, uv8, uv_lab, uv_geo, nb, un))
    return rows, ov, sel


def run_mav(args, log):
    d = pickle.load(open(MAV_INDEX, 'rb'))
    by_seq = {}
    for m in d['metas']:
        by_seq.setdefault(m['seq'], []).append(m['frame'])
    all_rows = []
    lwh_c = box_corners(MAV_LWH)
    off_c = range_corners(MAV_OFF_RANGE)
    for seq in sorted(by_seq):
        cls, scene, sq = seq.split('/')
        frames = sorted(set(by_seq[seq]), key=lambda f: int(os.path.splitext(f)[0]))
        img_paths = [os.path.join(MAV_ROOT, cls, 'JPEGImages', scene, sq, f) for f in frames]
        lab_paths = [os.path.join(MAV_ROOT, cls, 'labels', scene, sq, os.path.splitext(f)[0] + '.txt') for f in frames]
        cache = {}

        def geom(i):
            if i not in cache:
                Rm, t = read_truth_Rt(lab_paths[i])
                cache[i] = dict(Rm=Rm, t=t, corners_cam=(Rm @ lwh_c.T).T + t, t_geo=Rm @ MAV_GEOM_OFF + t,
                                off_corners=(Rm @ off_c.T).T + t)
            return cache[i]
        rows, ov, sel = analyze_sequence('mav6d', seq, cls, frames, img_paths, geom, proj_mav, MAV_K,
                                         args.min_area_mav, args.max_frames, log)
        all_rows += rows
        for (i, rec, uv8, uv_lab, uv_geo, nb, un) in ov:
            img = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
            name = 'mav6d_%s_%s_%s_%s' % (cls, scene, sq, os.path.splitext(frames[i])[0])
            draw_overlay(img, rec, uv8, uv_lab, uv_geo, nb, un, os.path.join(OUT, 'overlay_' + name + '_crop.jpg'), crop=True)
            if i == sel[len(sel) // 4]:
                draw_overlay(img, rec, uv8, uv_lab, uv_geo, nb, un, os.path.join(OUT, 'overlay_' + name + '_full.jpg'), crop=False)
    return all_rows


def run_sim(args, log):
    all_rows = []
    for scene, combos in SIM_PICK.items():
        for combo in combos:
            base = os.path.join(SIM_ROOT, scene, 'carla_data', '00001', 'clear_day', combo)
            frames = sorted([os.path.basename(p) for p in glob.glob(os.path.join(base, 'images_rgb', '*.png'))],
                            key=lambda f: float(os.path.splitext(f)[0]))
            img_paths = [os.path.join(base, 'images_rgb', f) for f in frames]
            box_paths = [os.path.join(base, 'boxes_rgb', os.path.splitext(f)[0] + '.pkl') for f in frames]
            seq = '%s/%s' % (scene, combo)
            cache = {}

            def geom(i):
                if i not in cache:
                    b = pickle.load(open(box_paths[i], 'rb'))
                    if len(b) != 1:
                        raise RuntimeError('%s frame %s has %d boxes' % (seq, frames[i], len(b)))
                    c = np.stack([np.asarray(x, np.float64) for x in b[0][1:9]])
                    ex, ey, ez = c[1] - c[0], c[3] - c[0], c[4] - c[0]
                    Rm = np.stack([ex / np.linalg.norm(ex), ey / np.linalg.norm(ey), ez / np.linalg.norm(ez)], 1)  # 框轴系（未翻手性）
                    cache[i] = dict(Rm=Rm, t=c.mean(0), corners_cam=c, t_geo=None, off_corners=None,
                                    lwh=(np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez)), name=b[0][0])
                return cache[i]
            rows, ov, sel = analyze_sequence('sim', seq, combo, frames, img_paths, geom, proj_sim, SIM_K,
                                             args.min_area_sim, args.max_frames, log)
            fidx = {f: k for k, f in enumerate(frames)}
            for r in rows:
                g = cache[fidx[r['frame']]]
                r.update(l3d=float(g['lwh'][0]), w3d=float(g['lwh'][1]), h3d=float(g['lwh'][2]), actor=g['name'])
            all_rows += rows
            for (i, rec, uv8, uv_lab, uv_geo, nb, un) in ov:
                img = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
                name = 'sim_%s_%s_%s' % (scene.replace('_Capture', ''), combo, os.path.splitext(frames[i])[0])
                draw_overlay(img, rec, uv8, uv_lab, uv_geo, nb, un, os.path.join(OUT, 'overlay_' + name + '_crop.jpg'), crop=True)
                if i == sel[len(sel) // 4]:
                    draw_overlay(img, rec, uv8, uv_lab, uv_geo, nb, un, os.path.join(OUT, 'overlay_' + name + '_full.jpg'), crop=False)
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=['mav6d', 'sim'])
    ap.add_argument('--max-frames', type=int, default=80)
    ap.add_argument('--min-area-mav', type=int, default=40)
    ap.add_argument('--min-area-sim', type=int, default=20)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log_%s.txt' % args.domain), 'w', encoding='utf-8')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    t0 = time.time()
    rows = run_mav(args, log) if args.domain == 'mav6d' else run_sim(args, log)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k != 'J':
                keys.append(k)
    with open(os.path.join(OUT, 'frames_%s.csv' % args.domain), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, 'frames_%s.pkl' % args.domain), 'wb') as f:
        pickle.dump(rows, f)
    log('done %s: %d rows in %.1fs' % (args.domain, len(rows), time.time() - t0))


if __name__ == '__main__':
    main()
