# -*- coding: utf-8 -*-
"""MAV6D 标注约定核实（只读数据集/缓存，输出到 output/camnorm/annot_audit/mav_code）：
  A. 逐帧把 t（read_truth_Rt）投影到 原始带畸变 JPEG（K_CALIB+D）和 缓存 512x288 去畸变图（K_in）上，
     画中心/机体轴/官方角点框，人工看是否落在无人机上
  B. 用 util.py 的 get_projection 逻辑重算 9 个关键点，与 labels_yolo6d 对比
     （验证官方 yolo6d 标签 = 同一套约定；mavic2 是否也用 phantom4 框）
  C. 手性：四元数 R 的 det；缓存 boxes9d 欧拉角还原 R 后 z 列 . (camera2vicon_R @ [0,0,1])
  D. 框中心偏移 (-0.01,+0.01,-0.055) 在像素上的大小
"""
import os
import sys
import pickle
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, 'E:/Open3DUAVDet')
from uavdet3d.utils import camera_geometry as cg  # noqa: E402

ROOT = 'E:/MAV6D'
CACHE = 'E:/mmcache/mav6d_cn'
OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/mav_code'
os.makedirs(OUT, exist_ok=True)
K_CALIB = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.1, 533.9717], [0.0, 0.0, 1.0]])
D_CALIB = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
CAM2VIC = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                    [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                    [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                    [0, 0, 0, 1]])
# util.py 52-68 官方 phantom4 角点范围
mnx, mxx, mny, mxy, mnz, mxz = -0.18, 0.16, -0.16, 0.18, -0.17, 0.06
CORNERS9 = np.array([[(mnx + mxx) / 2, (mny + mxy) / 2, (mnz + mxz) / 2],
                     [mnx, mny, mnz], [mnx, mny, mxz], [mnx, mxy, mnz], [mnx, mxy, mxz],
                     [mxx, mny, mnz], [mxx, mny, mxz], [mxx, mxy, mnz], [mxx, mxy, mxz]])
log = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log.append(s)


def read_label(path):
    with open(path) as f:
        return np.array(list(map(float, f.read().strip().split(' '))))


def truth_Rt(v):
    up = v[9:]
    Tvu = np.eye(4)
    Tvu[:3, :3] = R.from_quat([up[3], up[4], up[5], up[6]]).as_matrix()
    Tvu[:3, 3] = up[:3]
    T = CAM2VIC @ Tvu
    return T[:3, :3], T[:3, 3]


def util_projection(v):
    """util.py projection_in_distort_image + get_projection 逐行复现（手写畸变公式）"""
    up = v[9:]
    r = R.from_quat([up[-4], up[-3], up[-2], up[-1]])
    cv_ = r.apply(CORNERS9) + up[:3]
    ph = np.concatenate((cv_.T, np.ones((1, 9))), 0)
    pc = CAM2VIC.dot(ph)
    xp = pc[0] / pc[2]
    yp = pc[1] / pc[2]
    D = D_CALIB
    r2 = xp * xp + yp * yp
    xd = xp * (1 + D[0] * r2 + D[1] * r2 ** 2 + D[4] * r2 ** 3) + 2 * D[2] * xp * yp + D[3] * (r2 + 2 * xp * xp)
    yd = yp * (1 + D[0] * r2 + D[1] * r2 ** 2 + D[4] * r2 ** 3) + D[2] * (r2 + 2 * yp * yp) + 2 * D[3] * xp * yp
    return np.stack([1979.4 * xd + 976.8189, 1979.1 * yd + 533.9717], 1)


def proj(pts, K, D=None):
    uv, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K,
                              np.zeros(5) if D is None else D)
    return uv.reshape(-1, 2)


EDGES = [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6), (5, 7), (6, 8), (7, 8), (1, 5), (2, 6), (3, 7), (4, 8)]


def draw(img, K, D, Rm, t, scale=1.0, thick=2):
    c9 = (CORNERS9 @ Rm.T) + t            # 官方角点框（相对 VICON 原点，含偏移）
    L = 0.25
    pts = np.vstack([c9, t, t + Rm[:, 0] * L, t + Rm[:, 1] * L, t + Rm[:, 2] * L])
    uv = proj(pts, K, D)
    for a, b in EDGES:
        cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), (0, 255, 255), thick, cv2.LINE_AA)
    o = tuple(np.int32(uv[9]))
    cv2.circle(img, o, int(4 * scale) + 2, (255, 0, 255), -1)           # 品红 = VICON 刚体原点 t
    cv2.circle(img, tuple(np.int32(uv[0])), int(4 * scale) + 2, (0, 165, 255), 2)  # 橙 = 官方框几何中心
    for k, col in ((10, (0, 0, 255)), (11, (0, 255, 0)), (12, (255, 0, 0))):
        cv2.arrowedLine(img, o, tuple(np.int32(uv[k])), col, thick, cv2.LINE_AA, tipLength=0.2)
    return uv


# ---------------- 选帧：test 里每机型 最近 2 帧 + 中位深度 1 帧 ----------------
idx = pickle.load(open(os.path.join(CACHE, 'test', 'index.pkl'), 'rb'))
rgb = np.load(os.path.join(CACHE, 'test', 'rgb.npy'), mmap_mode='r')
metas = idx['metas']
picks = []
for cls in ('phantom4', 'mavic2'):
    ii = [i for i in idx['valid_idx'] if metas[i]['cls'] == cls]
    Z = np.array([metas[i]['boxes9d'][0, 2] for i in ii])
    order = np.argsort(Z)
    picks += [ii[order[0]], ii[order[len(order) // 2]]]
tiles_raw, tiles_cache = [], []
for i in picks:
    m = metas[i]
    cls, scene, seq = m['seq'].split('/')
    stem = os.path.splitext(m['frame'])[0]
    v = read_label(os.path.join(ROOT, cls, 'labels', scene, seq, stem + '.txt'))
    Rm, t = truth_Rt(v)
    b = m['boxes9d'][0].astype(np.float64)
    Rc_cache = R.from_euler('xyz', b[6:9]).as_matrix()
    P('[%s/%s/%s %s] t=%s  |t_cache-t|=%.2e  |R_cache-R|=%.2e  det(R)=%.4f'
      % (cls, scene, seq, m['frame'], np.round(t, 4), np.abs(b[:3] - t).max(), np.abs(Rc_cache - Rm).max(), np.linalg.det(Rm)))
    raw = cv2.imread(os.path.join(ROOT, cls, 'JPEGImages', scene, seq, m['frame']))
    uv = draw(raw, K_CALIB, D_CALIB, Rm, t, 2.0, 2)
    uv_c = proj(t, K_CALIB, D_CALIB)[0]
    y6 = read_label(os.path.join(ROOT, cls, 'labels_yolo6d', scene, seq, stem + '.txt'))
    kp_yolo = y6[1:19].reshape(9, 2) * np.array([1920., 1080.])
    kp_util = util_projection(v)
    P('   raw: t -> (%.1f, %.1f)px ; official-box-centre -> (%.1f, %.1f)px ; yolo6d kp[0]=(%.1f,%.1f) ; '
      'max|util-yolo6d| over 9 kps = %.3f px ; yolo6d w,h = %.1f x %.1f px'
      % (uv_c[0], uv_c[1], uv[0, 0], uv[0, 1], kp_yolo[0, 0], kp_yolo[0, 1], np.abs(kp_util - kp_yolo).max(),
         y6[19] * 1920, y6[20] * 1080))
    for p in kp_yolo:
        cv2.circle(raw, tuple(np.int32(p)), 5, (255, 255, 255), 1)
    h = 260
    cu, cv_ = int(uv_c[0]), int(uv_c[1])
    pad = cv2.copyMakeBorder(raw, h, h, h, h, cv2.BORDER_CONSTANT)
    crop = cv2.resize(pad[cv_:cv_ + 2 * h, cu:cu + 2 * h], (520, 520), interpolation=cv2.INTER_CUBIC)
    cv2.putText(crop, '%s %s/%s z=%.2f raw+dist' % (cls, scene, seq, t[2]), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    tiles_raw.append(crop)
    small = np.ascontiguousarray(rgb[i])
    K_in = np.asarray(m['K_in'], np.float64)
    big = cv2.resize(small, (1536, 864), interpolation=cv2.INTER_CUBIC)
    K_big = cg.scale_K(K_in, 3.0, 3.0)
    draw(big, K_big, None, Rm, t, 1.0, 2)
    uv_c2 = proj(t, K_in)[0]
    P('   cache 512x288: t -> (%.1f, %.1f)px  (centre_inside=%s)' % (uv_c2[0], uv_c2[1], m['center_inside']))
    h = 130
    cu, cv_ = int(uv_c2[0] * 3), int(uv_c2[1] * 3)
    pad = cv2.copyMakeBorder(big, h, h, h, h, cv2.BORDER_CONSTANT)
    crop2 = cv2.resize(pad[cv_:cv_ + 2 * h, cu:cu + 2 * h], (520, 520), interpolation=cv2.INTER_CUBIC)
    cv2.putText(crop2, '%s %s/%s cache K_in' % (cls, scene, seq), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    tiles_cache.append(crop2)
cv2.imwrite(os.path.join(OUT, 'proj_raw_distorted.jpg'), np.concatenate(tiles_raw, 1))
cv2.imwrite(os.path.join(OUT, 'proj_cache_Kin.jpg'), np.concatenate(tiles_cache, 1))

# ---------------- B. 全 test 集 yolo6d 对比（两机型） ----------------
for cls in ('phantom4', 'mavic2'):
    errs = []
    n = 0
    for i in idx['valid_idx']:
        m = metas[i]
        if m['cls'] != cls:
            continue
        c, scene, seq = m['seq'].split('/')
        stem = os.path.splitext(m['frame'])[0]
        yp = os.path.join(ROOT, cls, 'labels_yolo6d', scene, seq, stem + '.txt')
        if not os.path.exists(yp):
            continue
        v = read_label(os.path.join(ROOT, cls, 'labels', scene, seq, stem + '.txt'))
        y6 = read_label(yp)
        kp_yolo = y6[1:19].reshape(9, 2) * np.array([1920., 1080.])
        errs.append(np.abs(util_projection(v) - kp_yolo).max())
        n += 1
    errs = np.array(errs)
    P('[B] %s: %d test frames with yolo6d labels; max|util.py phantom4-box projection - labels_yolo6d| '
      'median %.3f px, max %.3f px' % (cls, n, np.median(errs), errs.max()))

# ---------------- C. 手性 / z 朝上 / 欧拉往返，全 test ----------------
up_cam = CAM2VIC[:3, :3] @ np.array([0, 0, 1.0])
dets, zdots, xdots, ydots, zc = [], [], [], [], []
for i in idx['valid_idx']:
    b = metas[i]['boxes9d'][0].astype(np.float64)
    Rm = R.from_euler('xyz', b[6:9]).as_matrix()
    dets.append(np.linalg.det(Rm))
    zdots.append(Rm[:, 2] @ up_cam)
    xdots.append(Rm[:, 0] @ up_cam)
    ydots.append(Rm[:, 1] @ up_cam)
    zc.append(Rm[2, 2])
dets, zdots, xdots, ydots = map(np.array, (dets, zdots, xdots, ydots))
P('[C] test %d frames: det(R from euler xyz) min %.4f max %.4f' % (len(dets), dets.min(), dets.max()))
P('[C] body z . VICON-up(in cam) : median %.4f  p1 %.4f  min %.4f ; body x . up median %.4f ; body y . up median %.4f'
  % (np.median(zdots), np.percentile(zdots, 1), zdots.min(), np.median(xdots), np.median(ydots)))
P('[C] VICON up expressed in camera frame = %s  (camera y down => component ~ -1)' % np.round(up_cam, 4))
P('[C] body z . camera z (R[2,2]) median %.3f' % np.median(zc))
sub = np.random.RandomState(0).choice(idx['valid_idx'], 300, replace=False)
e = []
for i in sub:
    m = metas[i]
    cls, scene, seq = m['seq'].split('/')
    stem = os.path.splitext(m['frame'])[0]
    v = read_label(os.path.join(ROOT, cls, 'labels', scene, seq, stem + '.txt'))
    Rm, t = truth_Rt(v)
    b = m['boxes9d'][0].astype(np.float64)
    e.append(max(np.abs(R.from_euler('xyz', b[6:9]).as_matrix() - Rm).max(), np.abs(b[:3] - t).max()))
P('[C] 300 random test frames: max |cache(euler xyz -> R, t) - read_truth_Rt(R, t)| = %.2e (float32 storage)' % max(e))

# ---------------- D. 偏移量化 ----------------
Z = np.array([metas[i]['boxes9d'][0, 2] for i in idx['valid_idx']])
zmed = np.median(Z)
f_raw = np.sqrt(1979.4 * 1979.1)
K_in = np.asarray(metas[idx['valid_idx'][0]]['K_in'])
f_in = np.sqrt(K_in[0, 0] * K_in[1, 1])
off = np.array([-0.01, 0.01, -0.055])
P('[D] test depth median %.3f m; |offset| = %.4f m; z-offset 0.055 m -> %.1f px on raw 1920 (f=%.1f), %.1f px on cache 512 (f=%.1f); '
  'full offset -> %.1f px raw / %.1f px cache'
  % (zmed, np.linalg.norm(off), 0.055 / zmed * f_raw, f_raw, 0.055 / zmed * f_in, f_in,
     np.linalg.norm(off) / zmed * f_raw, np.linalg.norm(off) / zmed * f_in))
d = []
for i in idx['valid_idx']:
    b = metas[i]['boxes9d'][0].astype(np.float64)
    Rm = R.from_euler('xyz', b[6:9]).as_matrix()
    c2 = b[:3] + Rm @ off
    d.append(proj(c2, K_in)[0] - proj(b[:3], K_in)[0])
d = np.array(d)
P('[D] measured on cache (all test): box-centre minus VICON-origin pixel offset du median %.2f dv median %.2f |d| median %.2f px '
  '(dv>0 = box centre BELOW origin in image)' % (np.median(d[:, 0]), np.median(d[:, 1]), np.median(np.linalg.norm(d, axis=1))))
app = 0.34 / Z * f_in
P('[D] apparent box width 0.34 m on cache: median %.1f px -> offset/width = %.2f'
  % (np.median(app), np.median(np.linalg.norm(d, axis=1)) / np.median(app)))
with open(os.path.join(OUT, 'verify_projection.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log))
