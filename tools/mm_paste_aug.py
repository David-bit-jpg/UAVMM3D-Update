# -*- coding: utf-8 -*-
"""交叉贴（copy-paste）多模态增广：从 A 帧抠出最近的无人机，贴到 B 帧的干净背景里的【新位置 / 新距离】，
RGB / IR / DVS 按同一个几何变换贴，LiDAR 与雷达【从点云做起】——无人机的点做同一个刚体变换后写回 B 帧的点云
（各自传感器坐标系），深度图 / 标签图 / 雷达热图全部从新点云重新投影，不在 2D 上挪像素。

几何（全部在 RGB 相机 OpenCV 系里）：
  * 把无人机从像素 p_A 挪到 p_B：等价于绕相机中心的一次旋转 R_cam（把视线 K^-1 p_A 转到 K^-1 p_B）。
    纯旋转下图像是精确的单应变换 H = K R_cam K^-1，无人机的像素、3D 框、点云在这一步都是【精确】一致的；
  * 再沿视线把无人机拉近/推远（表观放大 s 倍）：3D 上是平移 t = (1/s - 1) R_cam c，点云精确；
    像素上近似为绕投影中心的相似缩放（小物体、>=2 m 时二阶小量）；
  * 新标签：中心 c' = R_cam c + t，旋转 R' = R_cam R，尺寸不变。
  * 点云：X' = R_cam X + t（相机系）-> 换回 B 帧的 LiDAR / 雷达传感器系写入；雷达再换回 (az, alt, depth)。
校验：新框 8 角点在窗口内；无人机 LiDAR 点在新 OBB 内的比例与在 A 帧原框内的比例相同（刚体）；
      贴上去的像素区域与新框投影凸包的重合度；B 背景在贴入处没有更近的 LiDAR 回波（遮挡）。

输入：变焦缓存（tools/build_mm_cache.py --zoom-to-mav6d，给出窗口、K、标签）+ 原始数据（五个模态 + 点云）
     + 翻译好的干净背景（tools/sim2real_bg_translate.py --no-paste）。
输出：每个样本一个 .npz（rgb 翻译背景版 / rgb_sim 教师版 / ir / dvs / depth / tag / radar_hm / 新点云 / 新标签 / 变换）
     和一张 3 行 4 列的可视化大图。

    D:/Miniconda3/envs/city/python.exe tools/mm_paste_aug.py --cache E:/mmcache/demo_zoom --plates E:/mmcache/demo_zoom_plates \
        --n 16 --out ../output/paste_demo
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from build_mm_cache import (CARLA_TO_OPENCV, OPENCV_TO_CARLA, cam_to_world, world_to_pixel, frame_sort_key,  # noqa: E402
                            corners_to_9params, class_of)
# 可视化用用户原有的投影代码（LiDAR 点投影 / 雷达速度热图），保证口径与原数据集工具一致
from uavdet3d.datasets.laam6d.dataset_utils import project_lidar_and_get_uvz_rgb_tag, radar_to_velocity_heatmap  # noqa: E402


def load_full_boxes(base, frame, max_label_range=40.0):
    """整帧的全部无人机框（不只是裁剪窗口内的）：抹除时要用它——窗口外的无人机经 IR/DVS 视差平移后会进窗口。"""
    try:
        raw = pickle.load(open(os.path.join(base, 'boxes_rgb', os.path.splitext(frame)[0] + '.pkl'), 'rb'))
    except Exception:
        return []
    out = []
    for row in raw:
        name = row[0] if isinstance(row[0], str) else '?'
        if class_of(name) is None:
            continue
        c = np.array(row[1:] if isinstance(row[0], str) else row, dtype=np.float64).reshape(8, 3)
        p9 = corners_to_9params(c).astype(np.float64)
        if p9[2] <= 0.5 or np.linalg.norm(p9[:3]) > max_label_range:
            continue
        out.append(p9)
    return out

PROTO = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
LIDAR_OFFSET = 6      # 已验证（build_mm_cache）
RADAR_OFFSET = 8      # 2026-09-06 验证：tag=1 的雷达回波与框中心距离 0.03-0.08 m（偏 6 帧时 0.5-2 m）


# ----------------------------------------------------------------------------- 几何工具
def box_corners(b):
    return (PROTO * b[3:6]) @ Rot.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]


def project(pts, K):
    uv = (K @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def hull_mask(shape_hw, uv, dilate, feather):
    """凸包软掩码 float32 [0,1]。"""
    m = np.zeros(shape_hw, np.uint8)
    hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2)).astype(np.int32)
    cv2.fillConvexPoly(m, hull, 255)
    if dilate > 0:
        m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    m = m.astype(np.float32) / 255.0
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


def fill_background(img, mask, sigma):
    """归一化卷积：掩码区域用周围像素抹成模糊背景。img uint8 (H,W[,3])。"""
    im = img.astype(np.float32)
    inv = (1.0 - mask)
    if im.ndim == 3:
        num = cv2.GaussianBlur(im * inv[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(inv, (0, 0), sigma)[..., None]
    else:
        num = cv2.GaussianBlur(im * inv, (0, 0), sigma)
        den = cv2.GaussianBlur(inv, (0, 0), sigma)
    fill = num / np.maximum(den, 1e-3)
    out = im * (inv if im.ndim == 2 else inv[..., None]) + fill * ((1 - inv) if im.ndim == 2 else (1 - inv)[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def rot_between(a, b):
    """把单位向量 a 转到 b 的最小旋转。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3)
    return Rot.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def inside_obb(pts, b):
    """点是否在 9 参数 OBB 内。"""
    Rm = Rot.from_euler('xyz', b[6:9]).as_matrix()
    local = (pts - b[:3]) @ Rm            # 到框系
    return np.all(np.abs(local) <= b[3:6] / 2.0 + 1e-6, axis=1)


def denoise_along_rays(pts, origin, b):
    """LiDAR 距离噪声是沿射线的（CARLA NoiseStdDev）：把每个点沿「传感器原点 -> 点」的射线收回到 OBB 内
    （slab 法求射线与框的交段，把射线参数夹到交段里；射线不穿框的点收到框中心所在的垂直平面）。
    返回去噪后的点和每条射线的单位方向。"""
    Rm = Rot.from_euler('xyz', b[6:9]).as_matrix()
    d = pts - origin
    t = np.linalg.norm(d, axis=1)
    dirs = d / np.maximum(t[:, None], 1e-9)
    o_l = (origin - b[:3]) @ Rm                  # 射线原点 / 方向到框系
    d_l = dirs @ Rm
    half = b[3:6] / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-half - o_l) / d_l
        t2 = (half - o_l) / d_l
    tmin = np.where(np.isfinite(t1), np.minimum(t1, t2), -np.inf)
    tmax = np.where(np.isfinite(t2), np.maximum(t1, t2), np.inf)
    t_in = tmin.max(axis=1)
    t_out = tmax.min(axis=1)
    hit = (t_out >= t_in) & (t_out > 0)
    t_c = float(np.dot(b[:3] - origin, dirs.mean(axis=0)))   # 框中心的近似射线距离
    t_new = np.where(hit, np.clip(t, np.maximum(t_in, 0), t_out), np.maximum(t_c, 0.1))
    return origin + dirs * t_new[:, None], dirs, hit


# ----------------------------------------------------------------------------- 数据读取
class Frame:
    """一帧的原始多模态数据 + 缓存索引里的窗口/标签。所有 3D 量在 RGB 相机 OpenCV 系。"""

    def __init__(self, root, meta, W, H):
        self.meta, self.W, self.H = meta, W, H
        base = os.path.join(root, meta['seq'])
        info = pickle.load(open(os.path.join(base, 'im_info.pkl'), 'rb'))
        lr = pickle.load(open(os.path.join(base, 'lidar_radar_info.pkl'), 'rb'))
        self.K = np.array(info['rgb']['intrinsic'], np.float64)           # 原生 K（1280x720）
        self.E = {m: np.array(info[m]['extrinsic'], np.float64) for m in ('rgb', 'ir', 'dvs')}
        self.Kc = {m: np.array(info[m]['intrinsic'], np.float64) for m in ('rgb', 'ir', 'dvs')}
        self.L_ext = np.array(lr['lidars'][0]['extrinsic'], np.float64)
        self.R_ext = np.array(lr['radars'][0]['extrinsic'], np.float64)
        self.rgb = cv2.imread(os.path.join(base, 'images_rgb', meta['frame']), cv2.IMREAD_COLOR)
        self.ir = cv2.imread(os.path.join(base, 'images_ir', meta['frame']), cv2.IMREAD_GRAYSCALE)
        self.dvs = cv2.imread(os.path.join(base, 'images_dvs', meta['frame']), cv2.IMREAD_COLOR)
        self.raw_h, self.raw_w = self.rgb.shape[:2]
        self.boxes = np.asarray(meta['boxes9d'], np.float64)
        self.names = list(meta['names'])
        self.qual = np.asarray(meta['qualified'], bool)
        self.boxes_full = load_full_boxes(base, meta['frame']) or [b for b in self.boxes]
        cands = [i for i in range(len(self.boxes)) if self.qual[i]]
        self.kept = min(cands, key=lambda i: np.linalg.norm(self.boxes[i, :3]))
        self.x0, self.y0 = meta['crop_xy']
        self.cw, self.ch = meta['crop_wh']
        self.K_win = self.K.copy()
        self.K_win[0, 2] -= self.x0
        self.K_win[1, 2] -= self.y0
        self.K_s = self.K_win.copy()                                     # 窗口 -> 512x288
        self.K_s[0] *= W / float(self.cw)
        self.K_s[1] *= H / float(self.ch)
        # IR / DVS 对齐到 RGB：在被保留目标深度处的视差整体平移（与缓存一致）
        cw_ = cam_to_world(self.boxes[self.kept][None, :3], self.E['rgb'])
        uv_rgb, _ = world_to_pixel(cw_, self.E['rgb'], self.K)
        self.shift = {}
        for m in ('ir', 'dvs'):
            uv_m, _ = world_to_pixel(cw_, self.E[m], self.Kc[m])
            self.shift[m] = (uv_rgb[0] - uv_m[0])
        self.ir_al = self._shifted(self.ir, self.shift['ir'])
        self.dvs_al = self._shifted(self.dvs, self.shift['dvs'])
        # 点云
        frames = sorted([f for f in os.listdir(os.path.join(base, 'images_rgb')) if f.endswith('.png')], key=frame_sort_key)
        fi = frames.index(meta['frame'])
        li = min(fi + LIDAR_OFFSET, len(frames) - 1)
        ri = min(fi + RADAR_OFFSET, len(frames) - 1)
        P = np.load(os.path.join(base, 'lidar_1', os.path.splitext(frames[li])[0] + '.npy'))
        self.lidar_raw = P.astype(np.float64)                             # (N,5) x,y,z,intensity,tag  LiDAR 系
        self.lidar_cam = self._lidar_to_cam(P[:, :3].astype(np.float64))
        rp = os.path.join(base, 'radar_1', os.path.splitext(frames[ri])[0] + '.npy')
        Q = np.load(rp).astype(np.float64) if os.path.exists(rp) else np.zeros((0, 5))
        self.radar_raw = Q                                                # (N,5) vel, az, alt, depth, tag  雷达系
        self.radar_cam = self._radar_to_cam(Q)
        # 每个 tag=1 的点归到最近的框（多无人机时分开）
        self.lidar_owner = self._assign(self.lidar_cam, self.lidar_raw[:, 4] == 1)
        self.radar_owner = self._assign(self.radar_cam, self.radar_raw[:, 4] == 1)

    def _shifted(self, img, shift):
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _lidar_to_cam(self, xyz):
        world = (self.L_ext @ np.c_[xyz, np.ones(len(xyz))].T).T
        return (CARLA_TO_OPENCV @ (np.linalg.inv(self.E['rgb']) @ world.T)).T[:, :3]

    def cam_to_lidar(self, cam):
        world = cam_to_world(cam, self.E['rgb'])
        return (np.linalg.inv(self.L_ext) @ np.c_[world, np.ones(len(world))].T).T[:, :3]

    def _radar_to_cam(self, Q):
        if len(Q) == 0:
            return np.zeros((0, 3))
        az, al, d = Q[:, 1], Q[:, 2], Q[:, 3]
        xyz = np.stack([d * np.cos(al) * np.cos(az), d * np.cos(al) * np.sin(az), d * np.sin(al)], 1)
        world = (self.R_ext @ np.c_[xyz, np.ones(len(xyz))].T).T
        return (CARLA_TO_OPENCV @ (np.linalg.inv(self.E['rgb']) @ world.T)).T[:, :3]

    def cam_to_radar(self, cam):
        world = cam_to_world(cam, self.E['rgb'])
        xyz = (np.linalg.inv(self.R_ext) @ np.c_[world, np.ones(len(world))].T).T[:, :3]
        d = np.linalg.norm(xyz, axis=1)
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        al = np.arcsin(np.clip(xyz[:, 2] / np.maximum(d, 1e-9), -1, 1))
        return az, al, d

    def _assign(self, cam, tagged):
        owner = -np.ones(len(cam), int)
        if tagged.any() and len(self.boxes):
            d = np.linalg.norm(cam[tagged][:, None, :] - self.boxes[None, :, :3], axis=2)   # (n, nbox)
            owner[np.where(tagged)[0]] = np.argmin(d, axis=1)
        return owner

    # ---- 窗口视图 ----
    def win(self, img):
        return img[self.y0:self.y0 + self.ch, self.x0:self.x0 + self.cw]

    def hull_full(self, b, dilate=10, feather=6):
        """原生全图坐标下的凸包软掩码。"""
        return hull_mask((self.raw_h, self.raw_w), project(box_corners(b), self.K), dilate, feather)

    def hull_in_cam(self, b, cam, dilate=6):
        """框凸包投到 IR / DVS 自己的相机里（不是 RGB 相机）：抹除要在各自的原始图上做，再按新目标深度对齐。"""
        cw = cam_to_world(box_corners(b), self.E['rgb'])
        uv, _ = world_to_pixel(cw, self.E[cam], self.Kc[cam])
        return hull_mask((self.raw_h, self.raw_w), uv, dilate, 0) > 0.5

    def erase_all(self, img, cam='rgb', radius=5, dilate=6):
        """把帧里【整帧全部】无人机用 cv2.inpaint（Telea）抹掉：比模糊填充更像背景的延续，SD 不会把大洞当成物体。
        cam='rgb' 时凸包按 RGB 相机投影；'ir'/'dvs' 按该相机投影（输入应是未平移的原始 IR / DVS 图）。
        DVS 事件图里运动的无人机会在「上一位置」留一条反极性的拖影（实测约 80 px），所以 DVS 的抹除范围外扩得多。"""
        if cam == 'dvs':
            dilate = max(dilate, 60)
        m = np.zeros((self.raw_h, self.raw_w), np.uint8)
        for bb in self.boxes_full:
            hm = (self.hull_full(bb, dilate=dilate, feather=0) > 0.5) if cam == 'rgb' else self.hull_in_cam(bb, cam, dilate)
            m = np.maximum(m, hm.astype(np.uint8))
        if not m.any():
            return img.copy()
        return cv2.inpaint(img, m, radius, cv2.INPAINT_TELEA)

    def erased(self, cam):
        """抹掉全部目标后的原始分辨率图（未平移），按需计算并缓存。"""
        if not hasattr(self, '_erased'):
            self._erased = {}
        if cam not in self._erased:
            src = {'rgb': self.rgb, 'ir': self.ir, 'dvs': self.dvs}[cam]
            self._erased[cam] = self.erase_all(src, cam)
        return self._erased[cam]

    def shift_for(self, cam, c):
        """IR / DVS 对齐到 RGB 所需的平移：3D 点 c（RGB 相机系）处的视差。"""
        cw = cam_to_world(np.asarray(c, np.float64)[None], self.E['rgb'])
        uv_rgb, _ = world_to_pixel(cw, self.E['rgb'], self.K)
        uv_m, _ = world_to_pixel(cw, self.E[cam], self.Kc[cam])
        return uv_rgb[0] - uv_m[0]

    def drone_matte(self, b, feather=1.0):
        """无人机的紧致 alpha（原生全图）：不是框的凸包，而是框内「与背景不同」的像素。
        先验 = 像素与「用外环像素填进凸包的背景估计」的差；差大 = 前景、差小 = 背景；再用 GrabCut 收边。
        返回 (alpha float32, 前景面积/凸包面积)。GrabCut 失败（前景太小）时退回差异阈值掩码。"""
        uv = project(box_corners(b), self.K)
        hull = hull_mask((self.raw_h, self.raw_w), uv, 2, 0) > 0.5
        ys, xs = np.where(hull)
        pad = 24
        y0, y1 = max(0, ys.min() - pad), min(self.raw_h, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(self.raw_w, xs.max() + pad + 1)
        crop = self.rgb[y0:y1, x0:x1]
        hc = hull[y0:y1, x0:x1]
        fill = fill_background(crop, hc.astype(np.float32), sigma=12)
        diff = np.abs(crop.astype(np.float32) - fill.astype(np.float32)).sum(-1)
        gc = np.full(hc.shape, cv2.GC_BGD, np.uint8)
        gc[hc] = cv2.GC_PR_FGD
        gc[hc & (diff < 30)] = cv2.GC_PR_BGD
        gc[hc & (diff > 120)] = cv2.GC_FGD
        fg = None
        if (gc == cv2.GC_FGD).sum() >= 10:
            try:
                bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
                cv2.grabCut(np.ascontiguousarray(crop), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
                fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & hc
            except cv2.error:
                fg = None
        if fg is None or fg.sum() < 0.02 * hc.sum():
            fg = hc & (diff > 45)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # 不外扩：外扩 1 px 会把原背景的天空色带成一圈亮边（第一版能看见青色描边）
        a = np.zeros((self.raw_h, self.raw_w), np.float32)
        a[y0:y1, x0:x1] = fg.astype(np.float32)
        if feather > 0:
            a = cv2.GaussianBlur(a, (0, 0), feather)
        a = np.clip(a, 0, 1)
        # 去色染（matting decontamination）：边缘像素 O = a*F + (1-a)*Bg_A，把 A 帧背景色 Bg_A（fill 估计）解出来，
        # 否则天空/雾的颜色会跟着机身边缘一起贴过去，形成一圈亮边
        ac = a[y0:y1, x0:x1]
        edge = (ac > 0.02) & (ac < 0.98)
        F = crop.astype(np.float32).copy()
        if edge.any():
            Fe = (crop[edge].astype(np.float32) - (1 - ac[edge][:, None]) * fill[edge].astype(np.float32)) / np.maximum(ac[edge][:, None], 0.25)
            F[edge] = np.clip(Fe, 0, 255)
        rgb_clean = self.rgb.copy()
        rgb_clean[y0:y1, x0:x1] = F.astype(np.uint8)
        return a, float(fg.sum()) / max(float(hc.sum()), 1.0), rgb_clean

    def render_depth_tag(self, cam_pts, tags):
        """点云（相机系）-> 512x288 深度(cm, uint16) / tag。与 build_mm_cache 相同的 z-buffer 规则。"""
        W, H = self.W, self.H
        depth = np.zeros((H, W), np.uint16)
        tag = np.zeros((H, W), np.uint8)
        if len(cam_pts) == 0:
            return depth, tag
        m = cam_pts[:, 2] > 0.1
        cv_, t = cam_pts[m], tags[m]
        uv = project(cv_, self.K_s)
        ui, vi = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ins = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, z, t = ui[ins], vi[ins], cv_[ins, 2], t[ins]
        order = np.argsort(-z)
        zc = np.clip(z * 100.0, 1, 65535).astype(np.uint16)
        depth[vi[order], ui[order]] = zc[order]
        tag[vi[t == 1], ui[t == 1]] = 1
        return depth, tag

    def render_radar_hm(self, cam_pts, vel):
        """雷达点（相机系）-> 512x288 速度热图（与 laam6d dataset_utils.radar_to_velocity_heatmap 同思路：投影、取 max、模糊）。"""
        W, H = self.W, self.H
        hm = np.zeros((H, W), np.float32)
        if len(cam_pts) == 0:
            return hm
        m = cam_pts[:, 2] > 0.1
        uv = project(cam_pts[m], self.K_s)
        v = np.abs(vel[m])
        ui, vi = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ins = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        for u, vv, val in zip(ui[ins], vi[ins], v[ins]):
            hm[vv, u] = max(hm[vv, u], val + 0.05)      # +0.05：静止回波也留个底，能看见有点
        return cv2.GaussianBlur(hm, (7, 7), 0)


# ----------------------------------------------------------------------------- 贴图
def paste(A, B, plates, rng, args):
    """把 A 的被保留无人机贴到 B 的干净背景。plates = dict(rgb=翻译背景, rgb_sim=抹除后的仿真背景, ir, dvs)，均 512x288。
    返回 dict 或 None（位置不合法）。"""
    W, H = A.W, A.H
    b = A.boxes[A.kept]
    c = b[:3]
    Rb = Rot.from_euler('xyz', b[6:9]).as_matrix()
    uvA = project(c[None], A.K)[0]                                 # A 原生全图像素
    lid_idx = np.where(A.lidar_owner == A.kept)[0]
    rad_idx = np.where(A.radar_owner == A.kept)[0]
    if len(lid_idx) < args.min_lidar_pts:
        return None
    # B 的背景点云：去掉全部无人机的点
    B_lid_bg = A_none = None
    B_lid_bg = np.where(B.lidar_owner < 0)[0]
    B_rad_bg = np.where(B.radar_owner < 0)[0]
    # B 背景深度（遮挡检查用）
    depth_bg, _ = B.render_depth_tag(B.lidar_cam[B_lid_bg], np.zeros(len(B_lid_bg)))
    # 目前表观大小 = 8 角点投影框的长边（B 的缓存像素；与 MAV6D 统计口径一致）
    uv0 = project(box_corners(b), B.K_s)
    px_now = float(max(uv0[:, 0].max() - uv0[:, 0].min(), uv0[:, 1].max() - uv0[:, 1].min()))

    r_old = float(np.linalg.norm(c))
    for _ in range(args.tries):
        if getattr(args, 'identity', False):
            # 自检：A=B、同位置、同距离 —— 走完整条链路后一切都应回到原样
            s, px_new = 1.0, px_now
            uvB = uvA.copy()
            R_cam = np.eye(3)
            c_new = c.copy()
            t = np.zeros(3)
            R_new = Rb.copy()
            b_new = b.copy()
            uv_new = project(box_corners(b_new), B.K_s)
            break
        if args.range_ref_arr is not None:
            # 用户定的规则（2026-09-06）：先抽 MAV6D 的距离 r_ref（2-5 m），再按机型大小等比例放远：
            #   r_new = r_ref * max(dim) / ref_dim   —— 小机型 2-5 m，大机型按比例 5-10 m 甚至更远，
            # 表观大小与 MAV6D 同量级、整框在视野内、看得清；缩放倍数由距离物理推出 s = r_old / r_new。
            r_ref = float(rng.choice(args.range_ref_arr))
            r_new = r_ref * float(max(b[3:6])) / args.ref_dim
            s = r_old / max(r_new, 1e-3)
            if not (args.scale[0] <= s <= args.scale[1]):
                continue
            px_new = px_now * s
            if not (args.px_vis[0] <= px_new <= args.px_vis[1]):      # 看得清（>= 40 px）且放得下（<= 200 px）
                continue
        elif args.size_ref_arr is not None:
            # 只按尺寸分布抽（没有距离参考时）
            px_target = float(rng.choice(args.size_ref_arr))
            s = px_target / max(px_now, 1e-3)
            if not (args.scale[0] <= s <= args.scale[1]):
                continue
            px_new = px_now * s
        else:
            s = float(rng.uniform(args.scale[0], args.scale[1]))
            px_new = px_now * s
            if not (args.px[0] <= px_new <= args.px[1]):
                continue
        # 目标位置：B 窗口内（缓存坐标）留边
        mg = px_new * 0.8 + 6
        if B.W - 2 * mg <= 0 or B.H - 2 * mg <= 0:
            continue
        u_c = float(rng.uniform(mg, B.W - mg))
        v_c = float(rng.uniform(mg, B.H - mg))
        # 换到 B 原生全图像素
        uvB = np.array([u_c * B.cw / B.W + B.x0, v_c * B.ch / B.H + B.y0])
        rA = np.linalg.inv(A.K) @ np.array([uvA[0], uvA[1], 1.0])
        rB = np.linalg.inv(B.K) @ np.array([uvB[0], uvB[1], 1.0])
        R_cam = rot_between(rA, rB)
        lam = 1.0 / s
        c_new = lam * (R_cam @ c)
        t = c_new - R_cam @ c
        R_new = R_cam @ Rb
        b_new = np.concatenate([c_new, b[3:6], Rot.from_matrix(R_new).as_euler('xyz')])
        # 新框 8 角点必须在 B 窗口内（缓存坐标）
        uv_new = project(box_corners(b_new), B.K_s)
        if not (np.all(uv_new[:, 0] >= 2) and np.all(uv_new[:, 0] < B.W - 2) and np.all(uv_new[:, 1] >= 2) and np.all(uv_new[:, 1] < B.H - 2)):
            continue
        # 遮挡：贴入区域内 B 背景不能有比 c_new 更近的回波
        hm_new = hull_mask((H, W), uv_new, 2, 0) > 0.5
        dz = depth_bg[hm_new]
        dz = dz[dz > 0]
        if dz.size and dz.min() / 100.0 < c_new[2] - 0.5:
            continue
        break
    else:
        return None

    # ---- 像素：单应 + 绕新中心的相似缩放（原生全图坐标）----
    Hrot = B.K @ R_cam @ np.linalg.inv(A.K)
    cB = project(c_new[None], B.K)[0]
    S = np.array([[s, 0, (1 - s) * cB[0]], [0, s, (1 - s) * cB[1]], [0, 0, 1.0]])
    Ht = S @ Hrot
    if args.hull_alpha:
        alpha_full, matte_frac, rgb_clean = A.hull_full(b, dilate=args.dilate, feather=args.feather), 1.0, A.rgb
    else:
        alpha_full, matte_frac, rgb_clean = A.drone_matte(b)

    def warp(img, interp=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, Ht, (B.raw_w, B.raw_h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def to_cache(img_full):
        return cv2.resize(B.win(img_full), (W, H), interpolation=cv2.INTER_AREA)

    alpha = to_cache(warp(alpha_full)).astype(np.float32)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    # B 的底图：RGB 学生版 = SD 翻译过的（--make-plates 抹除 + 翻译脚本）；其余现算：
    # 全部目标 inpaint 抹掉，IR / DVS 在各自原始图上抹再按【新目标深度】平移对齐（与缓存「在目标深度对齐」的约定一致，
    # 否则 B 原来的近目标会带来几百像素的平移把半幅图移出画面）
    plates = dict(plates)
    plates['rgb_sim'] = to_cache(B.erased('rgb'))
    plates['ir'] = to_cache(B._shifted(B.erased('ir'), B.shift_for('ir', c_new)))
    plates['dvs'] = to_cache(B._shifted(B.erased('dvs'), B.shift_for('dvs', c_new)))

    # ---- 色调匹配：无人机的亮度 / 对比度向贴入处背景的统计靠拢（雾天自然变淡、阴天变灰）----
    # A 帧无人机周围的背景环（原生坐标） vs B 底图贴入区域（缓存坐标），逐通道 I' = I + tone*[(I-muA)*(sigB/sigA) + muB - I]
    a_hard = alpha_full > 0.05
    ring = (cv2.dilate(a_hard.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~a_hard
    dst_region = cv2.dilate((alpha > 0.05).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0

    def tone(plate):
        if args.tone <= 0 or ring.sum() < 50 or dst_region.sum() < 50:
            return rgb_clean
        muA, sgA = A.rgb[ring].astype(np.float32).mean(0), A.rgb[ring].astype(np.float32).std(0) + 1.0
        muB, sgB = plate[dst_region].astype(np.float32).mean(0), plate[dst_region].astype(np.float32).std(0) + 1.0
        ratio = np.clip(sgB / sgA, 0.5, 1.5)
        I = rgb_clean.astype(np.float32)
        J = (I - muA) * ratio + muB
        return np.clip(I + args.tone * (J - I), 0, 255).astype(np.uint8)

    spr = {'rgb': to_cache(warp(tone(plates['rgb']) * alpha_full[..., None])),
           'rgb_sim': to_cache(warp(tone(plates['rgb_sim']) * alpha_full[..., None])),
           'ir': to_cache(warp(A.ir_al * alpha_full)),
           'dvs': to_cache(warp(A.dvs_al * alpha_full[..., None]))}

    # 预乘 alpha 已经在 sprite 里：合成 = plate*(1-alpha) + sprite
    def comp(plate, sprite, poisson=False):
        a = alpha[..., None] if plate.ndim == 3 else alpha
        img = np.clip(plate.astype(np.float32) * (1 - a) + sprite.astype(np.float32), 0, 255).astype(np.uint8)
        if poisson and plate.ndim == 3:
            # 泊松融合：保留机身内部梯度、边界颜色与背景连续。掩码不能贴到图像边界
            m = (cv2.dilate((alpha > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8)) * 255)
            m[:2, :] = m[-2:, :] = 0
            m[:, :2] = m[:, -2:] = 0
            ys, xs = np.where(m > 0)
            if len(ys) > 20:
                ctr = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
                try:
                    img = cv2.seamlessClone(img, plate, m, ctr, cv2.NORMAL_CLONE)
                except cv2.error:
                    pass
        return img

    out = {
        'rgb': comp(plates['rgb'], spr['rgb'], args.poisson), 'rgb_sim': comp(plates['rgb_sim'], spr['rgb_sim'], args.poisson),
        'ir': comp(plates['ir'], spr['ir']), 'dvs': comp(plates['dvs'], spr['dvs']), 'alpha': alpha,
        'matte_frac': matte_frac,
    }
    # ---- 点云：无人机的点刚体变换到 B ----
    # LiDAR：先沿 A 的 LiDAR 射线去噪（收回框内），刚体变换，再沿 B 的 LiDAR 射线按 B 序列的 σ 重新加噪——
    # 噪声方向 / 大小都是 B 的传感器的，不是把 A 的噪声整体搬过去（LiDAR 离相机 2.4 m，绕相机中心转会让散布方向偏）
    lid_A = A.lidar_cam[lid_idx]
    o_A = A._lidar_to_cam(np.zeros((1, 3)))[0]
    o_B = B._lidar_to_cam(np.zeros((1, 3)))[0]
    lid_A_clean, _, hit_A = denoise_along_rays(lid_A, o_A, b)
    lid_clean_cam = lid_A_clean @ R_cam.T + t
    sigma_B = float(max(B.meta.get('lidar_noise', 0.0), 0.0))
    dirs_B = lid_clean_cam - o_B
    dirs_B /= np.maximum(np.linalg.norm(dirs_B, axis=1, keepdims=True), 1e-9)
    if getattr(args, 'identity', False):
        lid_new_cam = lid_A @ R_cam.T + t            # 自检不去噪不加噪：点应逐个回到原坐标
    else:
        lid_new_cam = lid_clean_cam + dirs_B * (rng.randn(len(lid_idx), 1) * sigma_B)
    lid_new_raw = np.c_[B.cam_to_lidar(lid_new_cam), A.lidar_raw[lid_idx, 3], np.ones(len(lid_idx))]
    lidar_aug = np.r_[B.lidar_raw[B_lid_bg], lid_new_raw]
    cam_all = np.r_[B.lidar_cam[B_lid_bg], lid_new_cam]
    tags_all = np.r_[np.zeros(len(B_lid_bg)), np.ones(len(lid_idx))]
    out['depth'], out['tag'] = B.render_depth_tag(cam_all, tags_all)
    out['lidar_pts'] = lidar_aug
    rad_A = A.radar_cam[rad_idx]
    rad_new_cam = rad_A @ R_cam.T + t
    if len(rad_idx):
        az, al, d = B.cam_to_radar(rad_new_cam)
        rad_new_raw = np.c_[A.radar_raw[rad_idx, 0], az, al, d, np.ones(len(rad_idx))]
    else:
        rad_new_raw = np.zeros((0, 5))
    radar_aug = np.r_[B.radar_raw[B_rad_bg], rad_new_raw]
    rcam_all = np.r_[B.radar_cam[B_rad_bg], rad_new_cam]
    rvel_all = np.r_[B.radar_raw[B_rad_bg, 0], A.radar_raw[rad_idx, 0]]
    out['radar_hm'] = B.render_radar_hm(rcam_all, rvel_all)
    out['radar_pts'] = radar_aug
    # ---- 标签 / 变换 / 校验 ----
    out['box9d'] = b_new.astype(np.float32)
    out['name'] = A.names[A.kept]
    out['K_raw'] = B.K_win.astype(np.float32)
    out['raw_wh'] = (B.cw, B.ch)
    out['transform'] = {'R_cam': R_cam, 't': t, 's': s, 'range_old': float(np.linalg.norm(c)), 'range_new': float(np.linalg.norm(c_new))}
    frac_old = float(inside_obb(lid_A, b).mean())
    frac_new = float(inside_obb(lid_new_cam, b_new).mean())
    frac_clean = float(inside_obb(lid_clean_cam, b_new).mean())
    # A 帧原始 LiDAR（A 自己的窗口）留给查看器对照
    out['depth_src'], out['tag_src'] = A.render_depth_tag(A.lidar_cam, (A.lidar_owner == A.kept).astype(int))
    out['uv_src'] = project(box_corners(b), A.K_s)
    out['rgb_src'] = cv2.resize(A.win(A.rgb), (W, H), interpolation=cv2.INTER_AREA)
    rad_old = float(inside_obb(rad_A, b).mean()) if len(rad_idx) else -1
    rad_new = float(inside_obb(rad_new_cam, b_new).mean()) if len(rad_idx) else -1
    # 贴上的像素区域 vs 新框凸包（缓存坐标）
    hull_new = hull_mask((H, W), uv_new, args.dilate * W // B.cw + 1, 0) > 0.5
    a_bin = alpha > 0.5
    iou = float((hull_new & a_bin).sum()) / max(float((hull_new | a_bin).sum()), 1.0)
    # 新框在 B 的 512 分辨率下的表观大小
    # 贴上去的像素（alpha>0.5）应全部落在新框凸包内（紧致 matte 比凸包小，所以看「alpha 在凸包内的比例」而不是 IoU）
    a_in = float((hull_new & a_bin).sum()) / max(float(a_bin.sum()), 1.0)
    out['checks'] = {'lidar_in_box_old': frac_old, 'lidar_in_box_new': frac_new, 'lidar_in_box_denoised': frac_clean,
                     'lidar_ray_hit_frac': float(hit_A.mean()), 'sigma_A': float(max(A.meta.get('lidar_noise', 0.0), 0.0)),
                     'sigma_B': sigma_B, 'lidar_n': int(len(lid_idx)),
                     'radar_in_box_old': rad_old, 'radar_in_box_new': rad_new, 'radar_n': int(len(rad_idx)),
                     'alpha_vs_box_iou': iou, 'alpha_in_hull': a_in, 'matte_frac': matte_frac, 'px_new': float(px_new),
                     'corners_inside': True,
                     'tag_px_in_hull': float((out['tag'] > 0)[hull_new].sum() / max((out['tag'] > 0).sum(), 1))}
    out['uv_new'] = uv_new
    return out


# ----------------------------------------------------------------------------- 可视化
def draw_box_img(img, b, K, color=(60, 220, 60), width=1, label=None):
    uv = project(box_corners(b), K)
    for i, j in EDGES:
        cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[j]).astype(int)), color, width, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (int(uv[:, 0].min()), max(10, int(uv[:, 1].min()) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return img


def depth_vis(depth, tag):
    d = depth.astype(np.float32) / 100.0
    img = np.zeros(depth.shape + (3,), np.uint8)
    m = depth > 0
    if m.any():
        n = np.clip(d / 40.0, 0, 1)
        col = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_JET)
        img[m] = col[m]
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    tm = cv2.dilate(tag, np.ones((3, 3), np.uint8)) > 0
    img[tm] = (255, 0, 255)
    return img


def hm_vis(hm):
    n = hm / max(float(hm.max()), 1e-6)
    return cv2.applyColorMap((255 * n).astype(np.uint8), cv2.COLORMAP_INFERNO)


def bev(cam_pts, tags, b, size=(512, 288), rng_m=(-8, 8, 0, 24)):
    """相机系俯视 (x 右, z 前)。灰=背景点，品红=无人机点，绿=框。"""
    W, H = size
    img = np.full((H, W, 3), 25, np.uint8)
    xl, xr, z0, z1 = rng_m

    def to_px(x, z):
        return (np.clip((x - xl) / (xr - xl) * (W - 1), 0, W - 1)).astype(int), (np.clip((1 - (z - z0) / (z1 - z0)) * (H - 1), 0, H - 1)).astype(int)
    if len(cam_pts):
        m = (cam_pts[:, 0] >= xl) & (cam_pts[:, 0] <= xr) & (cam_pts[:, 2] >= z0) & (cam_pts[:, 2] <= z1)
        u, v = to_px(cam_pts[m, 0], cam_pts[m, 2])
        img[v, u] = (140, 140, 140)
        tm = m & (tags == 1)
        u, v = to_px(cam_pts[tm, 0], cam_pts[tm, 2])
        img[v, u] = (255, 0, 255)
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    cs = box_corners(b)[[0, 1, 2, 3]]             # 底面四角
    u, v = to_px(cs[:, 0], cs[:, 2])
    cv2.polylines(img, [np.stack([u, v], 1).reshape(-1, 1, 2).astype(np.int32)], True, (60, 220, 60), 1, cv2.LINE_AA)
    for k, z in enumerate(range(int(z0), int(z1) + 1, 4)):
        _, vv = to_px(np.array([0.0]), np.array([float(z)]))
        cv2.putText(img, '%dm' % z, (2, int(vv[0]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1)
    return img


def lidar_panel_user(B, lidar_pts, img_shape=(288, 512)):
    """用用户原代码 project_lidar_and_get_uvz_rgb_tag 投影增广后的点云（B 的 LiDAR 系），按深度上色，tag=1 品红。"""
    res = project_lidar_and_get_uvz_rgb_tag(lidar_pts.astype(np.float64), B.L_ext, B.E['rgb'], B.K_s,
                                            np.zeros(img_shape + (3,), np.uint8))
    img = np.zeros(img_shape + (3,), np.uint8)
    if len(res) == 0:
        return img
    z = np.clip(res[:, 2] / 40.0, 0, 1)
    cols = cv2.applyColorMap((255 * (1 - z)).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    for (u, v), c, t in zip(res[:, :2], cols, res[:, 9]):
        if t == 1:
            cv2.circle(img, (int(round(u)), int(round(v))), 2, (255, 0, 255), -1)
        else:
            cv2.circle(img, (int(round(u)), int(round(v))), 1, tuple(int(x) for x in c), -1)
    return img


def radar_panel_user(B, radar_pts, img_shape=(288, 512)):
    """用用户原代码 radar_to_velocity_heatmap（投影 + 取 max + 5x5 高斯）出速度热图，JET 上色。"""
    if len(radar_pts) == 0:
        return np.zeros(img_shape + (3,), np.uint8)
    hm = radar_to_velocity_heatmap(radar_pts.astype(np.float64), B.R_ext, B.E['rgb'], B.K_s, image_shape=img_shape, method='max')[0]
    n = hm / max(float(hm.max()), 1e-6)
    return cv2.applyColorMap((255 * n).astype(np.uint8), cv2.COLORMAP_JET)


def make_gallery(A, B, out, path, up=1.5):
    """精简版：RGB(翻译背景)+框 | IR+框 | DVS+框 / LiDAR 投影+框 | 雷达热图+框 | RGB 放大 3x。返回一行 5 格的缩略图给总览用。"""
    W, H = A.W, A.H
    b_new, K_s = out['box9d'], B.K_s
    g = (60, 220, 60)
    tr, ck = out['transform'], out['checks']
    rgb = draw_box_img(out['rgb'].copy(), b_new, K_s, g, 1, '%s %.1fm' % (out['name'], tr['range_new']))
    ir3 = draw_box_img(cv2.cvtColor(out['ir'], cv2.COLOR_GRAY2BGR), b_new, K_s, g, 1)
    dvs = draw_box_img(out['dvs'].copy(), b_new, K_s, g, 1)
    lid = draw_box_img(lidar_panel_user(B, out['lidar_pts']), b_new, K_s, g, 1)
    rad = draw_box_img(radar_panel_user(B, out['radar_pts']), b_new, K_s, g, 1)
    uv = out['uv_new']
    cx, cy = float(uv[:, 0].mean()), float(uv[:, 1].mean())
    zw, zh = 170, 96
    x0 = int(np.clip(cx - zw / 2, 0, W - zw))
    y0 = int(np.clip(cy - zh / 2, 0, H - zh))
    zoom = cv2.resize(rgb[y0:y0 + zh, x0:x0 + zw], (W, H), interpolation=cv2.INTER_NEAREST)
    panels = [('RGB (translated bg) + box', rgb), ('IR + box', ir3), ('DVS + box', dvs),
              ('LiDAR projection (user code) + box', lid), ('radar velocity heatmap (user code) + box', rad), ('RGB zoom 3x', zoom)]
    pw, ph = int(W * up), int(H * up)
    sheet = np.full((2 * (ph + 22), 3 * (pw + 6), 3), 30, np.uint8)
    for k, (t, im) in enumerate(panels):
        r, c = divmod(k, 3)
        x, y = c * (pw + 6), r * (ph + 22)
        sheet[y + 22:y + 22 + ph, x:x + pw] = cv2.resize(im, (pw, ph), interpolation=cv2.INTER_LINEAR)
        cv2.putText(sheet, t, (x + 6, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    info = 'A %s/%s -> B %s/%s | %s dim %.2fm | range %.1f->%.1f m | s=%.2f | %.0f px | matte %.2f' % (
        A.meta['seq'].split('/')[3], os.path.splitext(A.meta['frame'])[0], B.meta['seq'].split('/')[3], os.path.splitext(B.meta['frame'])[0],
        out['name'], float(max(A.boxes[A.kept][3:6])), tr['range_old'], tr['range_new'], tr['s'], ck['px_new'], ck['matte_frac'])
    cv2.putText(sheet, info, (pw + 12, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tw, th = 384, 216
    strip = np.full((th + 16, 5 * (tw + 4), 3), 30, np.uint8)
    for k, im in enumerate([rgb, ir3, dvs, lid, rad]):
        strip[16:16 + th, k * (tw + 4):k * (tw + 4) + tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.putText(strip, '%s | %s | %s %.1fm | %.0f px | s=%.2f' % (os.path.basename(path)[:-4], A.meta['seq'].split('/')[3], out['name'],
                tr['range_new'], ck['px_new'], tr['s']), (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    return strip


def make_sheet(A, B, out, path, up=2):
    W, H = A.W, A.H
    b_new, K_s = out['box9d'], B.K_s
    tr, ck = out['transform'], out['checks']
    g, y = (60, 220, 60), (40, 200, 235)
    # 行 1
    a_img = cv2.resize(A.win(A.rgb), (W, H), interpolation=cv2.INTER_AREA)
    for i, bb in enumerate(A.boxes):
        draw_box_img(a_img, bb, A.K_s, g if i == A.kept else y, 1, '%.1fm' % np.linalg.norm(bb[:3]) if i == A.kept else None)
    b_img = cv2.resize(B.win(B.rgb), (W, H), interpolation=cv2.INTER_AREA)
    for bb in B.boxes:
        draw_box_img(b_img, bb, B.K_s, y, 1)
    aug = draw_box_img(out['rgb'].copy(), b_new, K_s, g, 1, '%.1fm' % tr['range_new'])
    aug_sim = draw_box_img(out['rgb_sim'].copy(), b_new, K_s, g, 1)
    row1 = [('A: source (green = pasted drone)', a_img), ('B: background frame (yellow = erased)', b_img),
            ('AUG rgb (translated bg) + new box', aug), ('AUG rgb_teacher (sim bg)', aug_sim)]
    # 行 2
    ir3 = cv2.cvtColor(out['ir'], cv2.COLOR_GRAY2BGR)
    row2 = [('AUG ir', draw_box_img(ir3, b_new, K_s, g, 1)), ('AUG dvs', draw_box_img(out['dvs'].copy(), b_new, K_s, g, 1)),
            ('AUG lidar depth + tag(magenta), from re-projected cloud', draw_box_img(depth_vis(out['depth'], out['tag']), b_new, K_s, g, 1)),
            ('AUG radar velocity heatmap, from re-projected cloud', draw_box_img(hm_vis(out['radar_hm']), b_new, K_s, g, 1))]
    # 行 3：放大 + BEV
    uv = out['uv_new']
    cx, cy = float(uv[:, 0].mean()), float(uv[:, 1].mean())
    zw, zh = 128, 72
    x0 = int(np.clip(cx - zw / 2, 0, W - zw))
    y0 = int(np.clip(cy - zh / 2, 0, H - zh))
    z1 = cv2.resize(aug[y0:y0 + zh, x0:x0 + zw], (W, H), interpolation=cv2.INTER_NEAREST)
    z2 = cv2.resize(out['rgb'][y0:y0 + zh, x0:x0 + zw], (W, H), interpolation=cv2.INTER_NEAREST)
    lidA = A.lidar_cam
    bevA = bev(lidA, (A.lidar_owner == A.kept).astype(int), A.boxes[A.kept])
    cam_all = np.r_[B.lidar_cam[B.lidar_owner < 0], A.lidar_cam[A.lidar_owner == A.kept] @ tr['R_cam'].T + tr['t']]
    tags_all = np.r_[np.zeros((B.lidar_owner < 0).sum()), np.ones((A.lidar_owner == A.kept).sum())]
    bevB = bev(cam_all, tags_all, b_new)
    row3 = [('AUG zoom 4x (box)', z1), ('AUG zoom 4x (no box)', z2),
            ('BEV lidar: A original (magenta = drone pts)', bevA), ('BEV lidar: B after paste (rigid-moved pts)', bevB)]
    rows = [row1, row2, row3]
    pw, ph = W * up, H * up
    sheet = np.full((3 * (ph + 22), 4 * (pw + 6), 3), 30, np.uint8)
    for r, row in enumerate(rows):
        for cidx, (t, im) in enumerate(row):
            x, yy = cidx * (pw + 6), r * (ph + 22)
            sheet[yy + 22:yy + 22 + ph, x:x + pw] = cv2.resize(im, (pw, ph), interpolation=cv2.INTER_NEAREST if r == 2 and cidx < 2 else cv2.INTER_LINEAR)
            cv2.putText(sheet, t, (x + 6, yy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    info = ('A %s/%s  ->  B %s/%s | s=%.2f range %.1f->%.1f m | px %.0f | lidar in-box %.2f->%.2f (%d pts) | radar in-box %.2f->%.2f (%d) | matte %.2f of hull, %.2f inside new hull | tag px in hull %.2f'
            % (A.meta['seq'].split('/')[3], os.path.splitext(A.meta['frame'])[0], B.meta['seq'].split('/')[3], os.path.splitext(B.meta['frame'])[0],
               tr['s'], tr['range_old'], tr['range_new'], ck['px_new'], ck['lidar_in_box_old'], ck['lidar_in_box_new'], ck['lidar_n'],
               ck['radar_in_box_old'], ck['radar_in_box_new'], ck['radar_n'], ck['matte_frac'], ck['alpha_in_hull'], ck['tag_px_in_hull']))
    cv2.putText(sheet, info, (2 * (pw + 6) + 6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ----------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='变焦缓存（--zoom-to-mav6d）')
    ap.add_argument('--erased', required=True, help='--make-plates 的输出目录：全部目标用 cv2.inpaint 抹掉的 rgb/ir/dvs（512x288）')
    ap.add_argument('--plates', default='', help='sim2real_bg_translate --no-paste --no-erase 对 --erased 的翻译结果（学生用的真实感背景）')
    ap.add_argument('--make-plates', action='store_true', help='只做第一步：生成 --erased 目录（之后用 SD 翻译它的 rgb.npy）')
    ap.add_argument('--hull-alpha', action='store_true', help='用框凸包当 alpha（旧做法，会把无人机周围的天空一起贴过去；默认用紧致 matte）')
    ap.add_argument('--split', default='train')
    ap.add_argument('--root', default='E:/data_collect')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--out', default='../output/paste_demo')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scale', type=float, nargs=2, default=[0.5, 2.0], help='允许的表观缩放区间（>1 = 拉近）；超出就换目标尺寸/换源帧')
    ap.add_argument('--px', type=float, nargs=2, default=[30, 120], help='无 --size-ref 时：贴入后表观大小区间（缓存像素）')
    ap.add_argument('--size-ref', default='',
                    help='MAV6D 表观大小经验分布（.npy，网络输入分辨率下 8 角点投影框的长边像素）：每次贴入的目标尺寸直接从里面抽，'
                         '贴出来的尺寸分布严格等于 MAV6D 的（tools/mav6d_size_stats.py 生成）')
    ap.add_argument('--range-ref', default='',
                    help='MAV6D 目标距离经验分布（.npy，米；mav6d_size_stats.py 一起写出的 *_range_m.npy）：'
                         '贴入时先抽距离，缩放由距离物理推出（用户 2026-09-06：距离 range 要和 MAV6D 差不多）')
    ap.add_argument('--max-dim', type=float, default=9.0, help='源无人机最大边长上限（m），默认不限（各机型都要有）')
    ap.add_argument('--ref-dim', type=float, default=0.34,
                    help='等比例放远的参考边长 = MAV6D 无人机的框边长 0.34 m：r_new = r_MAV6D * max(dim)/0.34，'
                         '各机型的表观大小与 MAV6D 同分布（phantom 级 3-8 m、m210 级 5-12 m、Matrice-600 级 11-26 m）')
    ap.add_argument('--px-vis', type=float, nargs=2, default=[40, 140], help='贴入后表观大小的合理区间（MAV6D 的 1%%-99%% 分位）')
    ap.add_argument('--tone', type=float, default=0.6, help='色调匹配强度 0-1（0 = 关）')
    ap.add_argument('--poisson', action='store_true', help='RGB 用泊松融合（seamlessClone）而不是纯 alpha 合成')
    ap.add_argument('--gallery', action='store_true', help='出精简版样图（2x3：RGB/IR/DVS/LiDAR投影/雷达热图/放大）+ 每 20 张一页的总览')
    ap.add_argument('--selftest', type=int, default=0,
                    help='坐标链路自检：取 N 帧，A=B、恒等变换走完整条链路，检查 LiDAR/雷达点逐点回到原坐标、深度图与缓存一致、框不变')
    ap.add_argument('--balance-classes', action='store_true', default=True, help='源机型轮流选，保证每个机型都有')
    ap.add_argument('--no-sheets', action='store_true', help='不出大图（只做统计 / 批量生成）')
    ap.add_argument('--tries', type=int, default=30)
    ap.add_argument('--min-lidar-pts', type=int, default=8)
    ap.add_argument('--dilate', type=int, default=10)
    ap.add_argument('--feather', type=int, default=6)
    ap.add_argument('--sigma', type=float, default=25.0)
    ap.add_argument('--min-vis', type=float, default=5.0)
    ap.add_argument('--max-cache-frames', type=int, default=300, help='内存里最多缓存多少帧原始数据（LRU）')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    args.size_ref_arr = None
    args.range_ref_arr = None
    if args.size_ref:
        args.size_ref_arr = np.load(args.size_ref).astype(np.float64)
        args.size_lo, args.size_hi = np.percentile(args.size_ref_arr, [1, 99])
        print('目标尺寸参考分布 %s：%d 个，p05/p50/p95 = %.0f / %.0f / %.0f px（合理性区间 p01-p99 = %.0f-%.0f）' % (
            args.size_ref, len(args.size_ref_arr), *np.percentile(args.size_ref_arr, [5, 50, 95]), args.size_lo, args.size_hi))
    if args.range_ref:
        args.range_ref_arr = np.load(args.range_ref).astype(np.float64)
        print('目标距离参考分布 %s：%d 个，p05/p50/p95 = %.2f / %.2f / %.2f m' % (
            args.range_ref, len(args.range_ref_arr), *np.percentile(args.range_ref_arr, [5, 50, 95])))

    cs = os.path.join(args.cache, args.split)
    idx = pickle.load(open(os.path.join(cs, 'index.pkl'), 'rb'))
    metas, valid, W, H = idx['metas'], list(idx['valid_idx']), int(idx['W']), int(idx['H'])
    vis = np.load(os.path.join(cs, 'vis_score.npy'))
    import collections as _c
    cache = _c.OrderedDict()

    def get(i):
        # LRU：每帧原生五模态 + 点云约 10 MB，几千帧的池子不能全留在内存
        if i in cache:
            cache.move_to_end(i)
            return cache[i]
        cache[i] = Frame(args.root, metas[i], W, H)
        while len(cache) > args.max_cache_frames:
            cache.popitem(last=False)
        return cache[i]

    es = os.path.join(args.erased, args.split)
    if args.make_plates:
        # 第一步：每个有效帧 -> 抹掉全部目标的 rgb / ir / dvs（原生分辨率 inpaint 后缩到 512x288），与索引一起写成一个缓存目录
        import shutil
        os.makedirs(es, exist_ok=True)
        n = len(metas)
        e_rgb = np.lib.format.open_memmap(os.path.join(es, 'rgb.npy'), 'w+', np.uint8, (n, H, W, 3))
        for k, i in enumerate(valid):
            F = get(int(i))
            e_rgb[i] = cv2.resize(F.win(F.erased('rgb')), (W, H), interpolation=cv2.INTER_AREA)
            cache.pop(int(i), None)
            if (k + 1) % 20 == 0:
                print('  抹除 %d/%d' % (k + 1, len(valid)), flush=True)
        e_rgb.flush()
        for name in ('index.pkl', 'vis_score.npy'):
            shutil.copyfile(os.path.join(cs, name), os.path.join(es, name))
        # depth/tag 原样复制（占位，翻译脚本会复制它们；交叉贴时从点云重投影，不用这两个）
        for name in ('depth.npy', 'tag.npy'):
            shutil.copyfile(os.path.join(cs, name), os.path.join(es, name))
        print('抹除底图写到', es, '——接下来用 sim2real_bg_translate.py --src %s --no-paste --no-erase --min-vis 0 --max-targets 99 翻译它' % args.erased)
        return 0

    assert args.plates, '需要 --plates（翻译好的背景）'
    ps = os.path.join(args.plates, args.split)
    plate_rgb = np.load(os.path.join(ps, 'rgb.npy'), mmap_mode='r')
    translated = np.load(os.path.join(ps, 'translated.npy'))
    bgs = [i for i in valid if translated[i]]
    srcs = [i for i in valid if vis[i] >= args.min_vis]
    print('背景帧 %d（有翻译底图），源无人机帧 %d（RGB 可见）' % (len(bgs), len(srcs)))

    if args.selftest > 0:
        # ---- 坐标链路自检：A=B、恒等变换 ----
        c_depth = np.load(os.path.join(cs, 'depth.npy'), mmap_mode='r')
        c_tag = np.load(os.path.join(cs, 'tag.npy'), mmap_mode='r')
        args.identity, args.tone = True, 0.0
        n_ok = 0
        print('%-34s %10s %10s %10s %9s %9s %8s %8s' % ('frame', 'lidar_xyz', 'lidar_int', 'radar_raw', 'depth_eq', 'tag_eq', 'box', 'rgb_eq'))
        for i in bgs[:args.selftest]:
            F = get(int(i))
            if (F.lidar_owner == F.kept).sum() < args.min_lidar_pts:
                continue
            out = paste(F, F, {'rgb': np.ascontiguousarray(plate_rgb[i])}, rng, args)
            if out is None:
                print(metas[i]['frame'], 'paste 返回 None'); continue
            n_l = int((F.lidar_owner == F.kept).sum())
            n_r = int((F.radar_owner == F.kept).sum())
            lid_back = out['lidar_pts'][-n_l:]
            lid_orig = F.lidar_raw[F.lidar_owner == F.kept]
            d_xyz = float(np.abs(lid_back[:, :3] - lid_orig[:, :3]).max())
            d_int = float(np.abs(lid_back[:, 3] - lid_orig[:, 3]).max())
            if n_r:
                rad_back = out['radar_pts'][-n_r:]
                rad_orig = F.radar_raw[F.radar_owner == F.kept]
                # 方位角可能差 2π
                da = np.abs(np.angle(np.exp(1j * (rad_back[:, 1] - rad_orig[:, 1]))))
                d_rad = float(max(da.max(), np.abs(rad_back[:, 2] - rad_orig[:, 2]).max(), np.abs(rad_back[:, 3] - rad_orig[:, 3]).max(),
                                  np.abs(rad_back[:, 0] - rad_orig[:, 0]).max()))
            else:
                d_rad = float('nan')
            dep_eq = float((out['depth'] == np.asarray(c_depth[i])).mean())
            tag_eq = float((out['tag'] == np.asarray(c_tag[i])).mean())
            d_box = float(np.abs(out['box9d'] - F.boxes[F.kept]).max())
            # 合成图在 matte 内应与原图窗口逐像素一致（alpha=1 处）
            orig_win = cv2.resize(F.win(F.rgb), (W, H), interpolation=cv2.INTER_AREA)
            hard = out['alpha'] >= 0.999
            # 合成走了 float 预乘 alpha + 缩放，与直接缩放的原图差 ±1 灰度是正常的：按容差 2 统计
            rgb_eq = float((np.abs(out['rgb_sim'][hard].astype(int) - orig_win[hard].astype(int)).max(axis=1) <= 2).mean()) if hard.any() else float('nan')
            ok = d_xyz < 1e-6 and d_int < 1e-6 and (np.isnan(d_rad) or d_rad < 1e-6) and dep_eq > 0.995 and tag_eq > 0.995 and d_box < 1e-6
            n_ok += ok
            print('%-34s %10.2e %10.2e %10s %9.4f %9.4f %8.1e %8.3f %s' % (
                metas[i]['frame'][:34], d_xyz, d_int, ('%.2e' % d_rad) if not np.isnan(d_rad) else 'n/a', dep_eq, tag_eq, d_box, rgb_eq, 'OK' if ok else 'FAIL'))
        print('自检通过 %d 帧' % n_ok)
        return 0

    by_w = {}
    src_cls = {}
    n_big = 0
    for i in srcs:
        m = metas[i]
        q = [(b, n) for b, n, qq in zip(m['boxes9d'], m['names'], m['qualified']) if qq]
        if not q:
            continue
        kept, kname = min(q, key=lambda t: np.linalg.norm(t[0][:3]))
        if max(kept[3:6]) > args.max_dim:
            n_big += 1
            continue
        by_w.setdefault(m['seq'].split('/')[3], []).append(i)
        src_cls[i] = class_of(kname) or kname
    cls_count = {c: 0 for c in set(src_cls.values())}
    print('源帧 %d（剔除超过 %.1f m 的 %d），机型：%s' % (len(src_cls), args.max_dim, n_big,
          {c: sum(1 for v in src_cls.values() if v == c) for c in cls_count}))
    k = 0
    attempts = 0
    results = []
    strips = []
    while k < args.n and attempts < args.n * 20:
        attempts += 1
        ib = int(rng.choice(bgs))
        w = metas[ib]['seq'].split('/')[3]
        # 同天气；LiDAR 噪声等级也尽量相同（把 ±3 m 噪声的无人机点贴进零噪声的点云会露馅），没有再放宽
        pool = [i for i in by_w.get(w, []) if i != ib and metas[i]['lidar_noise'] == metas[ib]['lidar_noise']]
        if not pool:
            pool = [i for i in by_w.get(w, []) if i != ib]
        if not pool:
            continue
        if args.balance_classes:
            # 机型轮流：在本天气可用的机型里选目前样本最少的那个
            avail = sorted(sorted(set(src_cls[i] for i in pool)), key=lambda c: (cls_count[c], rng.rand()))   # 先按名字排，seed 可复现
            pool = [i for i in pool if src_cls[i] == avail[0]]
        ia = int(rng.choice(pool))
        A, B = get(ia), get(ib)
        plates = {'rgb': np.ascontiguousarray(plate_rgb[ib])}
        out = paste(A, B, plates, rng, args)
        if out is None:
            continue
        name = '%02d_%s_A%s_B%s' % (k, w, os.path.splitext(metas[ia]['frame'])[0], os.path.splitext(metas[ib]['frame'])[0])
        if args.gallery:
            strips.append(make_gallery(A, B, out, os.path.join(args.out, name + '.jpg')))
            if len(strips) % 20 == 0 or k + 1 == args.n:
                page = np.vstack(strips[-(len(strips) - 20 * ((len(strips) - 1) // 20)):])
                cv2.imwrite(os.path.join(args.out, 'contact_%02d.jpg' % ((len(strips) - 1) // 20)), page, [cv2.IMWRITE_JPEG_QUALITY, 85])
        elif not args.no_sheets:
            make_sheet(A, B, out, os.path.join(args.out, name + '.jpg'))
        np.savez_compressed(os.path.join(args.out, name + '.npz'),
                            rgb=out['rgb'], rgb_sim=out['rgb_sim'], ir=out['ir'], dvs=out['dvs'], depth=out['depth'], tag=out['tag'],
                            radar_hm=out['radar_hm'], lidar_pts=out['lidar_pts'], radar_pts=out['radar_pts'], box9d=out['box9d'],
                            name=out['name'], K_raw=out['K_raw'], raw_wh=np.array(out['raw_wh']), R_cam=out['transform']['R_cam'],
                            t=out['transform']['t'], s=out['transform']['s'], A=metas[ia]['seq'] + '/' + metas[ia]['frame'],
                            B=metas[ib]['seq'] + '/' + metas[ib]['frame'],
                            depth_src=out['depth_src'], tag_src=out['tag_src'], uv_src=out['uv_src'], rgb_src=out['rgb_src'],
                            checks=np.array(dict(out['checks'], range_old=out['transform']['range_old'],
                                                 range_new=out['transform']['range_new'], cls=src_cls.get(ia, '?')), dtype=object))
        ck = out['checks']
        print('%s  s=%.2f  range %.1f->%.1f  px %.0f  lidar in-box A %.2f -> 去噪 %.2f -> B加噪(σ %.1f->%.1f) %.2f (%d, 射线穿框 %.2f)  radar %.2f->%.2f (%d)  matte %.2f  in-hull %.2f' % (
            name, out['transform']['s'], out['transform']['range_old'], out['transform']['range_new'], ck['px_new'],
            ck['lidar_in_box_old'], ck['lidar_in_box_denoised'], ck['sigma_A'], ck['sigma_B'], ck['lidar_in_box_new'], ck['lidar_n'], ck['lidar_ray_hit_frac'],
            ck['radar_in_box_old'], ck['radar_in_box_new'], ck['radar_n'], ck['matte_frac'], ck['alpha_in_hull']), flush=True)
        ck['cls'] = src_cls.get(ia, '?')
        ck['range_new'] = out['transform']['range_new']
        ck['dim'] = float(max(A.boxes[A.kept][3:6]))
        cls_count[ck['cls']] = cls_count.get(ck['cls'], 0) + 1
        results.append(ck)
        k += 1
    if results:
        by_c = {}
        for r in results:
            by_c.setdefault(r['cls'], []).append(r)
        for c, rs in sorted(by_c.items()):
            rg = [r['range_new'] for r in rs]
            px = [r['px_new'] for r in rs]
            print('  机型 %-16s 边长 %.2f m  样本 %3d  距离 %.1f-%.1f m (中位 %.1f)  表观 %.0f-%.0f px (中位 %.0f)' % (
                c, rs[0]['dim'], len(rs), min(rg), max(rg), np.median(rg), min(px), max(px), np.median(px)))
        keys = ['lidar_in_box_old', 'lidar_in_box_denoised', 'lidar_in_box_new', 'lidar_ray_hit_frac', 'matte_frac', 'alpha_in_hull', 'tag_px_in_hull', 'px_new']
        print('汇总（%d 样本）: ' % len(results) + '  '.join('%s=%.2f' % (kk, np.mean([r[kk] for r in results])) for kk in keys))
    return 0


if __name__ == '__main__':
    sys.exit(main())
