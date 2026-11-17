# -*- coding: utf-8 -*-
"""[UAV 改动] 无人机统一数据（D:/cmpfull/<split>/labels.pkl）所需的标定类与三角朝向编码。

- UAVCalib：只有内参 K（P2 = [K | 0]），接口与 kitti_utils.Calibration 中 GUPNet 用到的方法一致
  （rect_to_img / img_to_rect / ry2alpha / alpha2ry / flip），公式照抄原实现。
- 朝向：用 cmpfull/angles.py 的 KITTI 式 ypr（R = Ry(yaw)·Rx(pitch)·Rz(roll) @ R0，yaw = KITTI ry）。
  原 GUPNet 只对 alpha = ry2alpha(ry, u) 做 12-bin multi-bin；这里 yaw 照原逻辑走 alpha，
  pitch / roll 直接回归绝对角，三个角都用原 angle2class / class2angle。翻转 ry->pi-ry 原样作用于 yaw（pitch 取反）。
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lib.datasets.utils import angle2class, class2angle

PROTO = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], dtype=np.float64)


def corners_of(center, lwh, R):
    return (PROTO * np.asarray(lwh, np.float64)) @ np.asarray(R, np.float64).T + np.asarray(center, np.float64)


def roty(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class UAVCalib(object):
    def __init__(self, K):
        K = np.asarray(K, np.float64)
        self.P2 = np.concatenate([K, np.zeros((3, 1))], 1).astype(np.float32)
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = 0.0
        self.ty = 0.0

    def rect_to_img(self, pts_rect):
        pts_hom = np.hstack((pts_rect, np.ones((pts_rect.shape[0], 1), dtype=np.float32)))
        pts_2d_hom = np.dot(pts_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_hom[:, 2]).T
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]
        return pts_img, pts_rect_depth

    def img_to_rect(self, u, v, depth_rect):
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1), np.array(depth_rect).reshape(-1, 1)), axis=1)
        return pts_rect

    def ray_angle(self, u):
        # 与原 Calibration.ry2alpha / alpha2ry 里的 theta 相同
        return np.arctan2(u - self.cu, self.fu)

    def ry2alpha(self, ry, u):
        return wrap(ry - self.ray_angle(u))

    def alpha2ry(self, alpha, u):
        return wrap(alpha + self.ray_angle(u))

    def flip(self, img_size):
        # 原实现用 8 个点重新解 P2，对 P2=[K|0] 的结果就是 u -> W - u（cu' = W - cu），这里直接写出来
        self.P2 = self.P2.copy()
        self.P2[0, 2] = img_size[0] - self.P2[0, 2]
        self.cu = self.P2[0, 2]


# ----------------------------- 三角朝向编码（KITTI 式 ypr，见 cmpfull/angles.py） -----------------------------
import os as _os, sys as _sys
_CMP = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..'))  # .../cmpfull
if _CMP not in _sys.path:
    _sys.path.insert(0, _CMP)
from angles import ypr_from_R, R_from_ypr, euler_xyz_from_ypr  # 统一角度约定，所有对比方法共用


def ypr_to_code_angles(ypr, u, calib):
    """原 GUPNet：heading = alpha = ry2alpha(ry, u)，u = 2D 框中心横坐标。这里 ry = yaw 原样；
    pitch、roll 直接作为绝对角，用同一套 12-bin + 残差编码。"""
    return np.array([calib.ry2alpha(ypr[0], u), ypr[1], ypr[2]])


def code_angles_to_ypr(ang, u, calib):
    """原 GUPNet：ry = alpha2ry(alpha, u)。"""
    return np.array([calib.alpha2ry(ang[0], u), ang[1], ang[2]])


def encode_angles(ang):
    """每个角用原 GUPNet 的 angle2class（12 bin + 残差）。"""
    bins, res = np.zeros(3, np.int64), np.zeros(3, np.float32)
    for k in range(3):
        bins[k], res[k] = angle2class(wrap(ang[k]))
    return bins, res


def decode_angles(bins, res):
    return np.array([class2angle(int(bins[k]), float(res[k]), to_label_format=True) for k in range(3)])


def flip_ypr(center, ypr):
    """水平翻转增广：原 KITTI 为 ry -> pi - ry、pos[0] *= -1（再 wrap 到 [-pi, pi]）。
    yaw 原样照做；镜像 x -> -x 对完整 R 要求 pitch -> -pitch、roll 不变
    （推导：S·Ry(y)Rx(p)Rz(r)·D = Ry(pi-y)·Rx(-p)·Rz(r)，S=diag(-1,1,1)，D 翻物体横向轴；tests/roundtrip.py --flip 用角点验证）。"""
    c = np.asarray(center, np.float64).copy()
    c[0] *= -1
    return c, np.array([wrap(np.pi - ypr[0]), -ypr[1], ypr[2]])
