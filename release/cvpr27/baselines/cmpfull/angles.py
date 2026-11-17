# -*- coding: utf-8 -*-
"""统一角度约定（KITTI 式，所有对比方法共用）。
水平机体系（level）：物体 x(机头)→相机 x，物体 z(朝上)→相机 −y，物体 y→相机 z；R0 = 物体→水平机体的固定换轴。
R = R_level @ R0，R_level = Ry(yaw) · Rx(pitch) · Rz(roll)（scipy 内旋 'YXZ'，相机系的 y 轴就是 KITTI 的 ry 轴）。
  yaw ∈ (−180°, 180°] = KITTI 的 ry（ry=0 时机头朝相机 +x，长 l 沿相机 x，宽 w 沿相机 z，高 h 沿相机 −y）；
  pitch、roll 在仿真里都在 ±30° 以内（中间角离 ±90° 很远，不会跳分支）。
KITTI 尺寸顺序 (h, w, l) = (lwh[2], lwh[1], lwh[0])。"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
R0 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])


def ypr_from_R(R):
    return Rot.from_matrix(np.asarray(R) @ R0.T).as_euler('YXZ')            # [yaw, pitch, roll]，弧度


def R_from_ypr(ypr):
    return Rot.from_euler('YXZ', np.asarray(ypr)).as_matrix() @ R0


def euler_xyz_from_ypr(ypr):
    return Rot.from_matrix(R_from_ypr(ypr)).as_euler('xyz')                  # 回到 labels.pkl / boxes9d 的 euler_xyz


if __name__ == '__main__':      # 往返自检
    r = Rot.random(1000, random_state=0).as_matrix()
    e = max(np.abs(R_from_ypr(ypr_from_R(x)) - x).max() for x in r)
    print('往返最大误差', e)
