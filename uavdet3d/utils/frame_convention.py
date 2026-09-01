# -*- coding: utf-8 -*-
"""跨域坐标系 / 旋转约定的唯一真源。

这个模块把源域 (UAV-MM3D / LAAM6D) 和目标域 (MAV6D) 之间所有"约定不一致"的地方
集中到一处，避免散落在 dataset / pre_processor / encoder / eval 各文件里各写各的。

===============================================================================
两个域的约定对照
===============================================================================

                     源域 LAAM6D                    目标域 MAV6D
  GT 原始存储        世界系 (8 角点经                相机系 (read_truth_Rt 已用
                     convert_box_opencv_to_world)    camera->VICON 外参转好)
  送进编码器前       inv(extrinsic) 转回相机系       不转换
  检测头学习目标     相机系深度 Z / MAX_DIS          相机系深度 Z / MAX_DIS
  解码器输出         转回世界系                      留在相机系
  欧拉角顺序         'zyx'  a1=绕z a2=绕y a3=绕x     'xyz'  a1=绕x a2=绕y a3=绕z
  内参               已按 IM_RESIZE 缩放，投影后     原始分辨率，投影后
                     只乘 1/stride                   乘 new/raw/stride
  畸变               全零                            k1=-0.23，不可忽略

===============================================================================
为什么欧拉顺序必须统一
===============================================================================

rot 头是 6 通道 [cos a1, sin a1, cos a2, sin a2, cos a3, sin a3]。
两边 a1 的物理含义不同（一个绕 z、一个绕 x），所以：

  * 直接把源域训好的 rot 头权重迁到 MAV6D，等于把第一个和第三个旋转轴接反；
  * 而且这不是简单的通道交换 —— 欧拉角换顺序是三个角的非线性函数，
    没法靠重排权重修复，只能在【训练源域时就用同一套顺序】。

所以源域侧提供 EULER_SEQ 配置项（默认 'zyx'，保持原行为不变）。
做迁移实验时把它设成 'xyz'，rot 头才真正可迁。

实测：同一组角按两种顺序解释，旋转差异中位 134°、90 分位 175°。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# 各域的原生约定
LAAM6D_EULER_SEQ = 'zyx'
MAV6D_EULER_SEQ = 'xyz'

VALID_SEQS = ('xyz', 'zyx', 'ZYX', 'XYZ', 'zyz', 'xyx')

# 进程级默认值。只影响主进程里的度量函数；
# dataloader 的 worker 用的是 pre_processor 实例上的 self.euler_seq
# （实例会被 pickle 传给 worker，模块全局不会）。
_DEFAULT_EULER_SEQ = LAAM6D_EULER_SEQ


def set_default_euler_seq(seq):
    """设置进程级默认欧拉顺序。由 dataset.__init__ 从配置读出后调用。"""
    global _DEFAULT_EULER_SEQ
    seq = str(seq)
    if seq not in VALID_SEQS:
        raise ValueError('不支持的欧拉顺序: %r，可选 %s' % (seq, list(VALID_SEQS)))
    _DEFAULT_EULER_SEQ = seq
    return _DEFAULT_EULER_SEQ


def get_default_euler_seq():
    return _DEFAULT_EULER_SEQ


def _seq(seq):
    return _DEFAULT_EULER_SEQ if seq is None else seq


# --------------------------------------------------------------------------- #
# 旋转
# --------------------------------------------------------------------------- #
def rotmat_to_euler(mat, seq=None):
    """旋转矩阵 -> 欧拉角三元组 (a1, a2, a3)，弧度。"""
    return R.from_matrix(np.asarray(mat, dtype=np.float64)).as_euler(_seq(seq))


def euler_to_rotmat(angles, seq=None):
    """欧拉角三元组 -> 旋转矩阵。"""
    return R.from_euler(_seq(seq), np.asarray(angles, dtype=np.float64)).as_matrix()


def euler_to_quat(angles, seq=None):
    """欧拉角 -> 四元数 (x, y, z, w)，scipy 约定。"""
    return R.from_euler(_seq(seq), np.asarray(angles, dtype=np.float64)).as_quat()


def convert_euler_seq(angles, from_seq, to_seq):
    """把同一个旋转在两种欧拉顺序之间做【精确】转换（经旋转矩阵中转）。

    注意这是角度值的转换，不是网络权重的转换 —— rot 头的权重没法这样迁。
    """
    angles = np.asarray(angles, dtype=np.float64)
    single = (angles.ndim == 1)
    a = angles.reshape(-1, 3)
    out = R.from_euler(from_seq, a).as_euler(to_seq)
    return out.reshape(3) if single else out


def angular_error_deg(pred_angles, gt_angles, seq=None, fold_180=False):
    """两组欧拉角之间的测地线角度差（度）。

    fold_180: 把 >90° 的误差折成 180-err，等价于假设目标 180° 对称。
              会显著拉低数字，默认关闭。
    """
    seq = _seq(seq)
    rp = R.from_euler(seq, np.asarray(pred_angles, dtype=np.float64).reshape(-1, 3))
    rg = R.from_euler(seq, np.asarray(gt_angles, dtype=np.float64).reshape(-1, 3))
    err = np.degrees((rp.inv() * rg).magnitude())
    if fold_180:
        err[err > 90] = 180.0 - err[err > 90]
    return err


# --------------------------------------------------------------------------- #
# 平移 / 坐标系
# --------------------------------------------------------------------------- #
def world_to_camera(points, extrinsic):
    """世界系点 -> 相机系点。extrinsic 是相机在世界系下的位姿 (4x4)。

    源域的 GT 存的是世界坐标，pre_processor_laam6d 就是用 inv(extrinsic)
    把它转回相机系后再送进编码器的。MAV6D 的 GT 本来就在相机系，无需此步。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.linalg.inv(np.asarray(extrinsic, dtype=np.float64).reshape(4, 4))
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (T @ hom.T).T[:, :3]


def camera_to_world(points, extrinsic):
    """world_to_camera 的逆运算。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (T @ hom.T).T[:, :3]


# --------------------------------------------------------------------------- #
def self_check(n=2000, seed=0):
    """自检：顺序转换是否无损、世界<->相机是否互逆、两种顺序差异有多大。"""
    rng = np.random.RandomState(seed)
    ang = rng.uniform(-np.pi, np.pi, (n, 3))

    # 1) zyx -> xyz -> zyx 应当无损（比较旋转本身，不比角度值：欧拉角不唯一）
    back = convert_euler_seq(convert_euler_seq(ang, 'zyx', 'xyz'), 'xyz', 'zyx')
    err_roundtrip = angular_error_deg(ang, back, seq='zyx').max()

    # 2) 直接把 zyx 的角按 xyz 解释，误差有多大
    err_misread = angular_error_deg(ang, ang, seq='zyx')  # 同顺序 -> 0
    rp = R.from_euler('zyx', ang)
    rq = R.from_euler('xyz', ang)
    err_cross = np.degrees((rp.inv() * rq).magnitude())

    # 3) 世界 <-> 相机互逆
    ext = np.eye(4)
    ext[:3, :3] = R.random(random_state=seed).as_matrix()
    ext[:3, 3] = rng.uniform(-50, 50, 3)
    pts = rng.uniform(-30, 30, (n, 3))
    err_frame = np.abs(camera_to_world(world_to_camera(pts, ext), ext) - pts).max()

    print('欧拉顺序往返 (zyx->xyz->zyx) 最大误差 : %.3e 度' % err_roundtrip)
    print('同顺序自比 最大误差                  : %.3e 度' % err_misread.max())
    print('按错顺序解释的旋转差异               : 中位 %.1f°  90%% %.1f°  最大 %.1f°'
          % (np.median(err_cross), np.percentile(err_cross, 90), err_cross.max()))
    print('世界<->相机 往返 最大误差            : %.3e m' % err_frame)
    ok = err_roundtrip < 1e-6 and err_frame < 1e-9
    print('自检结果: %s' % ('通过' if ok else '未通过'))
    return ok


if __name__ == '__main__':
    self_check()
