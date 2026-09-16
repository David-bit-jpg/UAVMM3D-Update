# -*- coding: utf-8 -*-
"""UavIndoorSim（UE 5.8 室内仿真）无人机资产的标注修正：机头偏航 + 物理尺度。

两个问题都不是转换代码的 bug，而是【资产本身】与标签约定不一致（2026-09-15 标注一致性核对，
docs/audit/ANNOTATION_CONSISTENCY_2026-09-15.md）：

1. 机头偏航（NOSE_YAW_DEG）
   录制器把 BoundingCheck 盒的局部 +X（= actor +X = 飞行方向）当机头写标签（build_mm_cache 翻宽度轴后得到
   机体 x 前 / y 左 / z 上）。但 Drone Pack 里 7 个网格的真实机头都朝 actor 的 +Y（UE 右手侧），也就是
   标签的 -y：
     - 顶点导出（output/camnorm/annot_audit/sim_assets/verts）：6 个网格关于 X=0 镜像对称（侧向轴 = X），
       unk3 蓝图把网格转了 -90°，转完同样侧向轴 = X；
     - 侧剖面：avata2 镜头在 +Y 端；phantom4 云台相机在 +Y、下视视觉传感器在 -Y；mavic-mini 位置更低/更宽的
       后臂电机在 -Y；m210/m300 云台挂在 +Y；unk3 机身 -Y 端高圆（电池）、+Y 端低尖（机头斜面）；
       M600 红色前臂在 +Y（原图）；
     - 原图（nose_views/*.jpg）：标签「正后方」视角看到的是起落架纵向展开的侧视图；phantom4 在标签 -y 侧
       看到云台镜头正对相机。
   后果：标签 x 是机体侧向；「仿真没有正面视角」（az∈(-60,60) 为 0）只是这个 90° 错位的表象；
   翻转增广按标签 y 做镜像（真对称面是标签 x）又让翻转样本的航向再差 180°。
   修正：R' = R · Rz(-90°)（新 x = 旧 -y，新 y = 旧 x），l' = w，w' = l。

2. 物理尺度（SCALE_SIM_OVER_REAL）
   网格没有按真机尺寸建：phantom4 顶点轴距 ~0.62 m（真机 350 mm），高 0.34 m；mavic-mini 电机对角 ~0.55 m
   （真机 213 mm）。对针孔相机，把物体整体（位置和尺寸）绕相机中心缩 1/k 后投影逐像素不变，所以
   t' = t / k、lwh' = lwh / k 是与图像严格一致的标签变换（遮挡次序只在物体与背景之间变化，飞行目标可忽略）。
   k 取「网格尺寸 / 官方尺寸」，按与规格同口径的量折中（DJI 官方参数，2026-09-15 核实）：
     DJI-phantom4    轴距 350 mm；网格轴距比 ~1.75，图像可见高/宽比（对 MAV6D 真机）1.46~1.6        -> 1.65
     DJI-mavic-mini  带桨展开 245x290x55、对角 213 mm；网格 L/W/H 比 3.0/2.2/2.35，对角 ~2.6            -> 2.5
     DJI-avata2      185x212x64 mm；网格 L/W/H 比 2.39/2.14/2.86                                         -> 2.3
     Matrice-600-Pro 带桨 1668x1518x727、轴距 1133 mm；网格外廓比 1.20/1.32/1.06                          -> 1.25
     m210-rtk        展开 887x880x408、轴距 643 mm；网格高比 1.19、轴距比 ~1.5                            -> 1.35
     matrix-300-RTK  展开 810x670x430、轴距 895 mm；网格轴距比 ~1.4、高比 ~1.1                            -> 1.3
     drone-unk3      真机型号不明；取其余机型中位                                                          -> 1.5
   不确定度：phantom4 约 ±7%，其余 ±15%。

用法：apply_fix(boxes9d, names, nose=True, scale=True) -> 新 boxes9d（float32，欧拉 'xyz'）。
注意（2026-09-15 晚）：用户要求新旧原始数据都就地修正机头（tools/fix_raw_nose_labels.py，标记 label_fix_nose_x.json），
之后从原始数据建的缓存已经是机头 +x，只能再施加 scale，不能再 nose=True。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

FIX_VERSION = 'simfix-v1-20260915'

NOSE_YAW_DEG = {        # 真实机头相对标签 x 绕机体 z 的角度（新 x = R_old · Rz(yaw) · e_x）
    'DJI-avata2': -90.0,
    'DJI-mavic-mini': -90.0,
    'DJI-phantom4': -90.0,
    'Matrice-600-Pro': -90.0,
    'm210-rtk': -90.0,
    'matrix-300-RTK': -90.0,
    'drone-unk3': -90.0,
}

SCALE_SIM_OVER_REAL = {
    'DJI-avata2': 2.3,
    'DJI-mavic-mini': 2.5,
    'DJI-phantom4': 1.65,
    'Matrice-600-Pro': 1.25,
    'm210-rtk': 1.35,
    'matrix-300-RTK': 1.3,
    'drone-unk3': 1.5,
}


def base_name(name):
    """录制器 actor 名可能带 '_up' 等后缀；按表里最长的前缀匹配。"""
    hits = [k for k in NOSE_YAW_DEG if str(name).startswith(k)]
    if not hits:
        raise KeyError('sim_asset_fix: 未知机型 %r（表里没有，不能静默跳过）' % (name,))
    return max(hits, key=len)


def fix_box(b, name, nose=True, scale=True, euler_seq='xyz'):
    b = np.asarray(b, dtype=np.float64).copy()
    key = base_name(name)
    if nose:
        yaw = NOSE_YAW_DEG[key]
        Rm = R.from_euler(euler_seq, b[6:9]).as_matrix() @ R.from_euler('z', yaw, degrees=True).as_matrix()
        b[6:9] = R.from_matrix(Rm).as_euler(euler_seq)
        if abs(abs(yaw) - 90.0) < 1e-6:
            b[3], b[4] = b[4], b[3]
        else:
            assert abs(yaw) < 1e-6 or abs(abs(yaw) - 180.0) < 1e-6, '只支持 0/±90/180 的偏航修正'
    if scale:
        k = SCALE_SIM_OVER_REAL[key]
        b[:3] /= k
        b[3:6] /= k
    return b


def apply_fix(boxes9d, names, nose=True, scale=True, euler_seq='xyz'):
    boxes9d = np.asarray(boxes9d)
    if len(boxes9d) == 0:
        return boxes9d.astype(np.float32)
    out = np.stack([fix_box(b, n, nose, scale, euler_seq) for b, n in zip(boxes9d, names)])
    return out.astype(np.float32)
