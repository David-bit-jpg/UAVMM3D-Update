# -*- coding: utf-8 -*-
"""路线 3 室外版（11-11）：AntiUAV3D ≤80 m 目标域训练集（outdoor_train）的"PCMD 教师伪标签"，给对手方法用。不读任何真实标签。
来源 = 我们室外 PCMD 的一次运行（每组 1 个网络的版本，gproto/outdoor/run_outdoor.sh）：
  位置 + 尺寸：D:/mmcache/outdoor_real_out/train/index_pseudo_out.pkl（ST 检测 × 红外剪影蒸馏过的距离修正，conf ≥ 0.5；常数尺寸 s̄）；
  朝向：D:/S2MUAV3D/pcmd_out_work/pseudo_tr1.npz（红外 / LiDAR 塑形教师的对称集成，PnP 在裁窗内参 K_crop 上解，裁窗只平移缩放 → 相机系同原图），
        只取 gate（红外蒸馏教师 ⇄ LiDAR 教师 对称夹角 ≤ 10°）的帧，与 PCMD 室外学生训练用的帧池相同。
帧 = 有位置伪标签 ∩ gate。
四旋翼对称：教师的对称集成在 4 个等价朝向（绕机体竖轴转 90° 的倍数）里任取一个代表，室外这批与检测器差 90° 的占多数
  （直接夹角中位约 95°，对称夹角中位约 25°）。为了让对手方法的朝向监督和仿真标签的机头约定一致，这里把教师旋转换成
  4 个等价朝向里离本帧检测器旋转最近的那个（只换代表，不改朝向信息；评测按四旋翼对称算，不受影响；常数尺寸 l≈w，换代表时同时交换 l/w）。
约定核对（不读标签）：检测器旋转与（换代表后）教师旋转的夹角中位，应远小于 90°。
用法：python make_pseudo_pcmd_out.py <输出 pkl>   （格式同 pcmd_teacher_pseudo_real_train.pkl：list of dict(seq, frame, center, lwh, euler_xyz)）"""
import pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
idx = pickle.load(open('D:/mmcache/outdoor_real_out/train/index_pseudo_out.pkl', 'rb'))
pos = {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)[0] for m in idx['metas'] if m}
T = np.load('D:/S2MUAV3D/pcmd_out_work/pseudo_tr1.npz', allow_pickle=True)
L = pickle.load(open('D:/cmpfull/outdoor_train/labels.pkl', 'rb'))          # 只用 seq/frame 对齐（boxes 为空）
lab = {(l['seq'], l['frame']) for l in L}
out, dang, n_all = [], [], 0
for i in range(len(T['seq'])):
    k = (str(T['seq'][i]), str(T['frame'][i]))
    if k not in pos or k not in lab:
        continue
    n_all += 1
    if not T['gate'][i]:
        continue
    b = pos[k]; Rd = Rot.from_euler('xyz', b[6:9]).as_matrix(); R0 = np.asarray(T['R'][i], np.float64)
    q = int(np.argmin([Rot.from_matrix(Rd.T @ R0 @ RZ[j]).magnitude() for j in range(4)]))
    Rm = R0 @ RZ[q]; lwh = b[3:6].copy()
    if q % 2:
        lwh[[0, 1]] = lwh[[1, 0]]
    dang.append(np.degrees(Rot.from_matrix(Rot.from_euler('xyz', b[6:9]).as_matrix().T @ Rm).magnitude()))
    out.append(dict(seq=k[0], frame=k[1], center=b[:3].copy(), lwh=lwh, euler_xyz=Rot.from_matrix(Rm).as_euler('xyz')))
pickle.dump(out, open(sys.argv[1], 'wb'))
dang = np.array(dang)   # 换代表后的夹角
print('室外 PCMD 教师伪标签：%d 帧（位置伪标签 %d 帧，其中有朝向 %d 帧，gate 留 %d）；检测器旋转与教师旋转夹角 中位 %.1f°，<45° 占 %.0f%%（约定核对：应远小于 90°）' % (
    len(out), len(pos), n_all, len(out), np.median(dang), 100 * np.mean(dang < 45)))
