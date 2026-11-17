# -*- coding: utf-8 -*-
# 11-13 copy: --orient selects the orientation pseudo-label npz (clean chain: D:/S2MUAV3D/srcx_work/in2mav/pseudo_tr1.npz); default = original t6il. Logic unchanged.
"""路线 3（11-10）：给对手方法用的"PCMD 教师伪标签"（把我们的蒸馏知识接到别人的检测器上）。不读任何真实标签。
  位置：PCMD 位置伪标签 index_pseudo_<tag>.pkl（ST2 检测 × 被红外剪影蒸馏过的距离修正网络，系数见 PREREG）；
  尺寸：常数尺寸 s̄（同 PCMD）；
  朝向：红外 / LiDAR 塑形的 6 个彩色教师在裁窗上的一致朝向（orient_uda/pseudo/pseudo_train_t6il.npz 的 R）。
帧：朝向伪标签覆盖的每 3 帧取 1 帧（4289 帧）∩ 有位置伪标签的帧。
用法：python make_pseudo_pcmd.py <位置伪标签 tag，如 nl075> <输出 labels.pkl> [--gate-only]"""
import argparse, os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from angles import ypr_from_R

ap = argparse.ArgumentParser()
ap.add_argument('tag'); ap.add_argument('out')
ap.add_argument('--orient', default='E:/Open3DUAVDet/output/camnorm/geo1004/gproto/wf21/orient_uda/pseudo/pseudo_train_t6il.npz')
ap.add_argument('--gate-only', action='store_true', help='只用两位老师一致（≤10°）的帧')
a = ap.parse_args()
idx = pickle.load(open('E:/mmcache/mav6d_cn_obj/train/index_pseudo_%s.pkl' % a.tag, 'rb'))
pos = {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)[0] for m in idx['metas'] if m}
T = np.load(a.orient, allow_pickle=True)
L = pickle.load(open('D:/cmpfull/real_train/labels.pkl', 'rb'))
lab = {(l['seq'], l['frame']): l for l in L}
out, dang = [], []
for i in range(len(T['seq'])):
    k = (str(T['seq'][i]), str(T['frame'][i]))
    if k not in pos or k not in lab or (a.gate_only and not T['gate'][i]):
        continue
    b = pos[k]; Rm = np.asarray(T['R'][i], np.float64)
    R_det = Rot.from_euler('xyz', b[6:9]).as_matrix()
    dang.append(np.degrees(Rot.from_matrix(R_det.T @ Rm).magnitude()))
    d = dict(lab[k]); root = 'D:/cmpfull/real_train'
    d['file'] = os.path.join(root, d['file']).replace('\\', '/'); d['root'] = root
    d['boxes'] = [dict(center=b[:3].copy(), lwh=b[3:6].copy(), euler_xyz=Rot.from_matrix(Rm).as_euler('xyz'), R=Rm, cls='drone', ypr=ypr_from_R(Rm))]
    out.append(d)
pickle.dump(out, open(a.out, 'wb'))
dang = np.array(dang)
print('PCMD 教师伪标签：%d 帧（位置 %s，朝向 %s%s）；检测器旋转与教师旋转夹角中位 %.1f°（约定核对：应远小于 90°）' % (
    len(out), a.tag, os.path.basename(a.orient), '，只取一致帧' if a.gate_only else '', np.median(dang)))
