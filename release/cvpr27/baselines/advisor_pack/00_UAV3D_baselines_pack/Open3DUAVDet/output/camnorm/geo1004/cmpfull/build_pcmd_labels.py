# -*- coding: utf-8 -*-
"""路线 3：把 PCMD 教师伪标签表拼成对手方法能直接训练的 labels.pkl。不含任何真实标签。
  MAV6D：    data/cmpfull/pcmd_teacher_pseudo_real_train.pkl    → 目标域训练图 real_train（4,088 帧）
  AntiUAV3D：data/cmpfull/pcmd_teacher_pseudo_outdoor_train.pkl → 目标域训练图 outdoor_train（298 帧，来自我们室外 PCMD 每组 1 个网络的那次运行）
位置 = 被红外剪影蒸馏过的距离修正网络修正的检测；朝向 = 红外 / LiDAR 塑形的彩色教师在两位老师一致时给的朝向；尺寸 = 常数尺寸。
用法：python build_pcmd_labels.py <data/cmpfull 目录> <输出 labels.pkl> [real_train|outdoor_train]（缺省 real_train）"""
import os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from angles import ypr_from_R
data, out = sys.argv[1], sys.argv[2]
split = sys.argv[3] if len(sys.argv) > 3 else 'real_train'
tab = pickle.load(open(os.path.join(data, 'pcmd_teacher_pseudo_%s.pkl' % split), 'rb'))
root = os.path.abspath(os.path.join(data, split)).replace('\\', '/')
lab = {(l['seq'], l['frame']): l for l in pickle.load(open(os.path.join(root, 'labels.pkl'), 'rb'))}   # 只取图像信息（boxes 为空）
res = []
for t in tab:
    d = dict(lab[(t['seq'], t['frame'])]); R = Rot.from_euler('xyz', t['euler_xyz']).as_matrix()
    d['file'] = root + '/' + d['file']; d['root'] = root
    d['boxes'] = [dict(center=np.asarray(t['center']), lwh=np.asarray(t['lwh']), euler_xyz=np.asarray(t['euler_xyz']), R=R, cls='drone', ypr=ypr_from_R(R))]
    res.append(d)
pickle.dump(res, open(out, 'wb'))
print('PCMD 教师伪标签（%s）：%d 帧 → %s' % (split, len(res), out))
