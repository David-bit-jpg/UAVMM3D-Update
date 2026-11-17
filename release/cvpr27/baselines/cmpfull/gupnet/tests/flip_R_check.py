# 翻转增广的 R 往返自检：flip_ypr 给出的 (yaw,pitch,roll) 重建的 R 应等于 S·R·D
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from lib.datasets.uav_utils import flip_ypr, ypr_from_R, R_from_ypr
S, D = np.diag([-1., 1, 1]), np.diag([1., -1, 1])
err = 0
for R in Rot.random(500, random_state=0).as_matrix():
    _, y2 = flip_ypr(np.zeros(3), ypr_from_R(R))
    err = max(err, np.abs(R_from_ypr(y2) - S @ R @ D).max())
print('flip R 往返最大误差', err)
