# -*- coding: utf-8 -*-
"""MAV6D 目标表观大小的经验分布：每个目标 3D 框 8 角点（含畸变）投到网络输入分辨率（IM_RESIZE）后的 2D 框长边像素。
交叉贴增广 / 变焦裁剪的目标尺寸直接从这个数组里抽，仿真样本的尺寸分布就严格等于 MAV6D 的。

    D:/Miniconda3/envs/city/python.exe tools/mav6d_size_stats.py --root E:/MAV6D --out docs/results/mav6d_size_px.npy
"""
import argparse
import os
import sys

import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R

# 与 uavdet3d/datasets/mav6d/mav6d_utils.read_truth_Rt 完全一致（内联，避免导入整个包：实测导入会卡住）
CAM2VICON = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                      [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                      [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                      [0, 0, 0, 1]])


def read_truth_Rt(labpath):
    contents = open(labpath, 'r').readlines()[0].strip().split(' ')
    pose = [float(x) for x in contents]
    uav_pose = pose[9:]
    rot = R.from_quat([uav_pose[-4], uav_pose[-3], uav_pose[-2], uav_pose[-1]]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = uav_pose[0:3]
    M = CAM2VICON @ T
    return M[:3, :3].reshape(9), M[:3, 3].reshape(3)

PROTO = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
# 与 uavdet3d/datasets/mav6d/mav6d_det_dataset.py 一致
K = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.7, 542.8], [0.0, 0.0, 1.0]])
DIST = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487])
OB_SIZE = np.array([0.34, 0.34, 0.23])
RAW_WH = (1920, 1080)
NEW_WH = (512, 256)          # mav6d.yaml IM_RESIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='E:/MAV6D')
    ap.add_argument('--out', default='docs/results/mav6d_size_px.npy')
    ap.add_argument('--classes', nargs='*', default=['mavic2', 'phantom4'])
    ap.add_argument('--splits', nargs='*', default=['train', 'test'])
    args = ap.parse_args()
    # 读真实的 fy / cy（源码里的矩阵第二行）
    try:
        import re
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uavdet3d', 'datasets', 'mav6d', 'mav6d_det_dataset.py'), encoding='utf-8').read()
        m = re.search(r'self\.intrinsic\s*=\s*np\.array\(\[\[(.*?)\]\]\)', src, re.S)
        vals = [float(v) for v in re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', m.group(1))]
        if len(vals) == 9:
            K[:] = np.array(vals).reshape(3, 3)
    except Exception as e:  # noqa: BLE001
        print('用内置 K（读源码失败：%s）' % e)
    print('K =', K.round(2).tolist())
    sx, sy = NEW_WH[0] / RAW_WH[0], NEW_WH[1] / RAW_WH[1]
    sizes, ranges, per = [], [], {}
    for cls in args.classes:
        for split in args.splits:
            sp = os.path.join(args.root, cls, 'split', split + '.txt')
            if not os.path.exists(sp):
                continue
            n = 0
            for ln in open(sp):
                p = ln.strip().split('/')[-3:]
                if len(p) < 3:
                    continue
                lab = os.path.join(args.root, cls, 'labels', p[0], p[1], os.path.splitext(p[2])[0] + '.txt')
                if not os.path.exists(lab) or not os.path.getsize(lab):
                    continue
                Rm, t = read_truth_Rt(lab)
                Rm = np.asarray(Rm, np.float64).reshape(3, 3)
                t = np.asarray(t, np.float64).reshape(3)
                cs = (PROTO * OB_SIZE) @ Rm.T + t
                uv, _ = cv2.projectPoints(cs.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)
                uv = uv.reshape(-1, 2)
                w = (uv[:, 0].max() - uv[:, 0].min()) * sx
                h = (uv[:, 1].max() - uv[:, 1].min()) * sy
                sizes.append(max(w, h))
                ranges.append(float(np.linalg.norm(t)))
                n += 1
            per[(cls, split)] = n
    sizes = np.array(sizes, np.float32)
    ranges = np.array(ranges, np.float32)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.save(args.out, sizes)
    np.save(os.path.splitext(args.out)[0] + '_range_m.npy', ranges)
    print('帧数', per, '合计', len(sizes))
    q = np.percentile(sizes, [0, 5, 25, 50, 75, 95, 100])
    print('表观大小（%dx%d 输入下 2D 框长边 px）: min %.0f  p05 %.0f  p25 %.0f  p50 %.0f  p75 %.0f  p95 %.0f  max %.0f' % ((NEW_WH[0], NEW_WH[1]) + tuple(q)))
    q = np.percentile(ranges, [0, 5, 50, 95, 100])
    print('距离 m: min %.2f  p05 %.2f  p50 %.2f  p95 %.2f  max %.2f' % tuple(q))
    print('写出', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
