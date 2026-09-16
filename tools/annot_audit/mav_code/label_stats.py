# -*- coding: utf-8 -*-
"""MAV6D 原始标签 txt 的统计：token 数、前 8 个数（相机刚体位姿?）是否恒定、四元数模长、时间戳与文件名关系。"""
import os, glob, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
root = 'E:/MAV6D'
out_dir = 'E:/Open3DUAVDet/output/camnorm/annot_audit/mav_code'
CAM2VIC = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                    [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                    [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                    [0, 0, 0, 1]])
Rc, tc = CAM2VIC[:3, :3], CAM2VIC[:3, 3]
print('det(camera2vicon R) = %.6f, orthogonality err = %.2e' % (np.linalg.det(Rc), np.abs(Rc @ Rc.T - np.eye(3)).max()))
C_vicon = -Rc.T @ tc
print('camera optical centre in VICON frame = -R^T t =', np.round(C_vicon, 4))
print('camera axes in VICON (rows of R^T => columns of R): x_cam=%s y_cam=%s z_cam=%s' % (np.round(Rc.T[:, 0], 3), np.round(Rc.T[:, 1], 3), np.round(Rc.T[:, 2], 3)))
print('VICON +z expressed in camera frame (R @ [0,0,1]) =', np.round(Rc @ np.array([0, 0, 1.]), 4))
lines = []
for cls in ['mavic2', 'phantom4']:
    files = sorted(glob.glob(os.path.join(root, cls, 'labels', '*', '*', '*.txt')))
    ntok = {}
    body1 = []; body2 = []; ts = []; names = []
    for f in files:
        with open(f) as fh:
            c = fh.read().strip().split(' ')
        ntok[len(c)] = ntok.get(len(c), 0) + 1
        if len(c) >= 16:
            v = np.array(list(map(float, c)))
            body1.append(v[1:8]); body2.append(v[9:16]); ts.append((v[0], v[8])); names.append(f)
    body1 = np.array(body1); body2 = np.array(body2); ts = np.array(ts)
    msg = '[%s] %d files, token-count histogram %s' % (cls, len(files), ntok)
    print(msg); lines.append(msg)
    msg = '  first body (tokens 1..7): mean %s\n     std %s\n     min %s\n     max %s' % (np.round(body1.mean(0), 4), np.round(body1.std(0), 5), np.round(body1.min(0), 4), np.round(body1.max(0), 4))
    print(msg); lines.append(msg)
    msg = '  second body (tokens 9..15) = MAV: mean %s\n     min %s\n     max %s' % (np.round(body2.mean(0), 3), np.round(body2.min(0), 3), np.round(body2.max(0), 3))
    print(msg); lines.append(msg)
    q1 = np.linalg.norm(body1[:, 3:7], axis=1); q2 = np.linalg.norm(body2[:, 3:7], axis=1)
    msg = '  |q| body1 %.5f~%.5f   |q| MAV %.5f~%.5f' % (q1.min(), q1.max(), q2.min(), q2.max())
    print(msg); lines.append(msg)
    msg = '  ts0==ts8 in %.1f%% of files; ts0 range %d..%d' % (100 * np.mean(ts[:, 0] == ts[:, 1]), ts[:, 0].min(), ts[:, 0].max())
    print(msg); lines.append(msg)
    # filename vs timestamp
    stems = [os.path.splitext(os.path.basename(n))[0] for n in names]
    if stems[0].isdigit() and len(stems[0]) > 12:
        d = np.array([int(s) for s in stems]) - ts[:, 0]
        msg = '  filename_ts - label_ts (ns): median %.0f  min %.0f max %.0f  (ms: median %.1f)' % (np.median(d), d.min(), d.max(), np.median(d) / 1e6)
        print(msg); lines.append(msg)
    # first body vs camera position
    msg = '  first-body mean position %s  vs  camera centre from camera2vicon %s  |diff| = %.3f m' % (np.round(body1[:, :3].mean(0), 4), np.round(C_vicon, 4), np.linalg.norm(body1[:, :3].mean(0) - C_vicon))
    print(msg); lines.append(msg)
    # first-body orientation vs camera orientation
    R1 = R.from_quat(body1[:, 3:7].mean(0) / np.linalg.norm(body1[:, 3:7].mean(0))).as_matrix()
    # R1 maps body1 frame -> VICON. camera R^T maps camera->VICON
    Rcam_in_vicon = Rc.T
    rel = R1.T @ Rcam_in_vicon
    msg = '  R_body1^T @ R_cam(in vicon) = \n%s\n  (angle between them %.1f deg)' % (np.round(rel, 3), np.degrees(np.linalg.norm(R.from_matrix(rel).as_rotvec())))
    print(msg); lines.append(msg)
    # MAV body z (from quaternion) vs VICON up
    Rm = R.from_quat(body2[:, 3:7]).as_matrix()  # body->vicon
    zdot = Rm[:, 2, 2]  # body z axis expressed in vicon, z component
    xz = Rm[:, 2, 0]; yz = Rm[:, 2, 1]
    msg = '  MAV body z . VICON z: median %.4f  5%% %.4f  min %.4f | body x z-comp median %.4f | body y z-comp median %.4f' % (np.median(zdot), np.percentile(zdot, 5), zdot.min(), np.median(xz), np.median(yz))
    print(msg); lines.append(msg)
    # rotation about body/VICON z: yaw distribution
    yaw = np.degrees(np.arctan2(Rm[:, 1, 0], Rm[:, 0, 0]))
    msg = '  MAV yaw (VICON) percentiles 5/50/95: %.1f %.1f %.1f' % tuple(np.percentile(yaw, [5, 50, 95]))
    print(msg); lines.append(msg)
    # height above vicon origin
    msg = '  MAV z (VICON, m) percentiles 5/50/95: %.2f %.2f %.2f' % tuple(np.percentile(body2[:, 2], [5, 50, 95]))
    print(msg); lines.append(msg)
with open(os.path.join(out_dir, 'label_stats.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
