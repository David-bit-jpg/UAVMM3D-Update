# -*- coding: utf-8 -*-
"""拿具体的点手算一遍：原始标注 -> 相机系 9 参数 -> 投影 -> 编码 -> 解码 -> 评测(ADS 的 CARLA 外参技巧)，逐步打印数字。

    cd E:/Open3DUAVDet/tools && python handwalk_points.py

每一步都用两条路算：管线函数 vs 我自己按定义写的公式，并把投影结果画到原图裁块上留证据
（output/camnorm/walkthrough/handwalk_*.jpg）。
"""
import os
import pickle
import sys

import cv2
import numpy as np
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets.laam6d.ads_metric_laam6d import LAA3D_ADS_Metric   # noqa: E402
from uavdet3d.datasets.laam6d.dataset_utils import convert_9params_to_9points   # noqa: E402
from uavdet3d.datasets.mav6d.mav6d_utils import read_truth_Rt   # noqa: E402
from uavdet3d.datasets.mmcache.mmcache_det_dataset import MMCache_Det_Dataset   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import encoder_geometry_kwargs   # noqa: E402
from uavdet3d.utils import camera_geometry as cg, frame_convention   # noqa: E402
from uavdet3d.utils.object_encoder_mav6d import center_point_decoder, center_point_encoder   # noqa: E402
import build_mm_cache as BMC   # noqa: E402
import build_mav6d_cache as BMV   # noqa: E402
import torch   # noqa: E402

OUT = os.path.join(ROOT, 'output', 'camnorm', 'walkthrough')
os.makedirs(OUT, exist_ok=True)
np.set_printoptions(precision=4, suppress=True, linewidth=150)
PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
C2O = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def P(title, *a):
    print('  ' + title, *a, flush=True)


def draw(img, K, D, corners, center, axes_R=None, color=(0, 255, 255)):
    pts = corners if axes_R is None else np.vstack([corners, center, center + axes_R[:, 0] * 0.3, center + axes_R[:, 1] * 0.3, center + axes_R[:, 2] * 0.3])
    uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    for a, b in EDGES:
        cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), color, 2, cv2.LINE_AA)
    if axes_R is not None:
        o = tuple(np.int32(uv[8]))
        for k, col in ((9, (0, 0, 255)), (10, (0, 255, 0)), (11, (255, 0, 0))):
            cv2.arrowedLine(img, o, tuple(np.int32(uv[k])), col, 2, cv2.LINE_AA, tipLength=0.25)
    return uv


def crop_around(img, uv, pad=1.2, size=420):
    c = uv[:8].mean(0)
    h = int(max(60, pad * max(np.ptp(uv[:8, 0]), np.ptp(uv[:8, 1]))))
    cu, cv_ = int(c[0]), int(c[1])
    padded = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT)
    return cv2.resize(padded[cv_:cv_ + 2 * h, cu:cu + 2 * h], (size, size), interpolation=cv2.INTER_CUBIC)


def encode_decode(b, K, dc, tag):
    """管线编码器 -> 打印目标值 -> 我按定义手算 -> 解码器还原。"""
    kw = encoder_geometry_kwargs(dc, center_point_encoder, 'enc')
    hm, res, dis, dim, rot = center_point_encoder(np.concatenate([b, [0]])[None], np.array([K]), np.array([np.eye(4)]), np.zeros((1, 5)),
                                                  512, 288, 512, 288, 8, 1, ['drone'], 2, **kw)
    cy, cx = np.unravel_index(int(np.argmax(hm[0, 0])), hm[0, 0].shape)
    f_in = cg.focal(K)
    u = K @ b[:3]
    u = u[:2] / u[2]
    P('%s 编码器：输入分辨率焦距 f_in=%.2f，中心像素 (%.2f, %.2f) -> 热力图格 (%d, %d)，编码器峰值格 (%d, %d)' % (tag, f_in, u[0], u[1], u[0] // 8, u[1] // 8, cx, cy))
    P('   亚像素偏移 编码器 (%.4f, %.4f) | 手算 (%.4f, %.4f)' % (res[0, 0, cy, cx], res[0, 1, cy, cx], u[0] / 8 - cx, u[1] / 8 - cy))
    zv = b[2] * dc.DEPTH_F_REF / f_in
    P('   虚拟深度 Zv = Z*%d/f_in = %.4f*%d/%.2f = %.4f；编码器存 Zv（未除 MAX_DIS）= %.4f；训练目标 = Zv/MAX_DIS(%d) = %.4f'
      % (dc.DEPTH_F_REF, b[2], dc.DEPTH_F_REF, f_in, zv, dis[0, 0, cy, cx], dc.MAX_DIS, zv / dc.MAX_DIS))
    P('   尺寸 编码器 %s（训练目标除 MAX_SIZE=%d）' % (dim[0, :, cy, cx], dc.MAX_SIZE))
    d = b[:3] / np.linalg.norm(b[:3])
    axis = np.array([-d[1], d[0], 0.0])
    s = np.linalg.norm(axis)
    R_ray = R.from_rotvec(axis / s * np.arctan2(s, d[2])).as_matrix()
    R_cam = R.from_euler('xyz', b[6:9]).as_matrix()
    R_allo = R_ray.T @ R_cam
    a = R.from_matrix(R_allo).as_euler('xyz')
    six = np.array([np.cos(a[0]), np.sin(a[0]), np.cos(a[1]), np.sin(a[1]), np.cos(a[2]), np.sin(a[2])])
    P('   视线方向 d=%s，R_ray 把 +z 转到 d：R_ray@[0,0,1]=%s' % (d, R_ray @ np.array([0, 0, 1.0])))
    P('   相机系欧拉(xyz) %s -> 视线相对旋转欧拉 %s -> euler6 手算 %s' % (np.degrees(b[6:9]), np.degrees(a), six))
    P('   euler6 编码器 %s   最大差 %.1e' % (rot[0, :, cy, cx], np.abs(rot[0, :, cy, cx] - six).max()))
    kwd = encoder_geometry_kwargs(dc, center_point_decoder, 'dec')
    dec, conf = center_point_decoder(torch.from_numpy(hm).float(), torch.from_numpy(res).float(), torch.from_numpy(dis).double(),
                                     torch.from_numpy(dim).double(), torch.from_numpy(rot).float(), np.array([K]), np.array([np.eye(4)]),
                                     np.zeros((1, 5)), 512, 288, 512, 288, 8, 1, 1, **kwd)
    e = dec[0]
    P('   解码器还原 [x,y,z]=%s l,w,h=%s 欧拉=%s' % (e[:3], e[3:6], np.degrees(e[6:9])))
    P('   与原框差：位置 %.2e m，尺寸 %.2e m，旋转 %.2e°' % (np.linalg.norm(e[:3] - b[:3]), np.abs(e[3:6] - b[3:6]).max(),
      np.degrees((R.from_euler('xyz', e[6:9]).inv() * R.from_euler('xyz', b[6:9])).magnitude())))
    return e


def ads_path(b, K_in, raw_wh, tag):
    """评测里 ADS 用的 9 点转换 + box9d_to_2d(extrinsic=CARLA_TO_OPENCV) 技巧 vs 直接投影。"""
    frame_convention.set_default_euler_seq('xyz')
    pts9 = convert_9params_to_9points(b[None].astype(np.float32))[0]
    my8 = (PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
    P('%s ADS 9点：中心差 %.1e，8 角点 vs 手算 最大差 %.1e' % (tag, np.abs(pts9[0] - b[:3]).max(), np.abs(pts9[1:] - my8).max()))
    K_raw = cg.scale_K(K_in, raw_wh[0] / 512.0, raw_wh[1] / 288.0)
    m = LAA3D_ADS_Metric(eval_config=EasyDict(dict(DisMax=150, DisMin=0, MinPixel=16, AP2D=dict(IoUThresh=[0.5], RecallNum=41),
                                                 AP3D=dict(DisThresh=[1], RecallNum=41), Dof6=dict(DisNormMax=8, OriNormMax=30, SizeNormMax=1))),
                         classes=['drone'], metric_save_path=OUT)
    b2d, size = m.box9d_to_2d(pts9[None], intrinsic_mat=K_raw, extrinsic_mat=C2O, distortion_matrix=np.zeros(5))
    uv = (K_raw @ my8.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    ref = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    P('   box9d_to_2d（extrinsic=CARLA->OpenCV 抵消它内部的 inv(extrinsic)+CARLA->OpenCV）2D 框 %s' % np.round(b2d[0], 1))
    P('   直接用 K_raw 投影 8 角点的 2D 框 %s   最大差 %.2f px（<=1 是取整）' % (np.round(ref, 1), np.abs(b2d[0] - ref).max()))


def main():
    # ===================== 仿真 =====================
    print('=' * 30 + ' 仿真：一个原始帧 ' + '=' * 30)
    idx = pickle.load(open('E:/mmcache/indoor8cn/test/index.pkl', 'rb'))
    rgb = np.load('E:/mmcache/indoor8cn/test/rgb.npy', mmap_mode='r')
    cand = [int(i) for i in idx['valid_idx'] if len(idx['metas'][int(i)]['boxes9d']) == 1 and idx['metas'][int(i)]['boxes9d'][0][2] < 3.5]
    i = cand[len(cand) // 3]
    m = idx['metas'][i]
    base = os.path.join('D:/data_collect', m['seq'])
    raw = pickle.load(open(os.path.join(base, 'boxes_rgb', os.path.splitext(m['frame'])[0] + '.pkl'), 'rb'))
    info = pickle.load(open(os.path.join(base, 'im_info.pkl'), 'rb'))
    row = [r for r in raw if BMC.class_of(r[0]) is not None][0]
    c = np.array(row[1:], dtype=np.float64).reshape(8, 3)
    P('帧 %s/%s 机型 %s' % (m['seq'], m['frame'], row[0]))
    P('原始 8 角点（OpenCV 相机系，米）:\n' + str(c))
    ex, ey, ez = c[1] - c[0], c[3] - c[0], c[4] - c[0]
    P('三条边 |c1-c0|=%.4f |c3-c0|=%.4f |c4-c0|=%.4f；两两夹角余弦 %.4f %.4f %.4f' % (np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez),
      ex @ ey / np.linalg.norm(ex) / np.linalg.norm(ey), ex @ ez / np.linalg.norm(ex) / np.linalg.norm(ez), ey @ ez / np.linalg.norm(ey) / np.linalg.norm(ez)))
    P('三重积 (ex x ey)·ez = %.5f  -> %s' % (np.cross(ex, ey) @ ez, '左手系（UE），需要翻一个轴' if np.cross(ex, ey) @ ez < 0 else '右手系'))
    p9 = BMC.corners_to_9params(c)
    Rm = R.from_euler('xyz', p9[6:9]).as_matrix()
    P('corners_to_9params -> 中心 %s 尺寸 %s 欧拉(°) %s，det(R)=%.4f' % (p9[:3], p9[3:6], np.degrees(p9[6:9]), np.linalg.det(Rm)))
    P('R 的三列（机体 x,y,z 在相机系）:\n' + str(Rm))
    P('机体 x 与 c1-c0 方向夹角余弦 %.4f；机体 y 与 c3-c0 %.4f（翻了宽度轴则 -1）；机体 z 与 c4-c0 %.4f' % (
        Rm[:, 0] @ ex / np.linalg.norm(ex), Rm[:, 1] @ ey / np.linalg.norm(ey), Rm[:, 2] @ ez / np.linalg.norm(ez)))
    E = np.array(info['rgb']['extrinsic'], dtype=np.float64)
    up_cam = (C2O @ np.linalg.inv(E) @ np.array([0, 0, 1.0, 0]))[:3]
    P('世界竖直向上在相机系 = %s（相机 pitch %.1f°），机体 z·up = %.3f' % (up_cam, np.degrees(np.arcsin(up_cam[1] * -1)), Rm[:, 2] @ up_cam))
    P('缓存里存的 boxes9d = %s   与手算最大差 %.1e' % (m['boxes9d'][0], np.abs(m['boxes9d'][0] - p9).max()))
    K_leg = np.array(info['rgb']['intrinsic'], dtype=np.float64)
    K_true, fixed = cg.legacy_sim_intrinsic_fix(K_leg, 1280, 720)
    P('im_info 里的假内参 fx=%.1f fy=%.1f cx=%.1f cy=%.1f -> 识别为扰动(%s) -> 真内参 fx=fy=%.1f cx=%.1f cy=%.1f' % (
        K_leg[0, 0], K_leg[1, 1], K_leg[0, 2], K_leg[1, 2], fixed, K_true[0, 0], K_true[0, 2], K_true[1, 2]))
    img = cv2.imread(os.path.join(base, 'images_rgb', m['frame']))
    vis = img.copy()
    uv_true = draw(vis, K_true, np.zeros(5), corners_of(p9), p9[:3], Rm)
    uv_leg = draw(vis, K_leg, np.zeros(5), c, p9[:3], None, color=(255, 0, 255))
    P('真内参投影的角点 vs 假内参投影：平均偏差 (%.2f, %.2f) px' % tuple((uv_leg[:8] - uv_true[:8]).mean(0)))
    cv2.imwrite(os.path.join(OUT, 'handwalk_sim_raw.jpg'), crop_around(vis, uv_true))
    K_in = np.asarray(m['K_in'])
    P('K_in（512x288）= %s；手算 scale_K(K_true) 差 %.1e' % (K_in.reshape(-1), np.abs(cg.scale_K(K_true, 0.4, 0.4) - K_in).max()))
    small = np.array(rgb[i])
    uv_small = draw(small, K_in, np.zeros(5), corners_of(p9), p9[:3], Rm)
    cv2.imwrite(os.path.join(OUT, 'handwalk_sim_cache.jpg'), crop_around(small, uv_small, size=360))
    P('原图角点像素 * 0.4 与缓存图角点像素 之差（像素中心约定: (u+0.5)*0.4-0.5）最大 %.4f px' % np.abs((uv_true[:8] + 0.5) * 0.4 - 0.5 - uv_small[:8]).max())
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz.yaml', cfg)
    dc = cfg.DATA_CONFIG
    encode_decode(p9.astype(np.float64), K_in, dc, '仿真')
    ads_path(p9.astype(np.float64), K_in, (1280, 720), '仿真')
    # 数据集取样（无增广）与手算是否一致
    dc2 = EasyDict(dc)
    dc2.DATA_PATH = 'E:/mmcache/indoor8cn'
    dc2.VIEW_AUG = {'zoom': [1.0, 1.0]}
    dc2.DATA_SPLIT = {'train': 'test', 'test': 'test'}
    ds = MMCache_Det_Dataset(dc2, training=False)
    it = int(np.where(ds.valid_idx == i)[0][0])
    d = ds[it]
    P('数据集取样：gt_box9d=%s intrinsic=%s；与手算差 %.1e / %.1e' % (d['gt_box9d'][0], d['intrinsic'][0].reshape(-1),
      np.abs(d['gt_box9d'][0] - p9).max(), np.abs(d['intrinsic'][0] - K_in).max()))

    # ===================== MAV6D =====================
    print('=' * 30 + ' MAV6D：一个原始帧 ' + '=' * 30)
    midx = pickle.load(open('E:/mmcache/mav6d_cn/test/index.pkl', 'rb'))
    mrgb = np.load('E:/mmcache/mav6d_cn/test/rgb.npy', mmap_mode='r')
    j = int(midx['valid_idx'][len(midx['valid_idx']) // 5])
    mm = midx['metas'][j]
    cls, scene, sq = mm['seq'].split('/')
    lp = os.path.join('E:/MAV6D', cls, 'labels', scene, sq, os.path.splitext(mm['frame'])[0] + '.txt')
    nums = list(map(float, open(lp).readline().split()))
    P('帧 %s/%s：标签 21 个数，pose[9:] = %s' % (mm['seq'], mm['frame'], np.array(nums[9:])))
    t_v, q = np.array(nums[9:12]), np.array(nums[12:16])
    R_vu = R.from_quat(q).as_matrix()          # scipy: [x,y,z,w]
    P('四元数 [x,y,z,w]=%s -> R(VICON<-MAV) det=%.4f；MAV 原点在 VICON 系 %s' % (q, np.linalg.det(R_vu), t_v))
    T_cv = BMV.CAMERA2VICON
    T_vu = np.eye(4)
    T_vu[:3, :3], T_vu[:3, 3] = R_vu, t_v
    T_cu = T_cv @ T_vu
    R_c, t_c = T_cu[:3, :3], T_cu[:3, 3]
    R9, t9 = read_truth_Rt(lp)
    P('camera2vicon @ T -> 相机系 R:\n' + str(R_c) + '\n   t=%s；read_truth_Rt 差 %.1e' % (t_c, max(np.abs(R9.reshape(3, 3) - R_c).max(), np.abs(t9 - t_c).max())))
    up_c = T_cv[:3, :3] @ np.array([0, 0, 1.0])
    P('VICON z（竖直向上）在相机系 = %s（相机 y 轴向下，所以应为负 y 分量占主）；机体 z·up = %.3f' % (up_c, R_c[:, 2] @ up_c))
    b = mm['boxes9d'][0].astype(np.float64)
    P('缓存 boxes9d = %s（尺寸固定 0.34,0.34,0.23；中心 = VICON 原点 t）；R 差 %.1e' % (b, np.abs(R.from_euler('xyz', b[6:9]).as_matrix() - R_c).max()))
    K, D = BMV.K_CALIB, BMV.D_CALIB
    img = cv2.imread(os.path.join('E:/MAV6D', cls, 'JPEGImages', scene, sq, mm['frame']))
    vis = img.copy()
    uv_raw = draw(vis, K, D, corners_of(b), b[:3], R_c)
    off = np.array([-0.01, 0.01, -0.055])
    c_off = R_c @ off + t_c
    uv_off, _ = cv2.projectPoints(np.vstack([t_c, c_off]).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    uv_off = uv_off.reshape(-1, 2)
    cv2.circle(vis, tuple(np.int32(uv_off[1])), 6, (255, 0, 255), 2)
    P('原图(有畸变)投影：VICON 原点 %s；官方框几何中心(机体系偏 %s) %s；差 %.1f px' % (np.round(uv_off[0], 1), off, np.round(uv_off[1], 1), np.linalg.norm(uv_off[1] - uv_off[0])))
    cv2.imwrite(os.path.join(OUT, 'handwalk_mav_raw.jpg'), crop_around(vis, uv_raw))
    m1, m2, K_pin = cg.undistort_maps(K, D, (1920, 1080))
    K_in = np.asarray(mm['K_in'])
    P('去畸变针孔内参 K_pin fx=%.2f（=标定 1979.4 x %.4f 保住边界）；K_in=%s；手算差 %.1e' % (K_pin[0, 0], K_pin[0, 0] / K[0, 0], K_in.reshape(-1),
      np.abs(cg.scale_K(K_pin, 512 / 1920, 288 / 1080) - K_in).max()))
    # 去畸变一致性：原图畸变投影点 -> undistortPoints(K_pin) 应 == 针孔 K_pin 投影
    und_pts = cv2.undistortPoints(uv_raw[:8].reshape(-1, 1, 2), K, D, P=K_pin).reshape(-1, 2)
    pin = (K_pin @ corners_of(b).T).T
    pin = pin[:, :2] / pin[:, 2:3]
    P('畸变投影点经 undistortPoints -> 针孔投影点 最大差 %.3f px（去畸变几何自洽）' % np.abs(und_pts - pin).max())
    small = np.array(mrgb[j])
    uv_small = draw(small, K_in, np.zeros(5), corners_of(b), b[:3], R_c)
    cv2.imwrite(os.path.join(OUT, 'handwalk_mav_cache.jpg'), crop_around(small, uv_small, size=360))
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', cfg)
    encode_decode(b, K_in, cfg.DATA_CONFIG, 'MAV6D')
    ads_path(b, K_in, (1920, 1080), 'MAV6D')
    P('证据图：%s' % ', '.join(os.path.join(OUT, f) for f in ('handwalk_sim_raw.jpg', 'handwalk_sim_cache.jpg', 'handwalk_mav_raw.jpg', 'handwalk_mav_cache.jpg')))


def corners_of(b):
    return (PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]


if __name__ == '__main__':
    main()
