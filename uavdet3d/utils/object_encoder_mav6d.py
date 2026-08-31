# -*- coding: utf-8 -*-
"""
MAV6D 专用的 center-point 编解码器。

为什么不直接用 object_encoder.py 里那一对：

1. object_encoder.center_point_encoder 里写的是 `gt_box9d_with_cls[:, 0, :]` 和
   `obj[-1, 0]`，按三维数组取下标；而各 dataset 给出的 gt_box9d 都是 (N, 9)，
   拼上类别后是 (N, 10) 的二维数组 —— 直接调用必然
   `IndexError: too many indices for array`。
2. 它配套的 backProject_with_opencv_to_world 里带了一个 CARLA <-> OpenCV 的坐标轴
   置换矩阵。MAV6D 的 read_truth_Rt 返回的已经是标准 OpenCV 相机系下的平移，
   再置换一次轴就错了 —— 也就是说那一对 encoder/decoder 对 MAV6D 并不互逆。
3. MAV6D 镜头畸变不小（k1 = -0.23），原实现在投影和反投影两侧都把畸变当成 0。

这里的 encoder / decoder 是严格互逆的一对，坐标系自始至终是 OpenCV 相机系。
畸变是否参与由 use_distortion 控制（置 False 即退化成纯针孔模型，与源域
LAAM6D 的行为一致，做迁移对比时可能用得上）。

欧拉角约定 'xyz'，与 uavdet3d/datasets/mav6d/*.py 里的 as_euler('xyz') 一致。
"""
import copy

import cv2
import numpy as np
import torch

from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap


def project_points(xyz, intrinsic_mat, distortion_matrix, use_distortion=True):
    """相机系 3D 点 -> 原始分辨率下的像素坐标。返回 (uv (N,2), depth (N,))。"""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    intrinsic_mat = np.asarray(intrinsic_mat, dtype=np.float64).reshape(3, 3)
    depth = xyz[:, 2].copy()

    if use_distortion:
        dist = np.asarray(distortion_matrix, dtype=np.float64).reshape(1, -1)
        uv, _ = cv2.projectPoints(
            xyz.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            intrinsic_mat,
            dist,
        )
        uv = uv.reshape(-1, 2)
    else:
        proj = (intrinsic_mat @ xyz.T).T
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = proj[:, :2] / proj[:, 2:3]

    return uv, depth


def unproject_points(uv, depth, intrinsic_mat, distortion_matrix, use_distortion=True):
    """project_points 的严格逆运算：像素坐标 + 深度 -> 相机系 3D 点。"""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    intrinsic_mat = np.asarray(intrinsic_mat, dtype=np.float64).reshape(3, 3)

    if len(uv) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if use_distortion:
        dist = np.asarray(distortion_matrix, dtype=np.float64).reshape(1, -1)
        # 不传 P 时 undistortPoints 直接返回归一化平面坐标，即去畸变后的 (x/z, y/z)
        norm = cv2.undistortPoints(uv.reshape(-1, 1, 2), intrinsic_mat, dist).reshape(-1, 2)
    else:
        fx, fy = intrinsic_mat[0, 0], intrinsic_mat[1, 1]
        cx, cy = intrinsic_mat[0, 2], intrinsic_mat[1, 2]
        norm = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy], axis=1)

    return np.stack([norm[:, 0] * depth, norm[:, 1] * depth, depth], axis=1)


def center_point_encoder(gt_box9d_with_cls,
                         intrinsic_mat,
                         extrinsic_mat,
                         distortion_matrix,
                         new_im_width=None,
                         new_im_hight=None,
                         raw_im_width=None,
                         raw_im_hight=None,
                         stride=None,
                         im_num=None,
                         class_name_config=None,
                         center_rad=None,
                         use_distortion=True):
    """把 (N, 10) 的 [x,y,z,l,w,h,a1,a2,a3,cls] 编码成 CenterNet 风格监督图。

    返回 (hm, center_res, center_dis, dim, rot)，形状依次为
    (im_num, C, H/s, W/s) / (im_num, 2, ...) / (im_num, 1, ...) /
    (im_num, 3, ...) / (im_num, 6, ...)。
    """
    cls_num = len(class_name_config)
    hm_height = int(new_im_hight) // int(stride)
    hm_width = int(new_im_width) // int(stride)

    scale_x = float(new_im_width) / float(raw_im_width) / float(stride)
    scale_y = float(new_im_hight) / float(raw_im_hight) / float(stride)

    gt_box9d_with_cls = np.asarray(gt_box9d_with_cls, dtype=np.float64).reshape(-1, 10)

    gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = [], [], [], [], []

    for im_id in range(im_num):
        in_mat = np.asarray(intrinsic_mat[im_id], dtype=np.float64).reshape(3, 3)
        dis_mat = np.asarray(distortion_matrix[im_id], dtype=np.float64).reshape(-1)

        this_heat_map = torch.zeros(cls_num, hm_height, hm_width)
        this_res_map = np.zeros(shape=(2, hm_height, hm_width), dtype=np.float32)
        this_dis_map = np.zeros(shape=(1, hm_height, hm_width), dtype=np.float32)
        this_size_map = np.zeros(shape=(3, hm_height, hm_width), dtype=np.float32)
        this_angle_map = np.zeros(shape=(6, hm_height, hm_width), dtype=np.float32)

        if len(gt_box9d_with_cls) > 0:
            xyz = copy.deepcopy(gt_box9d_with_cls[:, 0:3])
            uv, depth = project_points(xyz, in_mat, dis_mat, use_distortion)

            for obj_i, obj in enumerate(gt_box9d_with_cls):
                Z = depth[obj_i]
                if not np.isfinite(Z) or Z <= 1e-6:
                    continue

                u_hm = uv[obj_i, 0] * scale_x
                v_hm = uv[obj_i, 1] * scale_y
                if not (np.isfinite(u_hm) and np.isfinite(v_hm)):
                    continue

                w_idx, h_idx = int(u_hm), int(v_hm)
                # 越界目标直接丢弃：draw_gaussian_to_heatmap 遇到负下标会做环绕切片，
                # draw_res_to_heatmap 在 y == height 时会越界
                if not (0 <= h_idx < hm_height and 0 <= w_idx < hm_width):
                    continue

                this_cls = int(obj[-1])
                if not (0 <= this_cls < cls_num):
                    continue

                center = [u_hm, v_hm]
                this_heat_map[this_cls] = draw_gaussian_to_heatmap(
                    this_heat_map[this_cls], center, center_rad)
                this_res_map[0], this_res_map[1] = draw_res_to_heatmap(
                    this_res_map[0], this_res_map[1], center)

                this_dis_map[0, h_idx, w_idx] = Z

                l, w, h = obj[3], obj[4], obj[5]
                a1, a2, a3 = obj[6], obj[7], obj[8]

                this_size_map[0, h_idx, w_idx] = l
                this_size_map[1, h_idx, w_idx] = w
                this_size_map[2, h_idx, w_idx] = h

                this_angle_map[0, h_idx, w_idx] = np.cos(a1)
                this_angle_map[1, h_idx, w_idx] = np.sin(a1)
                this_angle_map[2, h_idx, w_idx] = np.cos(a2)
                this_angle_map[3, h_idx, w_idx] = np.sin(a2)
                this_angle_map[4, h_idx, w_idx] = np.cos(a3)
                this_angle_map[5, h_idx, w_idx] = np.sin(a3)

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dis.append(this_dis_map)
        gt_dim.append(this_size_map)
        gt_rot.append(this_angle_map)

    return (np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis),
            np.array(gt_dim), np.array(gt_rot))


def nms_pytorch(confidence_map, max_num, distance_threshold=5):
    """在 2D 置信度图上做基于距离的 NMS，返回 (置信度, 展平下标)。"""
    device = confidence_map.device
    H, W = confidence_map.shape

    flat_conf = confidence_map.flatten()
    initial_k = min(max(max_num * 4, 1), flat_conf.numel())
    top_conf, top_indices = torch.topk(flat_conf, k=initial_k)

    top_y = (top_indices // W).float()
    top_x = (top_indices % W).float()
    top_coords = torch.stack([top_x, top_y], dim=1)

    keep = torch.ones(len(top_indices), dtype=torch.bool, device=device)
    for i in range(len(top_indices)):
        if not keep[i]:
            continue
        rest = top_coords[i + 1:]
        if len(rest) == 0:
            break
        d = torch.norm(rest - top_coords[i:i + 1], dim=1)
        keep[i + 1:] = keep[i + 1:] & (d >= distance_threshold)

    sel = torch.where(keep)[0][:max_num]
    return top_conf[sel], top_indices[sel]


def center_point_decoder(hm,
                         center_res,
                         center_dis,
                         dim,
                         rot,
                         intrinsic_mat,
                         extrinsic_mat,
                         distortion_matrix,
                         new_im_width=None,
                         new_im_hight=None,
                         raw_im_width=None,
                         raw_im_hight=None,
                         stride=None,
                         im_num=None,
                         max_num=10,
                         use_distortion=True):
    """center_point_encoder 的逆过程，输出 (N, 10) 的 [x,y,z,l,w,h,a1,a2,a3,cls]。

    坐标系为 OpenCV 相机系，与 MAV6D 的 gt_box9d 一致
    （extrinsic 恒为单位阵：read_truth_Rt 已把位姿变换到相机系下）。
    """
    pred_boxes9d = []
    all_confidence = []

    scale_x = float(new_im_width) / float(raw_im_width) / float(stride)
    scale_y = float(new_im_hight) / float(raw_im_hight) / float(stride)

    for im_id in range(im_num):
        in_mat = np.asarray(intrinsic_mat[im_id], dtype=np.float64).reshape(3, 3)
        dis_mat = np.asarray(distortion_matrix[im_id], dtype=np.float64).reshape(-1)

        this_hm = hm[im_id]                       # (C, H, W)
        this_hm_conf, cls_map = this_hm.max(0)

        confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=2)

        W_hm = this_hm_conf.shape[-1]
        rows = (linear_indices // W_hm).long()
        cols = (linear_indices % W_hm).long()

        res_x = center_res[im_id][0, rows, cols]
        res_y = center_res[im_id][1, rows, cols]

        u_hm = (cols.float() + res_x).detach().cpu().numpy()
        v_hm = (rows.float() + res_y).detach().cpu().numpy()

        # 回到原始分辨率的像素坐标
        uv_raw = np.stack([u_hm / scale_x, v_hm / scale_y], axis=1)

        depth = center_dis[im_id][0, rows, cols].detach().cpu().numpy()

        points = unproject_points(uv_raw, depth, in_mat, dis_mat, use_distortion)

        this_dim = dim[im_id]
        this_rot = rot[im_id]

        l = this_dim[0, rows, cols].detach().cpu().numpy().reshape(-1, 1)
        w = this_dim[1, rows, cols].detach().cpu().numpy().reshape(-1, 1)
        h = this_dim[2, rows, cols].detach().cpu().numpy().reshape(-1, 1)

        a1 = torch.atan2(this_rot[1, rows, cols], this_rot[0, rows, cols])
        a2 = torch.atan2(this_rot[3, rows, cols], this_rot[2, rows, cols])
        a3 = torch.atan2(this_rot[5, rows, cols], this_rot[4, rows, cols])
        a1 = a1.detach().cpu().numpy().reshape(-1, 1)
        a2 = a2.detach().cpu().numpy().reshape(-1, 1)
        a3 = a3.detach().cpu().numpy().reshape(-1, 1)

        cls = cls_map[rows, cols].detach().cpu().numpy().reshape(-1, 1).astype(np.float64)

        pred_boxes9d.append(np.concatenate([points, l, w, h, a1, a2, a3, cls], -1))
        all_confidence.append(confi.detach().cpu().numpy())

    return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)
