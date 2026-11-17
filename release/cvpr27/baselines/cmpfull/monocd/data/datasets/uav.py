# -*- coding: utf-8 -*-
"""[UAV port] Dataset for the unified cmpfull UAV data (D:/cmpfull/<split>/labels.pkl).

This is a line-by-line adaptation of data/datasets/kitti.py::KITTIDataset.__getitem__ (MonoCD/MonoFlex).
Everything that produces training targets (heatmap, 2D box, 3D-center offset, edge fusion indices,
10 keypoints + visibility + keypoint-depth masks, dims, depth, multi-bin orientation) follows the official
code; only the KITTI-specific parts are replaced:
  * labels: read labels.pkl (box geometric center, lwh, euler_xyz, R) instead of KITTI txt + calib txt
  * single class 'drone' (cls_id 0)
  * angles: the unified KITTI-style convention of cmpfull/angles.py, ypr = [yaw, pitch, roll],
    R = Ry(yaw) Rx(pitch) Rz(roll) @ R0. MonoFlex's internal object frame is KITTI's (x = length, y = height
    pointing DOWN, z = width), which is exactly the "level" frame of angles.py, so R_int = R @ R0^T =
    Ry(yaw) Rx(pitch) Rz(roll) and dims_int = KITTI (l, h, w) = (lwh[0], lwh[2], lwh[1]).
    yaw is KITTI's ry: the official alpha = ry - atan2(x, z) multi-bin encoding is applied to yaw unchanged;
    pitch and roll use the same 4-bin multi-bin + residual encoding on the absolute angle (equal-weight losses).
    (euler_xyz is NOT used for supervision: its middle angle is limited to +-90 deg and the other two jump.)
  * keypoints / corners computed with the full rotation R_int (not ry)
  * EL ("bottom y" in KITTI) := center_y + h/2 so that the official compensated-depth formula, which uses
    EL - H/2, sees the camera-frame Y of the box center.
  * image: resized to 960x540 and zero-padded at the bottom to 960x544 (SPEC), K scaled (no shift);
    pad_size = (0, 0) because padding is at the bottom (official pads centrally; pad_size handles both).
  * filtering: KITTI occlusion is not available (treated as 0). KITTI truncation is replaced by its
    definition computed from the projected 3D box (fraction of the projected 2D box outside the image).
    The KITTI-benchmark-specific 'FILTER_MORE_STRICTLY' thresholds (2D height >= 25 px, 2 m <= depth <= 65 m)
    are NOT applied (cfg.UAV.STRICT_HEIGHT_DEPTH_FILTER=False): they encode KITTI's evaluation protocol
    (Moderate difficulty) and would remove 45% of sim drones (flat objects, median 2D height 30 px) and all
    drones closer than 2 m, while our evaluation counts every GT drone.  The truncation <= 0.5 rule is kept.
"""
import os
import logging
import pickle
import random

import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

from model.heatmap_coder import gaussian_radius, draw_umich_gaussian, draw_umich_gaussian_2D
from structures.params_3d import ParamsList
from .kitti_utils import Calibration, approx_proj_center

# cmpfull/angles.py: R0 = our object frame -> level (KITTI internal) frame
R0 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])


def euler_xyz_to_matrix_np(e):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler('xyz', e).as_matrix()


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class UAVCalibration(Calibration):
    """Pinhole camera with P = [K | 0]; provides the attributes/methods MonoFlex uses from KITTI Calibration."""

    def __init__(self, K):
        K = np.asarray(K, np.float64)
        self.P = np.concatenate([K, np.zeros((3, 1))], axis=1)
        self.refresh()

    def refresh(self):
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)
        self.b_y = self.P[1, 3] / (-self.f_v)


def ypr_from_R(R):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(np.asarray(R) @ R0.T).as_euler('YXZ')


def R_level_from_ypr(ypr):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler('YXZ', np.asarray(ypr)).as_matrix()


def box_internal(b):
    """our box dict -> (center, dims_int (l,h,w), R_int, ypr). ypr from the label if present (sim labels), else
    computed from R with the same formula as cmpfull/angles.py (e.g. pseudo-label pkls)."""
    R = np.asarray(b['R'], np.float64)
    lwh = np.asarray(b['lwh'], np.float64)
    ypr = np.asarray(b['ypr'], np.float64).copy() if 'ypr' in b else ypr_from_R(R)
    return (np.asarray(b['center'], np.float64).copy(), np.array([lwh[0], lwh[2], lwh[1]]), R @ R0.T, ypr)


def corners_kitti_order(center, dims, R_int):
    """Same corner order as kitti_utils.Object3d.generate_corners3d, but around the geometric center and
    rotated by the full R_int (KITTI: bottom-center origin and ry only). 0-3: bottom face (+y), 4-7: top."""
    l, h, w = dims
    x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y = [h / 2] * 4 + [-h / 2] * 4
    z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    return (R_int @ np.array([x, y, z])).T + center


class UAVDataset(Dataset):
    def __init__(self, cfg, label_files, is_train=True, transforms=None, augment=True, with_targets=True):
        super().__init__()
        self.cfg = cfg
        self.records = []
        if isinstance(label_files, str):
            label_files = [label_files]
        for lf in label_files:
            root = os.path.dirname(os.path.abspath(lf))
            for r in pickle.load(open(lf, 'rb')):
                r = dict(r)
                r['_path'] = os.path.join(root, r['file'])
                self.records.append(r)
        self.is_train = is_train
        self.with_targets = with_targets
        self.transforms = transforms
        self.num_samples = len(self.records)
        self.classes = cfg.DATASETS.DETECT_CLASSES
        self.num_classes = len(self.classes)
        self.flip_prob = cfg.INPUT.AUG_PARAMS[0][0] if (is_train and augment) else 0.0

        self.img_w, self.img_h = cfg.UAV.IMG_W, cfg.UAV.IMG_H                    # 960 x 540
        self.input_width = cfg.INPUT.WIDTH_TRAIN                                 # 960
        self.input_height = cfg.INPUT.HEIGHT_TRAIN                               # 544
        self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_width = self.input_width // self.down_ratio
        self.output_height = self.input_height // self.down_ratio
        self.max_edge_length = (self.output_width + self.output_height) * 2
        self.max_objs = cfg.DATASETS.MAX_OBJECTS

        self.filter_annos = cfg.DATASETS.FILTER_ANNO_ENABLE
        self.filter_params = cfg.DATASETS.FILTER_ANNOS
        self.consider_outside_objs = cfg.DATASETS.CONSIDER_OUTSIDE_OBJS
        self.proj_center_mode = cfg.INPUT.APPROX_3D_CENTER
        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
        self.use_modify_keypoint_visible = cfg.INPUT.KEYPOINT_VISIBLE_MODIFY
        self.orientation_method = cfg.INPUT.ORIENTATION
        assert self.orientation_method == 'multi-bin'
        self.multibin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
        self.alpha_centers = np.array([0, np.pi / 2, np.pi, - np.pi / 2])
        self.num_angles = cfg.UAV.NUM_ANGLES
        self.yaw_index = cfg.UAV.YAW_INDEX
        self.heatmap_center = cfg.INPUT.HEATMAP_CENTER
        self.adjust_edge_heatmap = cfg.INPUT.ADJUST_BOUNDARY_HEATMAP
        self.edge_heatmap_ratio = cfg.INPUT.HEATMAP_RATIO

        # KITTI strict filter: only the truncation part is kept (see module docstring)
        self.filter_more_strictly = cfg.DATASETS.FILTER_MORE_STRICTLY
        self.strict_height_depth = cfg.UAV.STRICT_HEIGHT_DEPTH_FILTER
        self.min_height, self.min_depth, self.max_depth, self.max_truncation = 25, 2, 65, 0.5
        logging.getLogger("monocd.dataset").info("UAV dataset: %d frames from %s" % (self.num_samples, label_files))

    def __len__(self):
        return self.num_samples

    # ---- unchanged official helpers -------------------------------------------------------------------------
    def get_edge_utils(self, image_size, pad_size, down_ratio=4):
        img_w, img_h = image_size
        x_min, y_min = np.ceil(pad_size[0] / down_ratio), np.ceil(pad_size[1] / down_ratio)
        x_max, y_max = (pad_size[0] + img_w - 1) // down_ratio, (pad_size[1] + img_h - 1) // down_ratio
        step = 1
        edge_indices = []
        y = torch.arange(y_min, y_max, step); x = torch.ones(len(y)) * x_min
        e = torch.stack((x, y), dim=1); e[:, 0] = torch.clamp(e[:, 0], x_min); e[:, 1] = torch.clamp(e[:, 1], y_min)
        edge_indices.append(torch.unique(e, dim=0))
        x = torch.arange(x_min, x_max, step); y = torch.ones(len(x)) * y_max
        e = torch.stack((x, y), dim=1); e[:, 0] = torch.clamp(e[:, 0], x_min); e[:, 1] = torch.clamp(e[:, 1], y_min)
        edge_indices.append(torch.unique(e, dim=0))
        y = torch.arange(y_max, y_min, -step); x = torch.ones(len(y)) * x_max
        e = torch.stack((x, y), dim=1); e[:, 0] = torch.clamp(e[:, 0], x_min); e[:, 1] = torch.clamp(e[:, 1], y_min)
        edge_indices.append(torch.unique(e, dim=0).flip(dims=[0]))
        x = torch.arange(x_max, x_min - 1, -step); y = torch.ones(len(x)) * y_min
        e = torch.stack((x, y), dim=1); e[:, 0] = torch.clamp(e[:, 0], x_min); e[:, 1] = torch.clamp(e[:, 1], y_min)
        edge_indices.append(torch.unique(e, dim=0).flip(dims=[0]))
        return torch.cat([index.long() for index in edge_indices], dim=0)

    def encode_alpha_multibin(self, alpha, num_bin=2, margin=1 / 6):
        encode_alpha = np.zeros(num_bin * 2)
        bin_size = 2 * np.pi / num_bin
        margin_size = bin_size * margin
        bin_centers = self.alpha_centers
        range_size = bin_size / 2 + margin_size
        offsets = alpha - bin_centers
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi
        for i in range(num_bin):
            offset = offsets[i]
            if abs(offset) < range_size:
                encode_alpha[i] = 1
                encode_alpha[i + num_bin] = offset
        return encode_alpha

    # ---- UAV-specific -----------------------------------------------------------------------------------------
    def encode_orientation(self, euler, loc):
        """3 angles x official multi-bin: [cls(4)+off(4)] per angle, concatenated -> 3*8.
        The yaw angle (index YAW_INDEX) is turned into the observation angle alpha = yaw - atan2(x, z)."""
        out, alpha = [], None
        for k in range(self.num_angles):
            a = euler[k]
            if k == self.yaw_index:
                a = wrap_pi(a - np.arctan2(loc[0], loc[2]))
                alpha = a
            out.append(self.encode_alpha_multibin(a, num_bin=self.multibin_size))
        return np.concatenate(out), alpha

    def load_image_and_calib(self, r):
        img = Image.open(r['_path']).convert('RGB')
        W, H = img.size
        sx, sy = self.img_w / W, self.img_h / H
        if (W, H) != (self.img_w, self.img_h):
            img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
        K = np.asarray(r['K'], np.float64).copy()
        K[0] *= sx
        K[1] *= sy
        return img, UAVCalibration(K)

    def pad_image(self, image):
        """SPEC: zero padding at the bottom (960x540 -> 960x544); pad_size = (0, 0)."""
        img = np.array(image)
        h, w, c = img.shape
        ret_img = np.zeros((self.input_height, self.input_width, c), dtype=np.uint8)
        ret_img[:h, :w] = img
        return Image.fromarray(ret_img), np.array([0, 0])

    def __getitem__(self, idx):
        r = self.records[idx]
        img, calib = self.load_image_and_calib(r)
        img_w, img_h = img.size

        objs = []
        for b in (r['boxes'] if self.with_targets else []):
            objs.append(box_internal(b))

        # random horizontal flip (official RandomHorizontallyFlip, generalised to a full rotation):
        # official: x -> -x, ry -> pi - ry (heading mirrored, width axis reflected). With the full rotation:
        # R_int' = M R_int S_z (M = diag(-1,1,1) camera mirror, S_z = diag(1,1,-1) object width reflection)
        #        = Ry(pi - yaw) Rx(-pitch) Rz(roll)   ->  ypr' = (pi - yaw, -pitch, roll); cx' = W-1-cx
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            M = np.diag([-1., 1., 1.])
            Sz = np.diag([1., 1., -1.])
            objs = [(M @ c, d, M @ Ri @ Sz, np.array([wrap_pi(np.pi - e[0]), -e[1], e[2]])) for (c, d, Ri, e) in objs]
            P2 = calib.P.copy()
            P2[0, 2] = img_w - P2[0, 2] - 1
            P2[0, 3] = - P2[0, 3]
            calib.P = P2
            calib.refresh()

        img, pad_size = self.pad_image(img)
        ori_img = np.array(img).copy()

        x_min, y_min = int(np.ceil(pad_size[0] / self.down_ratio)), int(np.ceil(pad_size[1] / self.down_ratio))
        x_max, y_max = (pad_size[0] + img_w - 1) // self.down_ratio, (pad_size[1] + img_h - 1) // self.down_ratio

        if self.enable_edge_fusion:
            input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int64)
            edge_indices = self.get_edge_utils((img_w, img_h), pad_size).numpy()
            input_edge_count = edge_indices.shape[0]
            input_edge_indices[:edge_indices.shape[0]] = edge_indices
            input_edge_count = input_edge_count - 1

        if not self.with_targets:
            target = ParamsList(image_size=img.size, is_train=self.is_train)
            target.add_field("pad_size", pad_size)
            target.add_field("calib", calib)
            if self.enable_edge_fusion:
                target.add_field('edge_len', input_edge_count)
                target.add_field('edge_indices', input_edge_indices)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, idx

        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        target_centers = np.zeros([self.max_objs, 2], dtype=np.int32)
        gt_bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
        bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
        keypoints = np.zeros([self.max_objs, 10, 3], dtype=np.float32)
        keypoints_depth_mask = np.zeros([self.max_objs, 3], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        EL = np.zeros([self.max_objs], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)                 # yaw component (logging only)
        alphas = np.zeros([self.max_objs], dtype=np.float32)
        eulers = np.zeros([self.max_objs, 3], dtype=np.float32)
        rot_mats = np.tile(np.eye(3, dtype=np.float32), (self.max_objs, 1, 1))   # R_int
        offset_3D = np.zeros([self.max_objs, 2], dtype=np.float32)
        occlusions = np.zeros(self.max_objs)
        truncations = np.zeros(self.max_objs)
        orientations = np.zeros([self.max_objs, self.num_angles * self.multibin_size * 2], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        trunc_mask = np.zeros([self.max_objs], dtype=np.uint8)
        reg_weight = np.zeros([self.max_objs], dtype=np.float32)

        i = -1
        for (locs, dims, R_int, euler) in objs:
            if locs[-1] <= 0:
                continue
            corners_3d = corners_kitti_order(locs, dims, R_int)
            if (corners_3d[:, 2] <= 0).any():
                continue  # (not reachable on this data: all boxes are >= 1.3 m in front)
            corners_2d, _ = calib.project_rect_to_image(corners_3d)
            projected_box2d = np.array([corners_2d[:, 0].min(), corners_2d[:, 1].min(),
                                        corners_2d[:, 0].max(), corners_2d[:, 1].max()])
            # KITTI annotated 2D box -> projected box clipped to the image (KITTI uses the projected box when it
            # is fully inside the image, otherwise the annotated, i.e. visible, box)
            clipped = np.array([max(projected_box2d[0], 0), max(projected_box2d[1], 0),
                                min(projected_box2d[2], img_w - 1), min(projected_box2d[3], img_h - 1)])
            area_full = max(projected_box2d[2] - projected_box2d[0], 1e-6) * max(projected_box2d[3] - projected_box2d[1], 1e-6)
            area_vis = max(clipped[2] - clipped[0], 0) * max(clipped[3] - clipped[1], 0)
            float_truncation = float(np.clip(1 - area_vis / area_full, 0, 1))
            float_occlusion = 0.0
            # filtrate_objects (strict): truncation (+ optionally the KITTI height/depth thresholds)
            if self.filter_more_strictly:
                if float_truncation > self.max_truncation:
                    continue
                if self.strict_height_depth and ((clipped[3] - clipped[1]) < self.min_height or
                                                 locs[2] > self.max_depth or locs[2] < self.min_depth):
                    continue
            if projected_box2d[0] >= 0 and projected_box2d[1] >= 0 and \
                    projected_box2d[2] <= img_w - 1 and projected_box2d[3] <= img_h - 1:
                box2d = projected_box2d.copy()
            else:
                box2d = clipped.copy()
            if self.filter_annos:
                if float_truncation >= self.filter_params[0] and (box2d[2:] - box2d[:2]).min() <= self.filter_params[1]:
                    continue

            proj_center, depth = calib.project_rect_to_image(locs.reshape(-1, 3))
            proj_center = proj_center[0]
            proj_inside_img = (0 <= proj_center[0] <= img_w - 1) & (0 <= proj_center[1] <= img_h - 1)
            approx_center = False
            if not proj_inside_img:
                if self.consider_outside_objs:
                    approx_center = True
                    center_2d = (box2d[:2] + box2d[2:]) / 2
                    if self.proj_center_mode == 'intersect':
                        target_proj_center, edge_index = approx_proj_center(proj_center, center_2d.reshape(1, 2), (img_w, img_h))
                        if target_proj_center is None:
                            continue
                    else:
                        raise NotImplementedError
                else:
                    continue
            else:
                target_proj_center = proj_center.copy()

            bot_top_centers = np.stack((corners_3d[:4].mean(axis=0), corners_3d[4:].mean(axis=0)), axis=0)
            keypoints_3D = np.concatenate((corners_3d, bot_top_centers), axis=0)
            keypoints_2D, _ = calib.project_rect_to_image(keypoints_3D)
            keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (keypoints_2D[:, 0] <= img_w - 1)
            keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (keypoints_2D[:, 1] <= img_h - 1)
            keypoints_z_visible = (keypoints_3D[:, -1] > 0)
            keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
            keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))
            if self.use_modify_keypoint_visible:
                keypoints_visible = np.append(np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2), np.tile(keypoints_visible[8] | keypoints_visible[9], 2))
                keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))
                keypoints_visible = keypoints_visible.astype(np.float32)
                keypoints_depth_valid = keypoints_depth_valid.astype(np.float32)

            keypoints_2D = (keypoints_2D + pad_size.reshape(1, 2)) / self.down_ratio
            target_proj_center = (target_proj_center + pad_size) / self.down_ratio
            proj_center = (proj_center + pad_size) / self.down_ratio
            gt_box2d = box2d.copy()
            box2d[0::2] += pad_size[0]
            box2d[1::2] += pad_size[1]
            box2d /= self.down_ratio
            bbox_center = (box2d[:2] + box2d[2:]) / 2
            bbox_dim = box2d[2:] - box2d[:2]
            if self.heatmap_center == '2D':
                target_center = bbox_center.round().astype(int)
            else:
                target_center = target_proj_center.round().astype(int)
            target_center[0] = np.clip(target_center[0], x_min, x_max)
            target_center[1] = np.clip(target_center[1], y_min, y_max)
            pred_2D = True
            if not (target_center[0] >= box2d[0] and target_center[1] >= box2d[1] and target_center[0] <= box2d[2] and target_center[1] <= box2d[3]):
                pred_2D = False

            if (bbox_dim > 0).all() and (0 <= target_center[0] <= self.output_width - 1) and (0 <= target_center[1] <= self.output_height - 1):
                i += 1
                if i >= self.max_objs:
                    break
                cls_id = 0
                if self.adjust_edge_heatmap and approx_center:
                    bbox_width = min(target_center[0] - box2d[0], box2d[2] - target_center[0])
                    bbox_height = min(target_center[1] - box2d[1], box2d[3] - target_center[1])
                    radius_x, radius_y = bbox_width * self.edge_heatmap_ratio, bbox_height * self.edge_heatmap_ratio
                    radius_x, radius_y = max(0, int(radius_x)), max(0, int(radius_y))
                    assert min(radius_x, radius_y) == 0
                    heat_map[cls_id] = draw_umich_gaussian_2D(heat_map[cls_id], target_center, radius_x, radius_y)
                else:
                    radius = gaussian_radius(bbox_dim[1], bbox_dim[0])
                    radius = max(0, int(radius))
                    heat_map[cls_id] = draw_umich_gaussian(heat_map[cls_id], target_center, radius)

                cls_ids[i] = cls_id
                target_centers[i] = target_center
                offset_3D[i] = proj_center - target_center
                gt_bboxes[i] = gt_box2d
                if pred_2D:
                    bboxes[i] = box2d
                keypoints[i] = np.concatenate((keypoints_2D - target_center.reshape(1, -1), keypoints_visible[:, np.newaxis]), axis=1)
                keypoints_depth_mask[i] = keypoints_depth_valid
                dimensions[i] = dims
                locations[i] = locs
                orientations[i], alpha = self.encode_orientation(euler, locs)
                rotys[i] = euler[self.yaw_index]
                alphas[i] = alpha
                eulers[i] = euler
                rot_mats[i] = R_int
                EL[i] = locs[1] + dims[1] / 2
                reg_mask[i] = 1
                reg_weight[i] = 1
                trunc_mask[i] = int(approx_center)
                occlusions[i] = float_occlusion
                truncations[i] = float_truncation

        target = ParamsList(image_size=img.size, is_train=self.is_train)
        target.add_field("cls_ids", cls_ids)
        target.add_field("target_centers", target_centers)
        target.add_field("keypoints", keypoints)
        target.add_field("keypoints_depth_mask", keypoints_depth_mask)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("EL", EL)
        target.add_field("calib", calib)
        target.add_field("reg_mask", reg_mask)
        target.add_field("reg_weight", reg_weight)
        target.add_field("offset_3D", offset_3D)
        target.add_field("2d_bboxes", bboxes)
        target.add_field("pad_size", pad_size)
        target.add_field("ori_img", ori_img)
        target.add_field("rotys", rotys)
        target.add_field("eulers", eulers)
        target.add_field("rot_mats", rot_mats)
        target.add_field("trunc_mask", trunc_mask)
        target.add_field("alphas", alphas)
        target.add_field("orientations", orientations)
        target.add_field("hm", heat_map)
        target.add_field("gt_bboxes", gt_bboxes)
        target.add_field("occlusions", occlusions)
        target.add_field("truncations", truncations)
        if self.enable_edge_fusion:
            target.add_field('edge_len', input_edge_count)
            target.add_field('edge_indices', input_edge_indices)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, idx
