# -*- coding: utf-8 -*-
"""[UAV port] Drop-in replacement of lib/datasets/kitti/kitti_dataset.py for the unified cmpfull data
(D:/cmpfull/<split>/labels.pkl, see cmpfull/SPEC.md).

The structure of KITTI_Dataset.__getitem__ is kept line by line (photometric distortion, random flip,
random crop/scale/shift affine, mixup3d, 3D-center-in-image filter, l/r/t/b >= 0 filter, depth*crop_scale,
same target dict).  Only the KITTI-specific label semantics are replaced:
  * label source: labels.pkl (center = box geometric center, lwh, euler_xyz, R) instead of label_2/calib
  * one class 'drone' (cls id 0)
  * 3D center = geometric center (KITTI: pos + [0, -h/2, 0], i.e. also the geometric center)
  * 2D box = tight box of the 8 projected corners, clipped to the image (KITTI: annotated 2D box)
  * KITTI 'UnKnown' level filter -> truncation <= 0.5 (truncation = 1 - clipped/unclipped projected box area);
    the 25-px min-height part is KITTI evaluation protocol and is off by default (cfg min_box_height)
  * KITTI 'pos_z < 2' filter -> cfg min_depth (default 0: our drones are 1.4-8 m away)
  * heading: 3 Euler angles, each with the original 12-bin angle2class encoding;
    angles = KITTI-style ypr = [yaw, pitch, roll] from cmpfull/angles.py (R = Ry(yaw) Rx(pitch) Rz(roll) R0);
    yaw IS the KITTI ry and is encoded as alpha = ry - atan2(u - cu, fu) exactly as the original;
    pitch, roll (within +-30 deg in sim) are encoded as absolute angles with the same 12-bin scheme.
  * size_3d order [h, w, l] as KITTI, with h = lwh[2] (object z is the drone's up axis), w = lwh[1], l = lwh[0]
  * flip: R' = S R D (S = diag(-1,1,1) mirrors the camera x axis, D = diag(1,-1,1) flips the object's lateral
    axis so the front (x) and up (z) axes are mirrored, not reversed) == KITTI's ry -> pi - ry.
Input: every image resized to 960x540, zero-padded at the bottom to 960x544, K scaled accordingly
(this 960x544 canvas plays the role of the KITTI 'original image'; resolution = 960x544).
"""
import os
import pickle
import random
import numpy as np
import cv2
import torch.utils.data as data
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from lib.datasets.kitti.pd import PhotometricDistort
from lib.datasets.utils import angle2class, class2angle
from lib.datasets.kitti.kitti_utils import get_affine_transform, affine_transform

# SPEC: resize to 960x540 and zero-pad the bottom to 960x544. MonoDGP's RegionSegHead upsamples the stride-64
# level x2 and adds it to the stride-32 level, so the input height must be a multiple of 64 (544/32 = 17 is odd ->
# 9*2 != 17 crash). We therefore pad the bottom to 576 instead of 544: same 960x540 image content, same K, only
# 32 more zero rows (the official KITTI input 1280x384 is also a multiple of 64). No model code is changed.
CANVAS_W, CANVAS_H, IMG_H = 960, 576, 540
PROTO = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
S_FLIP = np.diag([-1., 1., 1.])
D_FLIP = np.diag([1., -1., 1.])
YAW_IDX = 0  # ypr[0] = yaw = KITTI ry (rotation about the camera y axis)
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))  # cmpfull/
import angles as ANG  # cmpfull/angles.py: unified ypr <-> R <-> euler_xyz conventions


def wrap(a):
    a = a % (2 * np.pi)
    if a > np.pi:
        a -= 2 * np.pi
    return a


def scaled_K(K, W, H):
    K = np.asarray(K, np.float64).copy()
    K[0] *= CANVAS_W / float(W)
    K[1] *= IMG_H / float(H)
    return K


def load_canvas(path):
    """read -> RGB -> resize to 960x540 -> zero pad bottom to 960x576 (PIL RGB, like KITTI get_image)."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    assert im is not None, path
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (CANVAS_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    out[:IMG_H] = im
    return Image.fromarray(out)


def encode_angles(euler, u2d, cu, fu):
    """euler (3,), u2d = 2D-box center x on the canvas. Returns bins (3,), res (3,)."""
    bins, res = np.zeros(3, np.int64), np.zeros(3, np.float32)
    for k in range(3):
        a = euler[k]
        if k == YAW_IDX:
            a = a - np.arctan2(u2d - cu, fu)  # == calib.ry2alpha
        a = wrap(a)
        bins[k], res[k] = angle2class(a)
    return bins, res


def decode_angles(heading72, u2d, cu, fu):
    """heading72: (72,) network output (per angle: 12 cls logits + 12 residuals). Returns ypr (3,)."""
    eul = np.zeros(3)
    for k in range(3):
        h = heading72[24 * k: 24 * (k + 1)]
        cls = int(np.argmax(h[:12]))
        a = class2angle(cls, h[12 + cls], to_label_format=True)
        if k == YAW_IDX:
            a = a + np.arctan2(u2d - cu, fu)  # == calib.alpha2ry
        eul[k] = wrap(a)
    return eul


class Obj(object):
    """KITTI Object3d analogue, all pixel quantities on the 960x544 canvas."""

    def __init__(self, b, K):
        self.center = np.asarray(b['center'], np.float64)
        self.lwh = np.asarray(b['lwh'], np.float64)
        self.R = np.asarray(b['R'], np.float64)
        self.euler = np.asarray(b['ypr'], np.float64).copy() if 'ypr' in b else ANG.ypr_from_R(self.R)  # [yaw, pitch, roll]
        self.cls_type = 'drone'
        self.occlusion = 0
        self.h = self.lwh[2]
        c = (PROTO * self.lwh) @ self.R.T + self.center
        self.valid = bool(np.all(c[:, 2] > 1e-2) and self.center[2] > 1e-2)
        if not self.valid:
            self.box2d = np.zeros(4, np.float32); self.trucation = 1.0
            return
        uv = c @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        x0, y0 = uv.min(0); x1, y1 = uv.max(0)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, CANVAS_W), min(y1, IMG_H)
        area = max(x1 - x0, 1e-6) * max(y1 - y0, 1e-6)
        area_c = max(cx1 - cx0, 0) * max(cy1 - cy0, 0)
        self.trucation = float(1 - area_c / area)
        self.box2d = np.array([cx0, cy0, cx1, cy1], np.float32)

    def flip(self, img_w):
        x1, x2 = self.box2d[0], self.box2d[2]
        self.box2d[0], self.box2d[2] = img_w - x2, img_w - x1
        # KITTI ry -> pi - ry. In this ypr convention the mirror that keeps nose/up (R' = S R D) is exactly
        # (yaw, pitch, roll) -> (pi - yaw, -pitch, roll)  (pitch = rotation about the nose axis; verified to 1e-15)
        self.R = S_FLIP @ self.R @ D_FLIP
        y, pch, rl = self.euler
        self.euler = np.array([wrap(np.pi - y), -pch, rl])


class UAV_Dataset(data.Dataset):
    def __init__(self, split, cfg, label_files=None):
        """split: 'train' (augment + targets) or 'test' (no aug, no targets). label_files: list of
        'path/to/labels.pkl' or 'path/to/labels.pkl@image_root' (image root defaults to the pkl's folder)."""
        self.split = split
        self.num_classes = 1
        self.max_objs = 50
        self.class_name = ['drone']
        self.cls2id = {'drone': 0}
        self.resolution = np.array([CANVAS_W, CANVAS_H])  # W * H
        self.writelist = ['drone']
        self.meanshape = cfg.get('meanshape', False)
        self.min_box_height = float(cfg.get('min_box_height', 0))
        self.min_depth = float(cfg.get('min_depth', 0))
        self.clip_2d = cfg.get('clip_2d', False)

        self.recs = []
        for lf in (label_files or cfg['label_files']):
            path, root = (lf.split('@', 1) + [None])[:2] if '@' in lf else (lf, None)
            root = root or os.path.dirname(os.path.abspath(path))
            for r in pickle.load(open(path, 'rb')):
                r = dict(r); r['_root'] = r.get('img_root', root)
                self.recs.append(r)
        self.idx_list = list(range(len(self.recs)))

        self.data_augmentation = split == 'train'
        self.istrain = split == 'train'
        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)
        self.random_mixup3d = cfg.get('random_mixup3d', 0.5)
        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)
        self.depth_scale = cfg.get('depth_scale', 'normal')

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # single-class mean size [h, w, l] (sim_train statistics); unused when meanshape False (official default)
        self.cls_mean_size = np.array([cfg.get('mean_size_hwl', [0.2507, 0.6198, 0.6464])], dtype=np.float32)
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        self.downsample = 32
        self.pd = PhotometricDistort()

    # ---------------- KITTI-API analogues ----------------
    def get_image(self, idx):
        r = self.recs[idx]
        return load_canvas(os.path.join(r['_root'], r['file']))

    def get_K(self, idx):
        r = self.recs[idx]
        return scaled_K(r['K'], r['W'], r['H'])

    def get_P2(self, idx):
        return np.concatenate([self.get_K(idx), np.zeros((3, 1))], 1).astype(np.float32)

    def get_label(self, idx):
        K = self.get_K(idx)
        return [Obj(b, K) for b in self.recs[idx]['boxes']]

    def __len__(self):
        return len(self.idx_list)

    def _filtered(self, obj):
        """KITTI: level_str == 'UnKnown' or pos[-1] < 2."""
        if not obj.valid or obj.trucation > 0.5:
            return True
        if (obj.box2d[3] - obj.box2d[1] + 1) < self.min_box_height:
            return True
        return obj.center[2] < self.min_depth

    def _encode(self, objects, start, n, K, trans, crop_scale, random_flip_flag, img_size, T, obj_region, check_threshold):
        cu, fu = K[0, 2], K[0, 0]
        P2 = np.concatenate([K, np.zeros((3, 1))], 1)
        for i in range(n):
            o = objects[i]
            j = i + start
            if o.cls_type not in self.writelist:
                continue
            if self._filtered(o):
                continue
            if check_threshold and o.center[2] > 65:  # [hard encoding] kept from the original
                continue

            bbox_2d = o.box2d.copy()
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)

            ymin, ymax = int(max(bbox_2d[1], 0)), int(min(bbox_2d[3], obj_region.shape[0]))
            xmin, xmax = int(max(bbox_2d[0], 0)), int(min(bbox_2d[2], obj_region.shape[1]))
            obj_region[ymin:ymax, xmin:xmax] = 1

            corner_2d = bbox_2d.copy()
            # projected 3D (geometric) center
            pc = K @ o.center
            center_3d = (pc[:2] / pc[2]).astype(np.float32)
            if random_flip_flag and not self.aug_calib:
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                continue
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                continue

            T['labels'][j] = self.cls2id[o.cls_type]
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            T['size_2d'][j] = 1. * w, 1. * h
            center_2d_norm = center_2d / self.resolution
            size_2d_norm = T['size_2d'][j] / self.resolution
            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution
            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]
            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l, r, t, b = [np.clip(v, 0, 1) for v in (l, r, t, b)]
                else:
                    continue
            T['boxes'][j] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            T['boxes_3d'][j] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            z = o.center[2]
            if self.depth_scale == 'normal':
                T['depth'][j] = z * crop_scale
            elif self.depth_scale == 'inverse':
                T['depth'][j] = z / crop_scale
            else:
                T['depth'][j] = z

            T['heading_bin'][j], T['heading_res'][j] = encode_angles(o.euler, (o.box2d[0] + o.box2d[2]) / 2, cu, fu)

            T['src_size_3d'][j] = np.array([o.lwh[2], o.lwh[1], o.lwh[0]], dtype=np.float32)  # [h, w, l]
            T['size_3d'][j] = T['src_size_3d'][j] - self.cls_mean_size[self.cls2id[o.cls_type]]
            if o.trucation <= 0.5 and o.occlusion <= 2:
                T['mask_2d'][j] = 1
            T['calibs'][j] = P2

    def __getitem__(self, item):
        index = int(self.idx_list[item])
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample
        dst_W, dst_H = img_size

        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag, random_mix_flag = False, False, False
        K = self.get_K(index)

        if self.data_augmentation:
            if np.random.random() < self.random_mixup3d:
                random_mix_flag = True
            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        if random_mix_flag:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = int(np.random.choice(self.idx_list))
                K_temp = self.get_K(random_index)
                if np.allclose(K_temp[[0, 1, 0, 1], [2, 2, 0, 1]], K[[0, 1, 0, 1], [2, 2, 0, 1]]):
                    img_temp = self.get_image(random_index)
                    dst_W_temp, dst_H_temp = np.array(img_temp.size)
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        if len(self.recs[index]['boxes']) + len(self.recs[random_index]['boxes']) < self.max_objs:
                            random_mix_flag = True
                            if random_flip_flag:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img = Image.blend(img, img_temp, alpha=0.5)
                            break

        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()), method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()), resample=Image.BILINEAR)

        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        info = {'img_id': index, 'img_size': img_size, 'bbox_downsample_ratio': img_size / features_size}
        P2 = self.get_P2(index)
        if self.split == 'test':
            return img, P2, img, info

        objects = self.get_label(index)
        if random_flip_flag:
            for o in objects:
                if o.valid:
                    o.flip(img_size[0])

        M = self.max_objs
        T = {'calibs': np.zeros((M, 3, 4), dtype=np.float32),
             'indices': np.zeros((M), dtype=np.int64),
             'img_size': img_size,
             'labels': np.zeros((M), dtype=np.int8),
             'boxes': np.zeros((M, 4), dtype=np.float32),
             'boxes_3d': np.zeros((M, 6), dtype=np.float32),
             'depth': np.zeros((M, 1), dtype=np.float32),
             'size_2d': np.zeros((M, 2), dtype=np.float32),
             'size_3d': np.zeros((M, 3), dtype=np.float32),
             'src_size_3d': np.zeros((M, 3), dtype=np.float32),
             'heading_bin': np.zeros((M, 3), dtype=np.int64),   # [UAV port] (M, 1) -> (M, 3)
             'heading_res': np.zeros((M, 3), dtype=np.float32),
             'mask_2d': np.zeros((M), dtype=bool)}
        obj_region = np.zeros((img.shape[1], img.shape[2]), dtype=bool)

        object_num = len(objects) if len(objects) < M else M
        self._encode(objects, 0, object_num, K, trans, crop_scale, random_flip_flag, img_size, T, obj_region, True)

        if random_mix_flag:
            objects = self.get_label(random_index)
            if random_flip_flag:
                for o in objects:
                    if o.valid:
                        o.flip(img_size[0])
            object_num_temp = len(objects) if len(objects) < (M - object_num) else (M - object_num)
            self._encode(objects, object_num, object_num_temp, self.get_K(random_index), trans, crop_scale,
                         random_flip_flag, img_size, T, obj_region, False)

        T['obj_region'] = obj_region
        return img, P2, T, info
