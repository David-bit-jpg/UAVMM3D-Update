# -*- coding: utf-8 -*-
"""[UAV 改动] 由 lib/datasets/kitti.py 改写：读统一数据 labels.pkl 代替 KITTI label_2/calib。
除下面标了 [UAV] 的地方，增广（flip/crop/scale/shift）、heatmap/offset/size/depth 编码全部照抄 kitti.py。
"""
import os
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image

from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.uav_utils import UAVCalib, corners_of, flip_ypr, ypr_to_code_angles, encode_angles, ypr_from_R, R_from_ypr

# [UAV] 统一输入：缩放到 960x540，下方补零到 960x544；K 只缩放不平移
NET_W, NET_H, IMG_H_VALID = 960, 544, 540


def load_records(label_paths):
    """--labels 可给多个 pkl；图像根目录 = 记录里的 'root'（若有）否则 pkl 所在目录。"""
    recs = []
    for p in label_paths:
        root = os.path.dirname(os.path.abspath(p))
        for r in pickle.load(open(p, 'rb')):
            r = dict(r)
            r.setdefault('root', root)
            recs.append(r)
    return recs


def sizes_hwl(lwh):
    # [UAV] GUPNet 的 size_3d 顺序是 KITTI 的 (h, w, l)，h 是第 0 维（参与 depth_geo = h3d / h2d * f）。
    # 物体系：x = l（前），y = w（横向），z = h（机体竖直轴）。
    return np.array([lwh[2], lwh[1], lwh[0]], dtype=np.float32)


def mean_size_from_records(recs):
    s = [sizes_hwl(b['lwh']) for r in recs for b in r['boxes']]
    return np.mean(np.stack(s), 0, keepdims=True).astype(np.float32)  # (1, 3) 单类


def load_resized(rec):
    """[UAV] 读图 -> 960x540 -> 下方补零到 960x544；返回 PIL 图和缩放后的 K。"""
    img = Image.open(os.path.join(rec['root'], rec['file'])).convert('RGB')
    W, H = img.size
    sx, sy = NET_W / float(W), IMG_H_VALID / float(H)
    img = img.resize((NET_W, IMG_H_VALID), Image.BILINEAR)
    canvas = Image.new('RGB', (NET_W, NET_H), (0, 0, 0))
    canvas.paste(img, (0, 0))
    K = np.asarray(rec['K'], np.float64).copy()
    K[0] *= sx
    K[1] *= sy
    return canvas, K


class UAV(data.Dataset):
    def __init__(self, label_paths, split, cfg, mean_size=None, records=None):
        # basic configuration
        self.num_classes = 1  # [UAV] 所有机型合并为 drone
        self.max_objs = 50
        self.class_name = ['drone']
        self.resolution = np.array([NET_W, NET_H])  # W * H  [UAV] 原 1280x384
        self.use_3d_center = cfg['use_3d_center']

        self.records = records if records is not None else load_records(label_paths)
        # [UAV] 类均值尺寸：单类，用 sim_train 统计（由调用方传入，否则用本数据统计）
        self.cls_mean_size = mean_size if mean_size is not None else mean_size_from_records(self.records)

        assert split in ['train', 'test']
        self.split = split
        # data augmentation configuration（官方默认）
        self.data_augmentation = True if split == 'train' else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.downsample = 4

    def get_calib(self, idx):
        return UAVCalib(self._scaled_K(idx))

    def _scaled_K(self, idx):
        rec = self.records[idx]
        K = np.asarray(rec['K'], np.float64).copy()
        K[0] *= NET_W / float(rec['W'])
        K[1] *= IMG_H_VALID / float(rec['H'])
        return K

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = item
        rec = self.records[index]
        img, K = load_resized(rec)
        img_size = np.array(img.size)

        # data augmentation for image（照抄）
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        coord_range = np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        calib = UAVCalib(K)
        features_size = self.resolution // self.downsample  # W * H
        #  ============================   get labels   ==============================
        if self.split != 'test':
            # [UAV] 由 labels.pkl 的 9D 框生成 KITTI 式对象：投影 8 角点得到 2D 框（裁到 960x540 有效区）
            objects = []
            for b in rec['boxes']:
                ypr = np.asarray(b['ypr'], np.float64) if 'ypr' in b else ypr_from_R(b['R'])  # [UAV] KITTI 式角度
                objects.append(dict(center=np.asarray(b['center'], np.float64),
                                    lwh=np.asarray(b['lwh'], np.float64), ypr=ypr))
            if random_flip_flag:
                calib.flip(img_size)
                for o in objects:
                    o['center'], o['ypr'] = flip_ypr(o['center'], o['ypr'])  # [UAV] 原：ry = pi - ry, pos[0] *= -1
            for o in objects:
                o['R'] = R_from_ypr(o['ypr'])
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32)
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 3), dtype=np.int64)    # [UAV] 1 -> 3 个角
            heading_res = np.zeros((self.max_objs, 3), dtype=np.float32)  # [UAV] 1 -> 3 个角
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
            for i in range(object_num):
                o = objects[i]
                cor = corners_of(o['center'], o['lwh'], o['R'])
                if cor[:, 2].min() <= 0.05:
                    continue
                uv, _ = calib.rect_to_img(cor.astype(np.float32))
                x1, y1 = uv.min(0)
                x2, y2 = uv.max(0)
                # [UAV] 截断率 = 投影框在 960x540 有效区外的面积比例（替代 KITTI truncation）；
                #       KITTI 的 level_str=='UnKnown' 过滤里只保留 truncation<=0.5 这一条（无 occlusion；
                #       25 px 高度门限和 pos_z<2 的过滤是 KITTI 专有的，去掉，见 NOTES）
                cx1, cy1 = max(x1, 0), max(y1, 0)
                cx2, cy2 = min(x2, img_size[0]), min(y2, IMG_H_VALID)
                area = max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)
                trunc = 1 - max(cx2 - cx1, 0) * max(cy2 - cy1, 0) / area
                if trunc > 0.5:
                    continue
                box2d_orig = np.array([cx1, cy1, cx2, cy2], dtype=np.float32)  # 裁到图内（同 KITTI 2D 标注）

                bbox_2d = box2d_orig.copy()
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                bbox_2d[:] /= self.downsample

                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)
                # [UAV] 原：pos + [0, -h/2, 0]（KITTI 底面中心 -> 几何中心）；我们的 center 已是几何中心
                center_3d = o['center'].reshape(-1, 3).astype(np.float32)
                center_3d, _ = calib.rect_to_img(center_3d)
                center_3d = center_3d[0]
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))

                cls_id = 0
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h

                depth[i] = o['center'][-1]

                # [UAV] 原：heading_angle = calib.ry2alpha(ry, box2d 中心 u)；yaw 原样，pitch/roll 绝对角（见 uav_utils）
                ang = ypr_to_code_angles(o['ypr'], (box2d_orig[0] + box2d_orig[2]) / 2, calib)
                heading_bin[i], heading_res[i] = encode_angles(ang)

                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = sizes_hwl(o['lwh'])
                size_3d[i] = src_size_3d[i] - self.cls_mean_size[cls_id]

                mask_2d[i] = 1  # [UAV] 原条件 trucation<=0.5 and occlusion<=2，已在上面等价过滤
            targets = {'depth': depth,
                       'size_2d': size_2d,
                       'heatmap': heatmap,
                       'offset_2d': offset_2d,
                       'indices': indices,
                       'size_3d': size_3d,
                       'offset_3d': offset_3d,
                       'heading_bin': heading_bin,
                       'heading_res': heading_res,
                       'cls_ids': cls_ids,
                       'mask_2d': mask_2d}
        else:
            targets = {}
        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        return inputs, calib.P2, coord_range, targets, info
