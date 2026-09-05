# -*- coding: utf-8 -*-
"""读 tools/build_mm_cache.py 打包的多模态缓存（SSD 上的 memmap）。

与 LAAM6D_Det_Dataset 的区别：
- 不再从 HDD 读 PNG/npy，训练是 GPU 瓶颈；
- 标签已是 RGB 相机 OpenCV 系的 9 参数框，旋转从角点【正确】解出（见 build_mm_cache.py）；
- 图像张量按 MODALITIES 拼通道：rgb(3) / ir(1) / depth(1) / tag(1)，顺序固定为
  rgb, ir, depth, tag 中被选中的那些。蒸馏时教师吃全部通道、学生只切前 3 个（rgb），
  所以 rgb 必须排第一，且 MODALITIES 里必须含 rgb。
- 编解码复用 MAV6D 那一对（相机系、带内参、严格互逆），畸变为 0。

配置（见 cfgs/dataset_configs/uavdet_3d/mmcache.yaml）：
    DATA_PATH:   缓存根目录，下面有 train/ test/
    MODALITIES:  ['rgb', 'ir', 'depth', 'tag'] 的子集
    DEPTH_MAX:   depth 通道归一化上限（米），超过截断到 1
"""
import copy
import os
import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..dataset import DatasetTemplate
from ...utils import frame_convention

MODAL_ORDER = ['rgb', 'ir', 'depth', 'tag']
MODAL_CH = {'rgb': 3, 'ir': 1, 'depth': 1, 'tag': 1}


class MMCache_Det_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)
        self.class_names = list(dataset_cfg.CLASS_NAMES)
        self.modalities = [m for m in MODAL_ORDER if m in list(dataset_cfg.get('MODALITIES', ['rgb']))]
        assert self.modalities and self.modalities[0] == 'rgb', 'MODALITIES 必须含 rgb 且 rgb 排第一'
        self.num_channels = sum(MODAL_CH[m] for m in self.modalities)
        self.depth_max = float(dataset_cfg.get('DEPTH_MAX', dataset_cfg.MAX_DIS))
        seq = dataset_cfg.get('EULER_SEQ', None)
        if seq:
            frame_convention.set_default_euler_seq(seq)

        self.split_dir = os.path.join(str(self.root_path), self.mode)
        with open(os.path.join(self.split_dir, 'index.pkl'), 'rb') as f:
            idx = pickle.load(f)
        self.valid_idx = idx['valid_idx']
        self.metas = idx['metas']
        self.H, self.W = idx['H'], idx['W']
        self.stride = dataset_cfg.STRIDE
        self.im_num = int(dataset_cfg.IM_NUM)          # 检测器后处理按它切图像栈
        self.class_name_dict = {i: c for i, c in enumerate(self.class_names)}
        self.new_im_width, self.new_im_hight = int(dataset_cfg.IM_RESIZE[0]), int(dataset_cfg.IM_RESIZE[1])
        assert (self.new_im_width, self.new_im_hight) == (self.W, self.H), \
            'IM_RESIZE %s 与缓存分辨率 %s 不一致' % (dataset_cfg.IM_RESIZE, (self.W, self.H))
        # RGB 可见度过滤：vis_score.npy 是每帧「最近合格目标处 RGB 局部对比度」(灰度级)。
        # mm20 有 61% 是夜间帧，夜间目标对比度中位只有 2.3-2.9（亮度 3-21/255），对 RGB 学生
        # 就是不可见的纯噪声；白天中位 8.5-15.4。按对比度筛而不按天气标签：有路灯的夜景能留下。
        min_vis = float(dataset_cfg.get('MIN_RGB_VIS', 0.0))
        if min_vis > 0:
            vp = os.path.join(self.split_dir, 'vis_score.npy')
            assert os.path.exists(vp), 'MIN_RGB_VIS 需要 %s（由 tools/mm_vis_score.py 生成）' % vp
            vis = np.load(vp)
            n0 = len(self.valid_idx)
            self.valid_idx = self.valid_idx[vis[self.valid_idx] >= min_vis]
            if self.logger is not None:
                self.logger.info('MMCache[%s]: RGB 可见度 >= %.1f 过滤 %d -> %d 帧' % (self.mode, min_vis, n0, len(self.valid_idx)))
        # 翻译缓存（tools/sim2real_bg_translate.py）：只用翻译过的帧，translated.npy 是每帧 bool
        if bool(dataset_cfg.get('REQUIRE_TRANSLATED', False)):
            tp = os.path.join(self.split_dir, 'translated.npy')
            assert os.path.exists(tp), 'REQUIRE_TRANSLATED 需要 %s（由 tools/sim2real_bg_translate.py 生成）' % tp
            tr = np.load(tp)
            n0 = len(self.valid_idx)
            self.valid_idx = self.valid_idx[tr[self.valid_idx]]
            if self.logger is not None:
                self.logger.info('MMCache[%s]: 只保留已翻译帧 %d -> %d' % (self.mode, n0, len(self.valid_idx)))
        # 教师看【原始】RGB：TEACHER_RGB_DIR 指向源缓存根目录，其 rgb.npy 作为额外 3 通道接在 image 最后，
        # 布局 [rgb(3) | ir | depth | tag | rgb_teacher(3)]；学生只取前 3 通道，CenterDetKD 给教师重排成 6 通道。
        trd = dataset_cfg.get('TEACHER_RGB_DIR', None)
        self.teacher_rgb_path = os.path.join(str(trd), self.mode, 'rgb.npy') if trd else None
        if self.teacher_rgb_path:
            assert os.path.exists(self.teacher_rgb_path), self.teacher_rgb_path
        interval = int(dataset_cfg.SAMPLED_INTERVAL[self.mode]) if 'SAMPLED_INTERVAL' in dataset_cfg else 1
        self.valid_idx = self.valid_idx[::max(interval, 1)]
        self._mm = None           # memmap 在 worker 里懒打开（spawn 后重新映射）
        # 在线增广（仓库原有的 DATA_AUGMENTOR 从未被调用过；这里是真的会执行的那份）
        self.aug = dict(dataset_cfg.get('AUG', {}) or {})
        # 按域归一化：rgb 三通道 (x/255 - mean) / std
        nm, ns = dataset_cfg.get('NORM_MEAN', None), dataset_cfg.get('NORM_STD', None)
        self.norm_mean = np.array(nm, np.float32) if nm else None
        self.norm_std = np.array(ns, np.float32) if ns else None
        if self.logger is not None:
            self.logger.info('MMCache[%s]: %d 帧, 模态 %s -> %d 通道, 缓存 %s'
                             % (self.mode, len(self.valid_idx), self.modalities, self.num_channels, self.split_dir))

    # ------------------------------------------------------------------ #
    def _open(self):
        if self._mm is None:
            self._mm = {k: np.load(os.path.join(self.split_dir, k + '.npy'), mmap_mode='r')
                        for k in ('rgb', 'ir', 'depth', 'tag')}
            if self.teacher_rgb_path:
                self._mm['rgb_t'] = np.load(self.teacher_rgb_path, mmap_mode='r')
                assert self._mm['rgb_t'].shape == self._mm['rgb'].shape, '教师 RGB 缓存与本缓存帧数/分辨率不一致'
        return self._mm

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, item):
        i = int(self.valid_idx[item])
        mm = self._open()
        meta = self.metas[i]

        chans = []
        for m in self.modalities:
            if m == 'rgb':
                chans.append(np.ascontiguousarray(mm['rgb'][i]).astype(np.float32).transpose(2, 0, 1) / 255.0)
            elif m == 'ir':
                chans.append(np.ascontiguousarray(mm['ir'][i]).astype(np.float32)[None] / 255.0)
            elif m == 'depth':
                d = np.ascontiguousarray(mm['depth'][i]).astype(np.float32) / 100.0      # 厘米 -> 米
                chans.append(np.clip(d / self.depth_max, 0.0, 1.0)[None])
            elif m == 'tag':
                chans.append(np.ascontiguousarray(mm['tag'][i]).astype(np.float32)[None])
        if self.teacher_rgb_path:
            chans.append(np.ascontiguousarray(mm['rgb_t'][i]).astype(np.float32).transpose(2, 0, 1) / 255.0)
        image = np.concatenate(chans, axis=0)            # (C, H, W)
        boxes = meta['boxes9d'].astype(np.float32).copy()
        names = list(meta['names'])
        K = np.array(meta['K_raw'], dtype=np.float64).copy()
        raw_w, raw_h = meta['raw_wh']

        if self.training and self.aug:
            image, boxes, names, K = self._augment(image, boxes, names, K, raw_w, raw_h)
        if self.norm_mean is not None:
            # 按本域自身统计量归一化 rgb 三通道（其余通道已在 [0,1]）。源域亮度中位 66、目标域 118，
            # 只除 255 的话两域第一层卷积看到的分布差很多。
            image[:3] = (image[:3] - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
            if self.teacher_rgb_path:
                image[-3:] = (image[-3:] - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
        image = image[None]                              # (1, C, H, W)

        data_dict = {
            'image': image,
            'gt_box9d': boxes,
            'gt_name': np.array(names),
            'intrinsic': np.array([K], dtype=np.float32),
            'extrinsic': np.array([np.eye(4, dtype=np.float32)]),
            'distortion': np.zeros((1, 5), dtype=np.float32),
            'raw_im_size': np.array([raw_w, raw_h]),
            'new_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'obj_size': np.array(self.dataset_cfg.get('OB_SIZE', [[1.0, 1.0, 1.0]])),
            'stride': self.stride,
            'scene_id': meta['seq'].split('/')[0],
            'seq_id': meta['seq'],
            'frame_id': meta['frame'],
        }
        data_dict = self.data_pre_processor(data_dict)
        return data_dict

    # ------------------------------------------------------------------ #
    def _augment(self, image, boxes, names, K, raw_w, raw_h):
        """在线增广（只在训练时）。几何变换同步作用于 image 各通道、内参 K 和相机系 3D 框。

        水平翻转：像素 u -> W-1-u 等价于相机系 x -> -x；3D 框中心 x 取反，
                  旋转 R -> M R M（M = diag(-1,1,1)，det 仍为 +1，是真旋转），K 的 cx -> W-1-cx。
                  注意 K 是【原始分辨率】下的，翻转在缓存分辨率上做，等价于原始分辨率翻转。
        随机尺度：图像按 s 缩放后再裁/补回 (H,W)，K 的 fx,fy,cx,cy 乘 s 并减去裁剪偏移
                  （K 以原始分辨率计，偏移要换算回原始像素）；中心出画的框丢掉。
                  这一项直接练「表观大小」的鲁棒性 —— 源域与 MAV6D 的表观尺度差 2.84 倍。
        光度：亮度/对比度/gamma/逐通道增益只作用在 rgb（以及轻微作用在 ir），depth/tag 不动。
        """
        rng = np.random
        a = self.aug
        C, H, W = image.shape

        # ---- 水平翻转 ----
        if rng.rand() < float(a.get('hflip', 0.0)):
            image = image[:, :, ::-1].copy()
            K[0, 2] = (raw_w - 1) - K[0, 2]
            if len(boxes):
                boxes[:, 0] *= -1
                M = np.diag([-1.0, 1.0, 1.0])
                for i in range(len(boxes)):
                    Rm = R.from_euler(frame_convention.get_default_euler_seq(), boxes[i, 6:9]).as_matrix()
                    boxes[i, 6:9] = R.from_matrix(M @ Rm @ M).as_euler(frame_convention.get_default_euler_seq())

        # ---- 随机尺度（缩放后裁剪/补边回原尺寸）----
        sr = a.get('scale', None)
        if sr:
            s = float(rng.uniform(sr[0], sr[1]))
            if abs(s - 1.0) > 1e-3:
                nh, nw = max(8, int(round(H * s))), max(8, int(round(W * s)))
                out = np.zeros_like(image)
                tc0 = C - 3 if self.teacher_rgb_path else C          # 教师 RGB 通道也用线性插值；depth/tag 用最近邻
                res = np.stack([cv2.resize(image[c], (nw, nh),
                                           interpolation=cv2.INTER_LINEAR if (c < 4 or c >= tc0) else cv2.INTER_NEAREST)
                                for c in range(C)], 0)
                if s >= 1.0:          # 放大后随机裁一块 (H,W)
                    oy, ox = rng.randint(0, nh - H + 1), rng.randint(0, nw - W + 1)
                    out = res[:, oy:oy + H, ox:ox + W]
                    dx, dy = -ox, -oy
                else:                 # 缩小后随机贴到画布里
                    oy, ox = rng.randint(0, H - nh + 1), rng.randint(0, W - nw + 1)
                    out[:, oy:oy + nh, ox:ox + nw] = res
                    dx, dy = ox, oy
                image = out
                # K 在原始分辨率：缩放 s 后平移 (dx,dy) 个缓存像素 = (dx*raw_w/W, dy*raw_h/H) 个原始像素
                K[0, 0] *= s; K[1, 1] *= s
                K[0, 2] = K[0, 2] * s + dx * raw_w / W
                K[1, 2] = K[1, 2] * s + dy * raw_h / H
                if len(boxes):
                    uv = (K @ boxes[:, :3].T.astype(np.float64)).T
                    uv = uv[:, :2] / uv[:, 2:3]
                    keep = (uv[:, 0] >= 0) & (uv[:, 0] < raw_w) & (uv[:, 1] >= 0) & (uv[:, 1] < raw_h)
                    boxes, names = boxes[keep], [n for n, k in zip(names, keep) if k]

        # ---- 光度（只动 rgb；ir 轻微）----
        if a.get('photometric', False):
            rgb = image[:3]
            gain = rng.uniform(0.7, 1.3)                       # 亮度
            contrast = rng.uniform(0.7, 1.3)
            gamma = rng.uniform(0.7, 1.4)
            cgain = rng.uniform(0.9, 1.1, size=(3, 1, 1))     # 逐通道
            m = rgb.mean()
            rgb = np.clip(((rgb - m) * contrast + m) * gain * cgain, 0, 1) ** gamma
            if a.get('noise', 0.0) > 0:
                rgb = np.clip(rgb + rng.randn(*rgb.shape).astype(np.float32) * float(a['noise']), 0, 1)
            image[:3] = rgb
            if C > 3 and 'ir' in self.modalities:
                image[3] = np.clip(image[3] * rng.uniform(0.85, 1.15), 0, 1)
        return image.astype(np.float32), boxes, names, K

    def generate_prediction_dicts(self, batch_dict, output_path=None):
        annos = []
        for b in range(batch_dict['batch_size']):
            pred = batch_dict['pred_boxes9d'][b]
            conf = batch_dict['confidence'][b]
            gt = batch_dict['gt_box9d'][b]
            gt = gt.cpu().numpy() if hasattr(gt, 'cpu') else np.asarray(gt)
            gt = gt[np.abs(gt).sum(1) > 0]            # 去掉 collate 补的零行
            annos.append({'pred_boxes9d': np.asarray(pred), 'confidence': np.asarray(conf),
                          'gt_box9d': gt, 'seq_id': batch_dict['seq_id'][b],
                          'frame_id': batch_dict['frame_id'][b]})
        return annos

    def evaluation(self, annos, metric_root_path=None, **kwargs):
        """类别无关：每个 GT 匹配最近的预测（<2 m），报位置/角度误差与召回。

        签名与 eval_utils.eval_one_epoch 的调用 dataset.evaluation(det_annos, result_dir) 一致，
        只返回字符串；指标同时写到 result_dir/metrics.json。
        """
        from scipy.spatial.transform import Rotation as R
        seq = frame_convention.get_default_euler_seq()
        pos, ang, n_gt, n_hit = [], [], 0, 0
        for a in annos:
            gt, pr = a['gt_box9d'], a['pred_boxes9d']
            n_gt += len(gt)
            if len(pr) == 0:
                continue
            for g in gt:
                d = np.linalg.norm(pr[:, :3] - g[:3], axis=1)
                k = int(np.argmin(d))
                if d[k] > 2.0:
                    continue
                n_hit += 1
                pos.append(d[k])
                ang.append(np.degrees((R.from_euler(seq, pr[k, 6:9]).inv() * R.from_euler(seq, g[6:9])).magnitude()))
        pos, ang = np.array(pos), np.array(ang)
        res = {'recall@2m': n_hit / max(n_gt, 1),
               'pos_median': float(np.median(pos)) if len(pos) else -1,
               'ang_median': float(np.median(ang)) if len(ang) else -1,
               'n_gt': n_gt}
        s = '\n'.join('  %-12s %.4f' % (k, v) for k, v in res.items())
        if metric_root_path is not None:
            import json
            os.makedirs(str(metric_root_path), exist_ok=True)
            with open(os.path.join(str(metric_root_path), 'metrics.json'), 'w') as f:
                json.dump(res, f, indent=2)
        return s
