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

import numpy as np

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
        interval = int(dataset_cfg.SAMPLED_INTERVAL[self.mode]) if 'SAMPLED_INTERVAL' in dataset_cfg else 1
        self.valid_idx = self.valid_idx[::max(interval, 1)]
        self._mm = None           # memmap 在 worker 里懒打开（spawn 后重新映射）
        if self.logger is not None:
            self.logger.info('MMCache[%s]: %d 帧, 模态 %s -> %d 通道, 缓存 %s'
                             % (self.mode, len(self.valid_idx), self.modalities, self.num_channels, self.split_dir))

    # ------------------------------------------------------------------ #
    def _open(self):
        if self._mm is None:
            self._mm = {k: np.load(os.path.join(self.split_dir, k + '.npy'), mmap_mode='r')
                        for k in ('rgb', 'ir', 'depth', 'tag')}
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
        image = np.concatenate(chans, axis=0)[None]      # (1, C, H, W)

        raw_w, raw_h = meta['raw_wh']
        data_dict = {
            'image': image,
            'gt_box9d': meta['boxes9d'].astype(np.float32).copy(),
            'gt_name': np.array(meta['names']),
            'intrinsic': np.array([meta['K_raw']], dtype=np.float32),
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
