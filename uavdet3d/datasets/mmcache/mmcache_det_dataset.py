# -*- coding: utf-8 -*-
"""读 tools/build_mm_cache.py 打包的多模态缓存（SSD 上的 memmap）。

与 LAAM6D_Det_Dataset 的区别：
- 不再从 HDD 读 PNG/npy，训练是 GPU 瓶颈；
- 标签已是 RGB 相机 OpenCV 系的 9 参数框，旋转从角点【正确】解出（见 build_mm_cache.py）；
- 图像张量按 MODALITIES 拼通道：rgb(3) / ir(1) / depth(1) / tag(1)，顺序固定为
  rgb, ir, depth, tag 中被选中的那些。蒸馏时教师吃全部通道、学生只切前 3 个（rgb），
  所以 rgb 必须排第一，且 MODALITIES 里必须含 rgb。
- 编解码复用 MAV6D 那一对（相机系、带内参、严格互逆），畸变为 0。
- 内参一律换算到【缓存分辨率】再用（meta 里有 K_in 就直接用；旧缓存只有 K_raw 时按 cv2.resize 的
  像素中心约定换算）。翻转 / 缩放增广都在同一个分辨率上用精确式（camera_geometry.py），
  编码器拿到的 raw_im_size == new_im_size。任何相机、任何原始分辨率走同一条路。
- 欧拉顺序显式来自配置 EULER_SEQ（缺省 'xyz'，与 build_mm_cache / build_mav6d_cache 的存储一致），
  不读模块全局 —— spawn 出来的 worker 里模块全局是 'zyx'（审查 P14：翻转后一半样本旋转标签被打乱）。

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
from ...utils import camera_geometry as cg
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
        # 缓存里的角一律按 'xyz' 存（build_mm_cache / build_mav6d_cache）。增广和评测都用这一个值
        self.euler_seq = str(dataset_cfg.get('EULER_SEQ', 'xyz'))
        # 物体的镜像对称面（翻转增广的旋转标签用）：机体系 x 前 / y 左 / z 上的无人机左右对称 -> 'y'
        self.flip_body_mirror = str(dataset_cfg.get('FLIP_BODY_MIRROR_AXIS', 'y'))
        frame_convention.set_default_euler_seq(self.euler_seq)      # 只影响主进程里的旧度量函数
        # 不分机型：HM_CLASS_NAMES 只有一个类时，所有标签名映射成它
        hm_names = dataset_cfg.get('HM_CLASS_NAMES', None)
        self.agnostic_name = hm_names[0] if hm_names and len(hm_names) == 1 else None

        # 读哪个子目录：DATA_SPLIT[mode]（缺省就是 train / test）。评验证集时把 DATA_SPLIT.test 设成 val
        split_name = dataset_cfg.DATA_SPLIT[self.mode] if 'DATA_SPLIT' in dataset_cfg else self.mode
        self.split_dir = os.path.join(str(self.root_path), str(split_name))
        # INDEX_FILE：同一份图像缓存可以配不同的标签索引（tools/fix_sim_labels.py 写的 index_nose.pkl /
        # index_nosescale.pkl = 仿真资产机头偏航 / 物理尺度修正，见 uavdet3d/utils/sim_asset_fix.py）
        self.index_file = str(dataset_cfg.get('INDEX_FILE', 'index.pkl'))
        with open(os.path.join(self.split_dir, self.index_file), 'rb') as f:
            idx = pickle.load(f)
        self.label_fix = idx.get('label_fix', None)
        if self.logger is not None:
            self.logger.info('MMCache[%s]: 标签索引 %s（label_fix=%s）' % (
                self.mode, self.index_file, None if self.label_fix is None else
                {k: self.label_fix[k] for k in ('version', 'nose', 'scale')}))
        self.valid_idx = idx['valid_idx']
        self.metas = idx['metas']
        self.H, self.W = idx['H'], idx['W']
        self.stride = dataset_cfg.STRIDE
        self.im_num = int(dataset_cfg.IM_NUM)          # 检测器后处理按它切图像栈
        self.class_name_dict = {i: c for i, c in enumerate(self.class_names)}
        self.new_im_width, self.new_im_hight = int(dataset_cfg.IM_RESIZE[0]), int(dataset_cfg.IM_RESIZE[1])
        # 存储格式：npy = 已缩到 IM_RESIZE 的 memmap；jpeg = 原分辨率 JPEG 字节（build_mm_cache --store jpeg），
        # 每次取样在原图上裁一个与输入同宽高比的窗口再缩放成网络输入 —— 等效换一台焦距放大 z 倍的相机（_view）
        self.store = str(idx.get('store', 'npy'))
        self.src_H, self.src_W = self.H, self.W
        if self.store == 'jpeg':
            assert self.modalities == ['rgb'] and not dataset_cfg.get('TEACHER_RGB_DIR', None), 'jpeg 存储只支持纯 RGB'
            self.H, self.W = self.new_im_hight, self.new_im_width
        else:
            assert (self.new_im_width, self.new_im_hight) == (self.W, self.H), \
                'IM_RESIZE %s 与缓存分辨率 %s 不一致' % (dataset_cfg.IM_RESIZE, (self.W, self.H))
        # 训练时焦距放大倍数 z 在 VIEW_AUG.zoom 区间内按对数均匀抽（z=1 = 整幅缩放）；验证/测试按 VAL_ZOOMS 轮流取，
        # 窗口以第一个目标为中心，结果确定。只对 jpeg 存储生效。
        self.view_zoom = [float(v) for v in (dataset_cfg.get('VIEW_AUG', {}) or {}).get('zoom', [1.0, 1.0])]
        self.val_zooms = [float(v) for v in dataset_cfg.get('VAL_ZOOMS', [1.0])]
        # 只降采样不放大（见 _view 的说明）。缺省 False = 与 2026-09-16 之前逐位一致
        self.no_upscale = bool((dataset_cfg.get('VIEW_AUG', {}) or {}).get('no_upscale', False))
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
        # 小数据量档把训练帧重复 TRAIN_REPEAT 遍拼成一轮（每遍增广不同）：129 帧一轮只有 17 个 iteration，
        # 每轮 worker 重启 + 存 checkpoint 的固定开销（实测 ~17 s）会比训练本身还长。只影响训练，不改变用到哪些帧
        repeat = int(dataset_cfg.get('TRAIN_REPEAT', 1)) if self.training else 1
        if repeat > 1:
            self.valid_idx = np.tile(self.valid_idx, repeat)
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
        if self._mm is None and self.store == 'jpeg':
            self._mm = {'rgb_jpg': np.memmap(os.path.join(self.split_dir, 'rgb_jpg.bin'), dtype=np.uint8, mode='r'),
                        'rgb_jpg_index': np.load(os.path.join(self.split_dir, 'rgb_jpg_index.npy'))}
        if self._mm is None:
            # 只打开用得到的模态（RGB-only 缓存里没有 ir/depth/tag 文件）
            self._mm = {k: np.load(os.path.join(self.split_dir, k + '.npy'), mmap_mode='r')
                        for k in self.modalities}
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
        for m in (self.modalities if self.store != 'jpeg' else []):
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
        boxes = meta['boxes9d'].astype(np.float64).copy()
        names = list(meta['names'])
        if self.agnostic_name is not None:
            names = [self.agnostic_name] * len(names)
        K = self.cache_intrinsic(meta)                    # 缓存分辨率上的针孔内参（jpeg 存储时是原图分辨率）
        if self.store == 'jpeg':
            o, ln = mm['rgb_jpg_index'][i]
            src = cv2.imdecode(np.asarray(mm['rgb_jpg'][int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
            view, K, boxes, names = self._view(src, K, boxes, names, item)
            chans.append(view.astype(np.float32).transpose(2, 0, 1) / 255.0)
        image = np.concatenate(chans, axis=0)            # (C, H, W)

        if self.training and self.aug:
            image, boxes, names, K = self._augment(image, boxes, names, K)
        if self.norm_mean is not None:
            # 按本域自身统计量归一化 rgb 三通道（其余通道已在 [0,1]）。源域亮度中位 66、目标域 118，
            # 只除 255 的话两域第一层卷积看到的分布差很多。
            image[:3] = (image[:3] - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
            if self.teacher_rgb_path:
                image[-3:] = (image[-3:] - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
        image = image[None]                              # (1, C, H, W)

        data_dict = {
            'image': image,
            'gt_box9d': boxes.reshape(-1, 9).astype(np.float32),
            'gt_name': np.array(names),
            'intrinsic': np.array([K], dtype=np.float64),
            'extrinsic': np.array([np.eye(4, dtype=np.float32)]),
            'distortion': np.zeros((1, 5), dtype=np.float32),
            'raw_im_size': np.array([self.new_im_width, self.new_im_hight]),   # K 已在输入分辨率
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
    def cache_intrinsic(self, meta):
        """meta -> 缓存分辨率 (W,H) 上的内参。新缓存直接存 K_in；旧缓存存原图 K_raw + raw_wh。"""
        if 'K_in' in meta:
            return np.array(meta['K_in'], dtype=np.float64).reshape(3, 3).copy()
        raw_w, raw_h = meta['raw_wh']
        return cg.scale_K(meta['K_raw'], self.W / float(raw_w), self.H / float(raw_h))

    def _view(self, img, K, boxes, names, item):
        """原分辨率帧 -> 网络输入 (W,H)：裁一个与输入同宽高比、宽 = 原宽/z 的窗口再缩放。

        等效于换一台焦距放大 z 倍、视场收窄的相机：K 先平移 (-x0,-y0) 再按 W/w 缩放（cv2 像素中心约定，精确），
        3D 框不动；虚拟深度 Zv = Z*f_ref/f_in 随 z 缩小 —— 仿真无人机因此也会出现在 MAV6D 那样「近、大」的视角里。
        （实测：只训整幅缩放 z=1 时，仿真验证帧裁成 z=2.5 的窗口，深度就系统偏大 2.1 倍，外推不了。）
        训练：z 在 VIEW_AUG.zoom 内对数均匀抽；窗口随机放，但保证随机选中的一个目标中心在窗口内（留 8 输入像素边）。
        验证 / 测试：z 按 VAL_ZOOMS 轮流取，窗口以第一个目标为中心（夹到图内），结果确定。
        中心出窗口的框丢掉。

        VIEW_AUG.no_upscale（2026-09-16 起默认开）：把窗口夹到【不小于网络输入】，即只降采样不放大。
        为什么：窗口小于输入时 cv2 只能插值放大，凭空造不出细节。实测仿真在 MAV6D 的工作点上要放大 1.30 倍，
        目标块细节量掉到真实的 1/16（docs/audit/ANNOTATION_CONSISTENCY_2026-09-15.md §12.2），
        而深度头恰好靠这个细节量判距离 —— 把真实图糊到同等水平，深度比就从 2.69 回到 1.13。
        这条规则只用【本域自己的源分辨率】决定，换任何相机都自动成立，不是对着某个测试集调的。
        """
        sH, sW = img.shape[:2]
        oW, oH = self.new_im_width, self.new_im_hight
        if self.training:
            z0, z1 = self.view_zoom
            z = float(np.exp(np.random.uniform(np.log(z0), np.log(z1)))) if z1 > z0 else z0
        else:
            z = self.val_zooms[item % len(self.val_zooms)]
        # 窗口宽高与输入严格同比例（输入 16:9 -> 宽取 16 的倍数）
        unit = int(np.gcd(oW, oH))
        uw, uh = oW // unit, oH // unit
        k_min = unit if self.no_upscale else 1          # unit = oW // uw，即窗口宽 >= 输入宽
        k = max(k_min, min(int(round(sW / z / uw)), sW // uw, sH // uh))
        w, h = k * uw, k * uh
        uv = None
        cand = []
        if len(boxes):
            pr = (K @ boxes[:, :3].T).T
            uv = pr[:, :2] / pr[:, 2:3]
            cand = [j for j in range(len(boxes)) if boxes[j, 2] > 0 and 0 <= uv[j, 0] < sW and 0 <= uv[j, 1] < sH]
        m = 8.0 * w / oW
        if cand:
            j = cand[np.random.randint(len(cand))] if self.training else cand[0]
            cu, cv_ = uv[j]
            lo_x, hi_x = max(0.0, cu - w + m), min(float(sW - w), cu - m)
            lo_y, hi_y = max(0.0, cv_ - h + m), min(float(sH - h), cv_ - m)
            if self.training and lo_x <= hi_x:
                x0 = np.random.uniform(lo_x, hi_x)
            else:
                x0 = np.clip(cu - w / 2.0, 0, sW - w)
            if self.training and lo_y <= hi_y:
                y0 = np.random.uniform(lo_y, hi_y)
            else:
                y0 = np.clip(cv_ - h / 2.0, 0, sH - h)
        else:
            x0 = np.random.uniform(0, sW - w) if self.training else (sW - w) / 2.0
            y0 = np.random.uniform(0, sH - h) if self.training else (sH - h) / 2.0
        x0, y0 = int(round(x0)), int(round(y0))
        win = img[y0:y0 + h, x0:x0 + w]
        if (w, h) != (oW, oH):
            win = cv2.resize(win, (oW, oH), interpolation=cv2.INTER_AREA if w > oW else cv2.INTER_LINEAR)
        K2 = cg.scale_K(cg.translate_K(K, -x0, -y0), oW / float(w), oH / float(h))
        if len(boxes):
            pr = (K2 @ boxes[:, :3].T).T
            uv2 = pr[:, :2] / pr[:, 2:3]
            keep = (boxes[:, 2] > 0) & (uv2[:, 0] >= 0) & (uv2[:, 0] < oW) & (uv2[:, 1] >= 0) & (uv2[:, 1] < oH)
            boxes, names = boxes[keep], [n for n, kk in zip(names, keep) if kk]
        return np.ascontiguousarray(win), K2, boxes, names

    def _augment(self, image, boxes, names, K):
        """在线增广（只在训练时）。几何变换同步作用于 image 各通道、内参 K（缓存分辨率）和相机系 3D 框。

        水平翻转：像素 u -> W-1-u 等价于相机系 x -> -x；框中心 x 取反，K 按 camera_geometry.hflip_K，
                  旋转 R -> M R S（camera_geometry.hflip_rotation）：M = diag(-1,1,1) 是相机系镜像，
                  S 是【物体自身的镜像对称面】—— 无人机左右对称、前后不对称（云台在前），S = 机体 y 取反。
                  注意不能用 M R M：那等于把机头机尾对调（2026-09-14 实测翻转后 x 轴指向机尾，
                  一半样本朝向标签错 180°，MAV6D 从零训练角度误差中位 93~113°）。
                  欧拉角用 self.euler_seq 显式转换（审查 P14）。
        随机尺度：图像缩放到 (round(W*s), round(H*s)) 后裁 / 补回 (W,H)。K 按实际缩放比 nw/W、nh/H
                  和 cv2.resize 的像素中心约定精确换算，再加裁剪偏移。3D 框不动 ——
                  虚拟深度模式下编码器按新焦距自动把 Zv 除以 s，「看起来更近」和标签一致（审查 P8）。
                  中心出画的框丢掉。
        光度：亮度/对比度/gamma/逐通道增益只作用在 rgb（以及轻微作用在 ir），depth/tag 不动。
        """
        rng = np.random
        a = self.aug
        C, H, W = image.shape

        # ---- 水平翻转 ----
        if rng.rand() < float(a.get('hflip', 0.0)):
            image = image[:, :, ::-1].copy()
            K = cg.hflip_K(K, W)
            if len(boxes):
                boxes[:, 0] *= -1
                Rm = R.from_euler(self.euler_seq, boxes[:, 6:9]).as_matrix()
                boxes[:, 6:9] = R.from_matrix(cg.hflip_rotation(Rm, self.flip_body_mirror)).as_euler(self.euler_seq)

        # ---- 随机尺度（缩放后裁剪/补边回原尺寸）----
        sr = a.get('scale', None)
        if sr:
            s = float(rng.uniform(sr[0], sr[1]))
            nh, nw = max(8, int(round(H * s))), max(8, int(round(W * s)))
            out = None
            if (nh, nw) != (H, W):
                tc0 = C - 3 if self.teacher_rgb_path else C          # 教师 RGB 通道也用线性插值；depth/tag 用最近邻
                res = np.stack([cv2.resize(image[c], (nw, nh),
                                           interpolation=cv2.INTER_LINEAR if (c < 4 or c >= tc0) else cv2.INTER_NEAREST)
                                for c in range(C)], 0)
                if nh >= H and nw >= W:       # 放大后随机裁一块 (H,W)
                    oy, ox = rng.randint(0, nh - H + 1), rng.randint(0, nw - W + 1)
                    out = res[:, oy:oy + H, ox:ox + W]
                    dx, dy = -ox, -oy
                elif nh <= H and nw <= W:     # 缩小后随机贴到画布里
                    out = np.zeros_like(image)
                    oy, ox = rng.randint(0, H - nh + 1), rng.randint(0, W - nw + 1)
                    out[:, oy:oy + nh, ox:ox + nw] = res
                    dx, dy = ox, oy
            if out is not None:
                image = np.ascontiguousarray(out)
                K = cg.translate_K(cg.scale_K(K, nw / float(W), nh / float(H)), dx, dy)
                if len(boxes):
                    uv = (K @ boxes[:, :3].T).T
                    uv = uv[:, :2] / uv[:, 2:3]
                    keep = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                    boxes, names = boxes[keep], [n for n, k in zip(names, keep) if k]

        # ---- 带宽（细节量）增广：随机把 rgb 压到更低的有效分辨率再放回来，几何完全不动 ----
        # 为什么：两域唯一没被任何增广覆盖的轴就是「细节量」，而它恰好和裁窗倍数强相关
        # （裁得越狠越糊），于是网络把细节量当成了距离线索 —— 真实图一换就崩（§12.2）。
        # 这里让细节量独立于 z 随机，切断这个伪相关；只会变糊不会变锐，所以是安全方向。
        bmax = float(a.get('blur', 0.0) or 0.0)
        if bmax > 1.0 and rng.rand() < float(a.get('blur_prob', 0.5)):
            d = float(np.exp(rng.uniform(0.0, np.log(bmax))))
            nh2, nw2 = max(8, int(round(H / d))), max(8, int(round(W / d)))
            nc = min(3, C)
            small = [cv2.resize(image[c], (nw2, nh2), interpolation=cv2.INTER_AREA) for c in range(nc)]
            image[:nc] = np.stack([cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR) for s in small], 0)

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
        seq = self.euler_seq
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
