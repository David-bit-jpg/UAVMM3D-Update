# -*- coding: utf-8 -*-
"""几何辅助学生（2026-10-04，用户问题「近处 LiDAR 点云带几何信息能不能帮朝向」）：
在 CenterDetMT（模态幻觉头）之上再加一个【目标相对深度】辅助头，只在源域训练时用，部署时丢掉（权重名 aux_heads.rdepth.*，
部署用的 CenterDet 载入时按「ckpt 有、模型没有」忽略）。

和 MT 的 depth 头的区别：MT 的 depth 头回归绝对米制深度（PS-D：RGB 读不出，抹掉 LiDAR 后可读性 0.78→−0.10）；
这里只回归「这个像素比目标中心远 / 近多少个机身尺寸」：
    t(p) = clip( (Z_lidar(p) − Z_c) / ||lwh||, −c, c ),   只在 tag==1（LiDAR 打在无人机上）且有回波的格子上算 L1
它是目标的 3D 形状在相机方向上的投影（哪边离相机近），直接编码朝向，与离相机多远无关 —— 原理上 RGB 的明暗、透视、
遮挡关系就带着这部分信息（与「绝对深度 RGB 读不出」不同），属于「把另一模态的几何当答案、不当特征」。
每个格子归给图像平面上投影最近的 GT 中心（KZ 裁窗里通常只有一个目标）。

配置（MODEL.AUX，0 = 关）：W_RDEPTH: 2.0, RDEPTH_CLIP: 1.0；其余键与 CenterDetMT 相同（W_IR / W_DEPTH / W_TAG 可同时开）。
数据：MODALITIES 必须含 depth 和 tag（camnorm_sim_full_kp_t_zoom_mm.yaml）。DISTILL 权重全 0 时不建教师。
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .center_det_kd import CenterDetKD
from .center_det_mt import CenterDetMT


class CenterDetGeo(CenterDetMT):
    def __init__(self, model_cfg, dataset):
        # 随机数对齐（2026-10-05 训练前修正，见 PREREG_kzaux.md 修正段）：基础网络一律按「所有辅助 / 蒸馏权重为 0」构建，
        # 消耗的全局随机数与对照臂 C0 完全相同；额外的辅助头和教师在 fork_rng 里建，不动全局 RNG。
        # train.py 先 build_dataloader 再 build_network，worker 种子在创建迭代器时从主进程 RNG 抽 —— 这样各臂的
        # 数据顺序与增广流逐位相同，配对比较才真正「同像素」。
        base = copy.deepcopy(model_cfg)
        if base.get('AUX', None) is not None:
            for k in ('W_IR', 'W_DEPTH', 'W_TAG', 'W_RDEPTH'):
                base.AUX[k] = 0.0
        if base.get('DISTILL', None) is not None:
            for k in ('W_FEAT', 'W_CENTER_COS', 'W_HM', 'W_REG', 'W_CROSS'):
                base.DISTILL[k] = 0.0
        super().__init__(model_cfg=base, dataset=dataset)
        self.model_cfg = model_cfg
        a = model_cfg.get('AUX', None) or {}
        d = model_cfg.get('DISTILL', None) or {}
        self.aux_w = {k: float(a.get(k, 0.0)) for k in ('W_IR', 'W_DEPTH', 'W_TAG', 'W_RDEPTH')}
        self.kd_w = {k: float(d.get(k, 0.0)) for k in ('W_FEAT', 'W_CENTER_COS', 'W_HM', 'W_REG', 'W_CROSS')}
        self.tag_pos_weight = float(a.get('TAG_POS_WEIGHT', 10.0))
        self.rd_clip = float(a.get('RDEPTH_CLIP', 1.0))
        dc = dataset.dataset_cfg
        self.depth_max = float(dc.get('DEPTH_MAX', dc.get('MAX_DIS', 1.0)))
        in_c = int(model_cfg.DENSE_HEAD_2D.INPUT_CHANNELS)
        mid = int(a.get('CONV_DIM', 128))
        with torch.random.fork_rng(devices=[]):
            for wk, key in (('W_IR', 'ir'), ('W_DEPTH', 'depth'), ('W_TAG', 'tag'), ('W_RDEPTH', 'rdepth')):
                if self.aux_w[wk] <= 0:
                    continue
                need = ('depth', 'tag') if key == 'rdepth' else (key,)
                for m in need:
                    assert m in self.aux_ch, 'AUX.%s 需要 DATA_CONFIG.MODALITIES 里含 %s' % (wk, m)
                self.aux_heads[key] = nn.Sequential(
                    nn.Conv2d(in_c, mid, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(mid), nn.ReLU(),
                    nn.Conv2d(mid, 1, kernel_size=3, stride=1, padding=1))
            if any(v > 0 for v in self.kd_w.values()):
                self._build_teacher(d)

    def rdepth_target(self, img, gt, Kin, hw):
        """img (N,C,H,W)；gt (N,G,9) 相机系框（补零行）；Kin (N,3,3) 输入分辨率内参。返回目标 (N,1,h,w) 与掩码。"""
        h, w = hw
        N, _, H, W = img.shape
        d0, _ = self.aux_ch['depth']
        t0, _ = self.aux_ch['tag']
        dm = img[:, d0:d0 + 1] * self.depth_max
        valid = ((img[:, t0:t0 + 1] > 0.5) & (img[:, d0:d0 + 1] > 0)).float()
        cnt = F.adaptive_avg_pool2d(valid, (h, w))
        dcell = F.adaptive_avg_pool2d(dm * valid, (h, w)) / cnt.clamp_min(1e-6)
        sy, sx = H / float(h), W / float(w)
        ys = (torch.arange(h, device=img.device, dtype=img.dtype) + 0.5) * sy
        xs = (torch.arange(w, device=img.device, dtype=img.dtype) + 0.5) * sx
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        tgt = torch.zeros_like(dcell)
        msk = cnt > 0
        for n in range(N):
            g = gt[n]
            g = g[g.abs().sum(1) > 0]
            if len(g) == 0:
                msk[n] = False
                continue
            c = g[:, :3]
            uv = (Kin[n] @ c.t()).t()
            uv = uv[:, :2] / uv[:, 2:3].clamp_min(1e-6)
            dd = (gx[None] - uv[:, 0, None, None]) ** 2 + (gy[None] - uv[:, 1, None, None]) ** 2
            k = dd.argmin(0)                                    # (h,w) 每格归给最近的 GT 中心
            zc = c[:, 2][k]
            diag = g[:, 3:6].norm(dim=1)[k].clamp_min(1e-3)
            tgt[n, 0] = ((dcell[n, 0] - zc) / diag).clamp(-self.rd_clip, self.rd_clip)
        return tgt, msk

    def forward(self, batch_dict):
        ret = CenterDetKD.forward(self, batch_dict)
        if not self.training or not self.aux_heads:
            return ret
        feat = self._student_out['features_2d']                 # (B,K,C,h,w)
        B, K, C, h, w = feat.shape
        x = feat.reshape(B * K, C, h, w)
        img = batch_dict['image']
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        tg = self.aux_targets(img, (h, w))
        if 'rdepth' in self.aux_heads:
            gt = torch.as_tensor(np.asarray(batch_dict['gt_box9d']) if not torch.is_tensor(batch_dict['gt_box9d']) else batch_dict['gt_box9d'],
                                 dtype=torch.float32, device=img.device).reshape(img.shape[0], -1, 9)
            Kin = torch.as_tensor(np.asarray(batch_dict['intrinsic']) if not torch.is_tensor(batch_dict['intrinsic']) else batch_dict['intrinsic'],
                                  dtype=torch.float32, device=img.device).reshape(img.shape[0], -1, 3, 3)[:, 0]
            tg['rdepth'] = self.rdepth_target(img, gt, Kin, (h, w))
        terms = self.dense_head_2d.loss_terms
        loss = ret['loss']
        for key, head in self.aux_heads.items():
            pred = head(x)
            t, m = tg[key]
            if key == 'tag':
                pw = torch.tensor(self.tag_pos_weight, device=pred.device)
                l = F.binary_cross_entropy_with_logits(pred, t, pos_weight=pw)
            elif m is not None:
                l = torch.abs(pred - t)[m].mean() if bool(m.any()) else (pred * 0).sum()
            else:
                l = torch.abs(pred - t).mean()
            terms['aux_' + key] = float(l.detach())
            loss = loss + self.aux_w['W_' + key.upper()] * l
        self.dense_head_2d.loss_terms = terms
        return {'loss': loss}
