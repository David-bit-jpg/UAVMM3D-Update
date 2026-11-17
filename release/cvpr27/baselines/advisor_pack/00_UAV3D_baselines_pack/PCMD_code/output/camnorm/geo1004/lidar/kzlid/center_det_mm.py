# -*- coding: utf-8 -*-
"""CenterDetMM（10-12 夜，用户：「朝向里也把 IR 和 LiDAR 一起加上试试，位置也试试」）：网络只看 RGB，训练时多一组辅助答案，IR 与 LiDAR 一起用。
数据集吐 [rgb(3) | ir | depth | tag]（多模态缓存，同一裁窗 / 几何增广，depth/tag 最近邻），学生只切前 3 通道（CenterDetKD 学生路径，无教师）。
辅助头接在 features_2d（与检测 / 关键点头同一层特征）上，每格预测 3 个量：
    通道 0  有没有打到无人机的 LiDAR 点（BCE）                                     ← LiDAR
    通道 1  点深度相对该机中心 / 机体最大边（有点格 L1）= 三维形状、哪边近哪边远        ← LiDAR
    通道 2  红外剪影：LiDAR 标出的机身区域里，红外亮度按该区域最暗–最亮归一（区域内 L1）  ← IR（区域由 LiDAR 给）
LiDAR 目标与 CenterDetLiD 相同（打点最多的真值机、中心深度带滤背景点）。LIDAR_AUX / IR_AUX 各自的 WEIGHT 设 0 即关。
部署时辅助头不用，存的权重多出 aux_head.*（导出时去掉）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from uavdet3d.model.detectors.center_det_kd import CenterDetKD


class CenterDetMM(CenterDetKD):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        La = model_cfg.get('LIDAR_AUX', None) or {}
        Ia = model_cfg.get('IR_AUX', None) or {}
        self.w_lid = float(La.get('WEIGHT', 1.0))
        self.w_ir = float(Ia.get('WEIGHT', 1.0))
        self.ch_d = int(La.get('DEPTH_CH', 4))
        self.ch_t = int(La.get('TAG_CH', 5))
        self.ch_ir = int(Ia.get('IR_CH', 3))
        self.depth_max = float(dataset.dataset_cfg.get('DEPTH_MAX', 24.0))
        C = int(model_cfg.DENSE_HEAD_2D.INPUT_CHANNELS)
        self.aux_head = nn.Sequential(nn.Conv2d(C, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 3, 1))

    @torch.no_grad()
    def targets(self, batch_dict, h, w):
        img = batch_dict['image']
        B, K, _, H, W = img.shape
        d = img[:, :, self.ch_d].reshape(B * K, 1, H, W).float() * self.depth_max
        tag = (img[:, :, self.ch_t] > 0.5).reshape(B * K, 1, H, W).float()
        ir = img[:, :, self.ch_ir].reshape(B * K, 1, H, W).float()
        hit = tag * (d > 0.05).float()
        gt = batch_dict.get('gt_box9d', None)
        zc = torch.zeros(B * K, device=img.device)
        Ls = torch.ones(B * K, device=img.device)
        ok = torch.zeros(B * K, dtype=torch.bool, device=img.device)
        keep = torch.zeros_like(hit)
        for n in range(B * K):
            if gt is None:
                break
            g = gt[n // K]
            g = g[g.abs().sum(-1) > 0]
            hn = hit[n, 0] > 0
            if len(g) == 0 or int(hn.sum()) < 5:
                continue
            dn = d[n, 0]
            v = dn[hn]
            best = (0, -1, None)
            for k in range(len(g)):
                L = g[k, 3:6].max().clamp(min=0.05)
                c = int(((v - g[k, 2]).abs() < 0.75 * L + 0.3).sum())
                if c > best[0]:
                    best = (c, k, L)
            if best[0] < 5:
                continue
            k, L = best[1], best[2]
            sel = hn & ((dn - g[k, 2]).abs() < 0.75 * L + 0.3)
            keep[n, 0] = sel.float()
            zc[n] = dn[sel].median()
            Ls[n] = L
            ok[n] = True
        sy, sx = H // h, W // w
        m = F.max_pool2d(keep, (sy, sx))[:, 0, :h, :w] * ok[:, None, None].float()
        cnt = F.avg_pool2d(keep, (sy, sx))
        dm = F.avg_pool2d(d * keep, (sy, sx)) / cnt.clamp(min=1e-6)
        rel = ((dm - zc[:, None, None, None]) / Ls[:, None, None, None]).clamp(-1.5, 1.5)[:, 0, :h, :w]
        # 红外剪影：LiDAR 标签（外扩过的机身区域）里，红外按该区域最暗–最亮归一
        reg = F.max_pool2d(tag, (sy, sx))[:, 0, :h, :w]
        irp = F.avg_pool2d(ir, (sy, sx))[:, 0, :h, :w]
        big = torch.tensor(1e6, device=img.device)
        lo = torch.where(reg > 0, irp, big).flatten(1).min(1).values
        hi = torch.where(reg > 0, irp, -big).flatten(1).max(1).values
        valid = (reg.flatten(1).sum(1) >= 4) & ((hi - lo) > 0.03)
        irt = ((irp - lo[:, None, None]) / (hi - lo).clamp(min=1e-3)[:, None, None]).clamp(0, 1)
        reg = reg * valid[:, None, None].float()
        return m, rel, reg, irt

    def forward(self, batch_dict):
        out = super().forward(batch_dict)
        if not self.training:
            return out
        f = self._student_out['features_2d']                  # (B,K,C,h,w)
        B, K, C, h, w = f.shape
        p = self.aux_head(f.reshape(B * K, C, h, w)).float()
        m, rel, reg, irt = self.targets(batch_dict, h, w)
        terms = self.dense_head_2d.loss_terms
        if self.w_lid > 0:
            l = F.binary_cross_entropy_with_logits(p[:, 0], m) + ((p[:, 1] - rel).abs() * m).sum() / m.sum().clamp(min=1.0)
            terms['lid'] = float(l.detach())
            out['loss'] = out['loss'] + self.w_lid * l
        if self.w_ir > 0:
            l2 = ((torch.sigmoid(p[:, 2]) - irt).abs() * reg).sum() / reg.sum().clamp(min=1.0)
            terms['irsil'] = float(l2.detach())
            out['loss'] = out['loss'] + self.w_ir * l2
        return out
