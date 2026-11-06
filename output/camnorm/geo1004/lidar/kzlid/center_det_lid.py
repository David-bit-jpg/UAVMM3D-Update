# -*- coding: utf-8 -*-
"""KZ-LiD（10-12，用户：LiDAR 蒸馏——测距之后做朝向）：朝向关键点网络 KZ 原样（只看 RGB），训练时多一个 LiDAR 深度剪影辅助头。
数据集吐 [rgb(3) | ir | depth | tag]（camnorm_sim_full_kp_t_zoom_mm，同一裁窗、同一几何增广）；学生只切前 3 通道（CenterDetKD 的学生路径，无教师）。
辅助头接在 features_2d（和关键点头同一层特征）上，预测每格：
    通道 0  有没有打到无人机的 LiDAR 点（BCE）
    通道 1  点深度相对该机中心的偏移 / 机体最大边（只在有点的格算 L1）—— 机身哪边近哪边远 = 三维形状与朝向
选打点最多的那架真值机，只留深度在其中心 ± (0.75 最大边 + 0.3 m) 内的点（去掉标签外扩带进来的背景点）；中心深度 = 这些点的中位，尺寸 = 该机最大边。部署时辅助头不用，存的权重多出 lid_head.*（加载到原网络时忽略）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from uavdet3d.model.detectors.center_det_kd import CenterDetKD


class CenterDetLiD(CenterDetKD):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        L = model_cfg.get('LIDAR_AUX', None) or {}
        self.w_lid = float(L.get('WEIGHT', 1.0))
        self.ch_d = int(L.get('DEPTH_CH', 4))
        self.ch_t = int(L.get('TAG_CH', 5))
        self.depth_max = float(dataset.dataset_cfg.get('DEPTH_MAX', 24.0))
        C = int(model_cfg.DENSE_HEAD_2D.INPUT_CHANNELS)
        self.lid_head = nn.Sequential(nn.Conv2d(C, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1))

    @torch.no_grad()
    def lid_targets(self, batch_dict, h, w):
        img = batch_dict['image']
        B, K, _, H, W = img.shape
        d = img[:, :, self.ch_d].reshape(B * K, 1, H, W).float() * self.depth_max
        hit = ((img[:, :, self.ch_t] > 0.5).reshape(B * K, 1, H, W) & (d > 0.05)).float()
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
            for k in range(len(g)):                          # 打点最多的那架机（深度带：真值中心 ± 0.75 最大边 + 0.3 m，去掉标签外扩带进来的背景点）
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
        hit = keep
        sy, sx = H // h, W // w
        m = F.max_pool2d(hit, (sy, sx))
        cnt = F.avg_pool2d(hit, (sy, sx))
        dm = F.avg_pool2d(d * hit, (sy, sx)) / cnt.clamp(min=1e-6)
        rel = ((dm - zc[:, None, None, None]) / Ls[:, None, None, None]).clamp(-1.5, 1.5)
        m = m * ok[:, None, None, None].float()
        return m[:, 0, :h, :w], rel[:, 0, :h, :w]

    def forward(self, batch_dict):
        out = super().forward(batch_dict)
        if not self.training:
            return out
        f = self._student_out['features_2d']                  # (B,K,C,h,w)
        B, K, C, h, w = f.shape
        p = self.lid_head(f.reshape(B * K, C, h, w)).float()
        m, rel = self.lid_targets(batch_dict, h, w)
        l = F.binary_cross_entropy_with_logits(p[:, 0], m) + ((p[:, 1] - rel).abs() * m).sum() / m.sum().clamp(min=1.0)
        self.dense_head_2d.loss_terms['lid'] = float(l.detach())
        out['loss'] = out['loss'] + self.w_lid * l
        return out
