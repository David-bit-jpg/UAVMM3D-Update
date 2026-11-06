# -*- coding: utf-8 -*-
"""CenterDetMMK（10-15，老师建议「LiDAR 抽关键点 → size → 朝向」，用户：「先做关键点试点」）：
KZ-IRLiD（CenterDetMM：网络只看 RGB，训练时 LiDAR 深度剪影 + 红外剪影辅助答案）再加 kps_head（features_2d 上，每格 40 个量）：
    通道 0-15   8 个 LiDAR 结构关键点（4 臂端、机头、机尾、顶、底）的 2D 投影，相对机体中心投影的偏移 / 64（裁窗像素）
    通道 16-39  这 8 个点在机体系的 3D 位置 / 该机型 LiDAR 实测最大外廓 —— "这架机长什么样"（机型的形状与比例）
关键点模板 = 仿真 LiDAR 按真值位姿累积到机体系后抽的点（lidar/kpsize/templates.npy，lidar_templates.py），每个机型一套；
训练时按真值框尺寸认机型（同机型框尺寸是常数，已核对）。监督只放在真值中心格 3×3 邻域。
部署：读出格的 8 个 2D 点 + 预测的 3D 形状 → PnP 得朝向；形状 × PnP 尺度 = 尺寸。关键点头不进原部署链。"""
import numpy as np
import torch
import torch.nn as nn

from center_det_mm import CenterDetMM
from center_det_mmb import euler_xyz

ORDER = ('arm0', 'arm1', 'arm2', 'arm3', 'nose', 'tail', 'top', 'bottom')


class CenterDetMMK(CenterDetMM):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        Ka = model_cfg.get('KPS_AUX', None) or {}
        self.w_kp = float(Ka.get('KP_WEIGHT', 12.0))
        self.w_lay = float(Ka.get('LAYOUT_WEIGHT', 4.0))
        T = np.load(str(Ka.get('TEMPLATES')), allow_pickle=True).item()
        names = sorted(T)
        self.kp_names = names
        self.register_buffer('tpl_box', torch.tensor(np.stack([T[n]['box_lwh'] for n in names]), dtype=torch.float32), persistent=False)
        self.register_buffer('tpl_kp', torch.tensor(np.stack([T[n]['kp8'] for n in names]), dtype=torch.float32), persistent=False)
        self.register_buffer('tpl_L', torch.tensor([float(np.max(T[n]['lidar_ext'])) for n in names], dtype=torch.float32), persistent=False)
        C = int(model_cfg.DENSE_HEAD_2D.INPUT_CHANNELS)
        self.kps_head = nn.Sequential(nn.Conv2d(C, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 40, 1))

    @torch.no_grad()
    def kps_targets(self, batch_dict, h, w):
        """→ m (BK,h,w) 监督格, off (BK,16,h,w), lay (BK,24,h,w)"""
        img = batch_dict['image']
        B, K, _, H, W = img.shape
        dev = img.device
        sy, sx = H // h, W // w
        Kin = torch.as_tensor(batch_dict['intrinsic'], dtype=torch.float32, device=dev).reshape(B, -1, 3, 3)[:, 0]
        gt = batch_dict.get('gt_box9d', None)
        m = torch.zeros(B * K, h, w, device=dev)
        off = torch.zeros(B * K, 16, h, w, device=dev)
        lay = torch.zeros(B * K, 24, h, w, device=dev)
        n_unk = 0
        for n in range(B * K):
            if gt is None:
                break
            g = gt[n // K]
            g = g[g.abs().sum(-1) > 0].float()
            if len(g) == 0:
                continue
            Kn = Kin[n // K]
            d = (g[:, None, 3:6] - self.tpl_box[None]).abs().max(-1).values           # (G, T)
            dm, ti = d.min(1)
            Rg = euler_xyz(g[:, 6:9])
            for k in range(len(g)):
                if float(dm[k]) > 0.01:
                    n_unk += 1
                    continue
                kb = self.tpl_kp[ti[k]]                                                # (8,3) 机体系
                P = Rg[k] @ kb.T + g[k, :3, None]                                      # (3,8) 相机系
                if (P[2] < 0.05).any() or g[k, 2] < 0.05:
                    continue
                q = Kn @ P
                uv = q[:2] / q[2:3]                                                    # (2,8)
                qc = Kn @ g[k, :3]
                uc, vc = qc[0] / qc[2], qc[1] / qc[2]
                ci, cj = int(torch.floor(vc / sy)), int(torch.floor(uc / sx))
                if not (0 <= ci < h and 0 <= cj < w):
                    continue
                o = ((uv - torch.stack([uc, vc])[:, None]) / 64.0).T.reshape(16)        # x0,y0,x1,y1,...
                l = (kb / self.tpl_L[ti[k]]).reshape(24)
                r0, r1, c0, c1 = max(0, ci - 1), min(h, ci + 2), max(0, cj - 1), min(w, cj + 2)
                m[n, r0:r1, c0:c1] = 1.0
                off[n, :, r0:r1, c0:c1] = o[:, None, None]
                lay[n, :, r0:r1, c0:c1] = l[:, None, None]
        self._kps_unknown = n_unk
        return m, off, lay

    def forward(self, batch_dict):
        out = super().forward(batch_dict)
        f = self._student_out['features_2d']                  # (B,K,C,h,w)
        B, K, C, h, w = f.shape
        pk = self.kps_head(f.reshape(B * K, C, h, w)).float()
        if not self.training:
            batch_dict['kps_map'] = pk
            return out
        m, off, lay = self.kps_targets(batch_dict, h, w)
        terms = self.dense_head_2d.loss_terms
        den = m.sum().clamp(min=1.0)
        l1 = ((pk[:, :16] - off).abs().mean(1) * m).sum() / den
        l2 = ((pk[:, 16:] - lay).abs().mean(1) * m).sum() / den
        terms['kps'] = float(l1.detach())
        terms['kps_lay'] = float(l2.detach())
        terms['kps_unk'] = float(self._kps_unknown)
        out['loss'] = out['loss'] + self.w_kp * l1 + self.w_lay * l2
        return out
