# -*- coding: utf-8 -*-
"""CenterDetMMB（10-14，用户「你自己安排，按这个顺序做」第 ③ 项）：KZ-IRLiD（CenterDetMM：网络只看 RGB，训练时 LiDAR + 红外辅助答案）
再多一个头 body_head（features_2d 上，每格 6 个量），专门对付"机头前后反"：
    通道 0-2  机体坐标图：打到无人机的 LiDAR 点在【机体系】里的坐标（x 前 / y 左 / z 上，各除以半边长 → 约 [-1,1]），有点格 L1
              —— LiDAR 版 NOCS（OLD-Net / SurfEmb 思路）：每个像素自己说"我在机身哪儿"，前后不对称的小部件（云台、脚架）能直接投票
    通道 3-5  机头方向：中心格（及 3×3 邻格）预测机体 x 轴在相机系里的单位向量，1 − cos 损失 —— 单独的前后判别（PointPillars 方向头思路）
机体坐标 = R_gtᵀ (d·K⁻¹[u,v,1] − t_gt) / (lwh/2)；d = LiDAR 平面深度（边车 depth，z-buffer 最近点），K = 增广后的输入内参。
水平翻转增广时数据集已按"机体左右镜像"改了标签（cg.hflip_rotation，S = 机体 y 取反），机体坐标随之正确。
部署时 KZ 关键点头不变；body_head 只在 kzbody 评测里用作"前后裁判"。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from center_det_mm import CenterDetMM


def euler_xyz(a):
    """scipy from_euler('xyz')（外旋）：R = Rz(c) Ry(b) Rx(a)。a: (N,3) → (N,3,3)"""
    cx, cy, cz = torch.cos(a[:, 0]), torch.cos(a[:, 1]), torch.cos(a[:, 2])
    sx, sy, sz = torch.sin(a[:, 0]), torch.sin(a[:, 1]), torch.sin(a[:, 2])
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).reshape(-1, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).reshape(-1, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).reshape(-1, 3, 3)
    return Rz @ Ry @ Rx


class CenterDetMMB(CenterDetMM):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        Ba = model_cfg.get('BODY_AUX', None) or {}
        self.w_body = float(Ba.get('WEIGHT', 1.0))
        self.w_nose = float(Ba.get('NOSE_WEIGHT', 1.0))
        C = int(model_cfg.DENSE_HEAD_2D.INPUT_CHANNELS)
        self.body_head = nn.Sequential(nn.Conv2d(C, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 6, 1))

    @torch.no_grad()
    def body_targets(self, batch_dict, h, w):
        """→ bm (BK,h,w) 有点格掩码, bxyz (BK,3,h,w) 机体坐标, nm (BK,h,w) 机头格掩码, nd (BK,3) 机头方向（相机系）"""
        img = batch_dict['image']
        B, K, _, H, W = img.shape
        dev = img.device
        d = img[:, :, self.ch_d].reshape(B * K, H, W).float() * self.depth_max
        tag = (img[:, :, self.ch_t] > 0.5).reshape(B * K, H, W)
        hit = tag & (d > 0.05)
        Kin = torch.as_tensor(batch_dict['intrinsic'], dtype=torch.float32, device=dev).reshape(B, -1, 3, 3)[:, 0]
        gt = batch_dict.get('gt_box9d', None)
        sy, sx = H // h, W // w
        bm = torch.zeros(B * K, h, w, device=dev)
        bxyz = torch.zeros(B * K, 3, h, w, device=dev)
        nm = torch.zeros(B * K, h, w, device=dev)
        nd = torch.zeros(B * K, 3, device=dev)
        vs, us = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        for n in range(B * K):
            if gt is None:
                break
            g = gt[n // K]
            g = g[g.abs().sum(-1) > 0].float()
            if len(g) == 0:
                continue
            Kn = Kin[n // K]
            # 机头方向：每个真值机在其中心格（3×3）
            Rg = euler_xyz(g[:, 6:9])
            q = (Kn @ g[:, :3].T).T
            uc, vc = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
            hn = hit[n]
            best = (0, -1)
            if int(hn.sum()) >= 5:
                v = d[n][hn]
                for k in range(len(g)):
                    L = g[k, 3:6].max().clamp(min=0.05)
                    c = int(((v - g[k, 2]).abs() < 0.75 * L + 0.3).sum())
                    if c > best[0]:
                        best = (c, k)
            # 中心格：选与 LiDAR 同一架（没有 LiDAR 时取最近的那架）
            k = best[1] if best[1] >= 0 else int(torch.argmin(g[:, 2]))
            ci, cj = int(torch.floor(vc[k] / sy)), int(torch.floor(uc[k] / sx))
            if 0 <= ci < h and 0 <= cj < w:
                nm[n, max(0, ci - 1):ci + 2, max(0, cj - 1):cj + 2] = 1.0
                nd[n] = Rg[k][:, 0]
            if best[0] < 5:
                continue
            L = g[k, 3:6].max().clamp(min=0.05)
            sel = hn & ((d[n] - g[k, 2]).abs() < 0.75 * L + 0.3)
            Ki = torch.linalg.inv(Kn)
            ray = torch.stack([us, vs, torch.ones_like(us)], 0).reshape(3, -1)            # 像素中心 = 整数坐标（OpenCV）
            P = (Ki @ ray) * d[n].reshape(1, -1)                                            # 相机系 3D
            Xb = Rg[k].T @ (P - g[k, :3, None])                                             # 机体系
            Xb = (Xb / (0.5 * g[k, 3:6, None]).clamp(min=0.02)).clamp(-1.5, 1.5).reshape(3, H, W)
            s = sel.float()[None, None]
            cnt = F.avg_pool2d(s, (sy, sx))[0, 0, :h, :w]
            acc = F.avg_pool2d(Xb[None] * s, (sy, sx))[0, :, :h, :w]
            bm[n] = (cnt > 0).float()
            bxyz[n] = acc / cnt.clamp(min=1e-6)
        return bm, bxyz, nm, nd

    def forward(self, batch_dict):
        out = super().forward(batch_dict)
        f = self._student_out['features_2d']                  # (B,K,C,h,w)
        B, K, C, h, w = f.shape
        pb = self.body_head(f.reshape(B * K, C, h, w)).float()
        if not self.training:
            batch_dict['body_map'] = pb
            return out
        bm, bxyz, nm, nd = self.body_targets(batch_dict, h, w)
        terms = self.dense_head_2d.loss_terms
        if self.w_body > 0:
            l = ((pb[:, :3] - bxyz).abs().sum(1) * bm).sum() / (3 * bm.sum()).clamp(min=1.0)
            terms['body'] = float(l.detach())
            out['loss'] = out['loss'] + self.w_body * l
        if self.w_nose > 0:
            pd = F.normalize(pb[:, 3:6], dim=1)
            cos = (pd * nd[:, :, None, None]).sum(1)
            l2 = ((1 - cos) * nm).sum() / nm.sum().clamp(min=1.0)
            terms['nose'] = float(l2.detach())
            out['loss'] = out['loss'] + self.w_nose * l2
        return out
