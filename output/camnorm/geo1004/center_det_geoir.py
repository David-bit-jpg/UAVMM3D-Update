# -*- coding: utf-8 -*-
"""IR 剪影答案（2026-10-05，用户策略「用 IR 模型先验提升 RGB 检测 / 定位」，PREREG_irloc.md）。

不放进 uavdet3d 包（KZ 实验在训练，包代码不动）：由 geo1004/train_reg.py 在运行时注册进 build_network 的类表。
在 CenterDetGeo 之上加一个只在源域训练时用的辅助头 'irmask'：预测每个特征格里「IR 亮剪影」占的比例。
目标只由 IR 通道和 GT 框算（不读 LiDAR / tag）：每个 GT 框投影成 2D 外接框，外扩 30% 的环带做背景参照，
阈值 = 环带均值 + 0.5·(凸包内 90 分位 − 环带均值)，8 个投影角点凸包内高于阈值的像素算剪影；框内外对比 < 0.03（IR 归一化到 [0,1]）的框不算。
损失 = 带 pos_weight 的 BCE（软目标 = 每格剪影占比）。部署时 aux_heads.irmask.* 被忽略。
构建时同 CenterDetGeo：额外的头在 fork_rng 里建，不改变全局随机数流（各臂训练像素逐位相同）。
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uavdet3d.model.detectors.center_det_geo import CenterDetGeo
from uavdet3d.model.detectors.center_det_kd import CenterDetKD

PROTO8 = torch.tensor([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)], dtype=torch.float32)


def euler_xyz(e):
    """scipy Rotation.from_euler('xyz', e)（外旋 x→y→z）：R = Rz @ Ry @ Rx。e (N,3) -> (N,3,3)"""
    cx, cy, cz = torch.cos(e[:, 0]), torch.cos(e[:, 1]), torch.cos(e[:, 2])
    sx, sy, sz = torch.sin(e[:, 0]), torch.sin(e[:, 1]), torch.sin(e[:, 2])
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).reshape(-1, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).reshape(-1, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).reshape(-1, 3, 3)
    return Rz @ Ry @ Rx


class CenterDetGeoIR(CenterDetGeo):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        a = model_cfg.get('AUX', None) or {}
        self.aux_w['W_IRMASK'] = float(a.get('W_IRMASK', 0.0))
        self.irmask_pos_weight = float(a.get('IRMASK_POS_WEIGHT', 10.0))
        self.ir_min_contrast = float(a.get('IR_MIN_CONTRAST', 0.03))
        if self.aux_w['W_IRMASK'] > 0:
            assert 'ir' in self.aux_ch, 'W_IRMASK 需要 MODALITIES 含 ir'
            in_c = int(model_cfg.DENSE_HEAD_2D.INPUT_CHANNELS)
            mid = int(a.get('CONV_DIM', 128))
            with torch.random.fork_rng(devices=[]):
                self.aux_heads['irmask'] = nn.Sequential(
                    nn.Conv2d(in_c, mid, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(mid), nn.ReLU(),
                    nn.Conv2d(mid, 1, kernel_size=3, stride=1, padding=1))

    def irmask_target(self, img, gt, Kin, hw):
        # 几何一律 float32、关掉 autocast（--use_amp 下矩阵乘会变 bfloat16：转 numpy 报错，像素坐标也只有 ~3 位有效数字）
        with torch.autocast(device_type='cuda', enabled=False):
            return self._irmask_target(img.float(), gt.float(), Kin.float(), hw)

    def _irmask_target(self, img, gt, Kin, hw):
        h, w = hw
        N, _, H, W = img.shape
        c0, _ = self.aux_ch['ir']
        ir = img[:, c0]
        mask = torch.zeros((N, 1, H, W), dtype=img.dtype, device=img.device)
        proto = PROTO8.to(img.device)
        n_used = 0
        for n in range(N):
            g = gt[n]
            g = g[g.abs().sum(1) > 0]
            if len(g) == 0:
                continue
            R = euler_xyz(g[:, 6:9])
            cor = (proto[None] * g[:, None, 3:6]) @ R.transpose(1, 2) + g[:, None, :3]      # (G,8,3)
            uv = cor @ Kin[n].t()
            ok = (uv[..., 2] > 1e-3).all(1)
            uv = uv[..., :2] / uv[..., 2:3].clamp_min(1e-3)
            for k in torch.nonzero(ok).flatten().tolist():
                u0, v0 = uv[k, :, 0].min().item(), uv[k, :, 1].min().item()
                u1, v1 = uv[k, :, 0].max().item(), uv[k, :, 1].max().item()
                bw, bh = u1 - u0, v1 - v0
                if bw < 2 or bh < 2:
                    continue
                X0, Y0, X1, Y1 = [int(round(x)) for x in (u0, v0, u1, v1)]
                E0, F0, E1, F1 = [int(round(x)) for x in (u0 - 0.3 * bw, v0 - 0.3 * bh, u1 + 0.3 * bw, v1 + 0.3 * bh)]
                X0, Y0, X1, Y1 = max(X0, 0), max(Y0, 0), min(X1, W), min(Y1, H)
                E0, F0, E1, F1 = max(E0, 0), max(F0, 0), min(E1, W), min(F1, H)
                if X1 - X0 < 2 or Y1 - Y0 < 2:
                    continue
                # 候选像素只取 8 个投影角点的凸包（比轴对齐外接框紧得多，框角里的亮背景进不来）；背景参照 = 外扩环带去掉外接框
                hull = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(hull, cv2.convexHull(uv[k].detach().float().cpu().numpy().astype(np.int32)), 1)
                hull_t = torch.from_numpy(hull.astype(bool)).to(img.device)
                hv = ir[n][hull_t]
                ring = ir[n, F0:F1, E0:E1].clone()
                ring[Y0 - F0:Y1 - F0, X0 - E0:X1 - E0] = float('nan')
                rv = ring[~torch.isnan(ring)]
                if rv.numel() < 8 or hv.numel() < 4:
                    continue
                rm = rv.mean()
                q90 = torch.quantile(hv.float(), 0.9)
                if (q90 - rm) < self.ir_min_contrast:
                    continue
                thr = rm + 0.5 * (q90 - rm)
                mask[n, 0] = torch.maximum(mask[n, 0], ((ir[n] > thr) & hull_t).to(img.dtype))
                n_used += 1
        frac = F.adaptive_avg_pool2d(mask, (h, w))
        return frac, n_used

    def forward(self, batch_dict):
        ret = CenterDetKD.forward(self, batch_dict)
        if not self.training or not self.aux_heads:
            return ret
        feat = self._student_out['features_2d']
        B, K, C, h, w = feat.shape
        x = feat.reshape(B * K, C, h, w)
        img = batch_dict['image']
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        tg = self.aux_targets(img, (h, w))
        need_geo = 'rdepth' in self.aux_heads or 'irmask' in self.aux_heads
        if need_geo:
            gt = torch.as_tensor(np.asarray(batch_dict['gt_box9d']) if not torch.is_tensor(batch_dict['gt_box9d']) else batch_dict['gt_box9d'],
                                 dtype=torch.float32, device=img.device).reshape(img.shape[0], -1, 9)
            Kin = torch.as_tensor(np.asarray(batch_dict['intrinsic']) if not torch.is_tensor(batch_dict['intrinsic']) else batch_dict['intrinsic'],
                                  dtype=torch.float32, device=img.device).reshape(img.shape[0], -1, 3, 3)[:, 0]
            if 'rdepth' in self.aux_heads:
                tg['rdepth'] = self.rdepth_target(img, gt, Kin, (h, w))
            if 'irmask' in self.aux_heads:
                frac, nu = self.irmask_target(img, gt, Kin, (h, w))
                tg['irmask'] = (frac, None)
                self._irmask_boxes = nu
        terms = self.dense_head_2d.loss_terms
        loss = ret['loss']
        for key, head in self.aux_heads.items():
            pred = head(x)
            t, m = tg[key]
            if key in ('tag', 'irmask'):
                pw = torch.tensor(self.tag_pos_weight if key == 'tag' else self.irmask_pos_weight, device=pred.device)
                l = F.binary_cross_entropy_with_logits(pred, t, pos_weight=pw)
            elif m is not None:
                l = torch.abs(pred - t)[m].mean() if bool(m.any()) else (pred * 0).sum()
            else:
                l = torch.abs(pred - t).mean()
            terms['aux_' + key] = float(l.detach())
            loss = loss + self.aux_w['W_' + key.upper()] * l
        self.dense_head_2d.loss_terms = terms
        return {'loss': loss}
