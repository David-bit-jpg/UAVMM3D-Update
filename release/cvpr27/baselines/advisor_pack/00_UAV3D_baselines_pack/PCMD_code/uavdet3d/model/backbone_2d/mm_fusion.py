# -*- coding: utf-8 -*-
"""简单多模态融合骨干 MMFusionTv（2026-09-28，用户：「验证多模态模型比单模态好，纯仿真上做，用我们之前那种
flatten 叠加 + 多头注意力权重的简单多模态模型」）。

结构沿用组里原 UAV-MM3D / LGFusionNet 的晚期融合思路（`lightweight_attention_net1.py` 的 modality_attention）：
    每个模态一条独立编码分支（这里每条都是与 RAMP 相同的 TvResNet-34 + FPN，stride 8，512 通道），
    各分支特征叠成 (N, K, C, h, w) → 展平成 (N, K*C, h, w) → 小卷积网络出【每个像素、每个注意力头】一组 K 维模态权重
    → 对 K 做 softmax → 把 C 通道切成 H 组（多头），每组用自己那组权重对 K 个模态加权求和 → 拼回 C 通道 → 3x3 输出卷积。
    单模态（K=1）时 softmax 恒为 1，退化成普通 TvResNet，所以 RGB 基线就是 RAMP 本身。

配置（MODEL.BACKBONE_2D）：
    NAME: MMFusionTv
    MODAL_CHANNELS: [3, 1, 1]         # 与数据集 MODALITIES 顺序一致（rgb, ir, depth）
    ATTENTION_HEADS: 8                # 多头数，须整除 OUT_CHANNELS
    MIXSTYLE_RGB_ONLY: true           # MixStyle 只加在 RGB 分支（它是为 RGB 外观域泛化设计的）
    其余键（ARCH / PRETRAINED / FPN_CHANNELS / OUT_CHANNELS / MIXSTYLE / BGR_INPUT）传给每条分支。
非 RGB 分支：输入先过一个可学习 BatchNorm（IR / 深度在 [0,1]，没有做域统计量归一化），第一层卷积用 ImageNet RGB 滤波器均值初始化。
batch_dict['modal_weights']：本 batch 各模态平均注意力权重（诊断用，不参与损失）。
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tv_resnet import TvResNet


class MMFusionTv(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.modal_channels = [int(c) for c in model_cfg.get('MODAL_CHANNELS', [3])]
        self.K = len(self.modal_channels)
        self.out_channels = int(model_cfg.OUT_CHANNELS)
        self.heads = int(model_cfg.get('ATTENTION_HEADS', 8))
        assert self.out_channels % self.heads == 0, (self.out_channels, self.heads)
        rgb_only_mix = bool(model_cfg.get('MIXSTYLE_RGB_ONLY', True))
        self.branches = nn.ModuleList()
        self.in_norms = nn.ModuleList()
        for k, c in enumerate(self.modal_channels):
            sub = copy.deepcopy(model_cfg)
            sub.INPUT_CHANNELS = c
            sub.pop('INPUT_SLICE', None)
            if k > 0 and rgb_only_mix and 'MIXSTYLE' in sub:
                sub.MIXSTYLE = dict(sub.MIXSTYLE)
                sub.MIXSTYLE['p'] = 0.0
            self.branches.append(TvResNet(sub))
            self.in_norms.append(nn.Identity() if k == 0 else nn.BatchNorm2d(c))
        if self.K > 1:
            C, H, K = self.out_channels, self.heads, self.K
            mid = max(C // 2, 64)
            self.attn = nn.Sequential(
                nn.Conv2d(K * C, mid, kernel_size=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, H * K, kernel_size=1))
            nn.init.zeros_(self.attn[-1].weight)           # 初始等权融合，训练中再学偏好
            nn.init.zeros_(self.attn[-1].bias)
            self.out = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.offsets = [sum(self.modal_channels[:k]) for k in range(self.K)]

    def forward(self, batch_dict):
        x = batch_dict['image']                              # (B, Kimg, Ctot, H, W)
        B, Ki, Ct, Hh, Ww = x.shape
        assert Ct >= sum(self.modal_channels), (Ct, self.modal_channels)
        feats = []
        for k, (c, o) in enumerate(zip(self.modal_channels, self.offsets)):
            xk = x[:, :, o:o + c].reshape(B * Ki, c, Hh, Ww)
            xk = self.in_norms[k](xk).reshape(B, Ki, c, Hh, Ww)
            sub = {'image': xk}
            self.branches[k](sub)
            feats.append(sub['features_2d'].reshape(B * Ki, *sub['features_2d'].shape[2:]))   # (N, C, h, w)
        if self.K == 1:
            y = feats[0]
        else:
            N, C, h, w = feats[0].shape
            st = torch.stack(feats, dim=1)                   # (N, K, C, h, w)
            logits = self.attn(st.reshape(N, self.K * C, h, w)).reshape(N, self.heads, self.K, h, w)
            wts = torch.softmax(logits.float(), dim=2).to(st.dtype)        # 每个头、每个像素：K 个模态权重和为 1
            st = st.reshape(N, self.K, self.heads, C // self.heads, h, w)
            y = (st * wts.permute(0, 2, 1, 3, 4).unsqueeze(3)).sum(dim=1).reshape(N, C, h, w)
            y = self.out(y)
            batch_dict['modal_weights'] = wts.detach().mean(dim=(0, 1, 3, 4))   # (K,)
        batch_dict['features_2d'] = y.reshape(B, Ki, *y.shape[1:])
        return batch_dict
