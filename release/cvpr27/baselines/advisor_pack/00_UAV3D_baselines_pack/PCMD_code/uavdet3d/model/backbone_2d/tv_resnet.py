# -*- coding: utf-8 -*-
"""torchvision ResNet-18/34（ImageNet 预训练）+ 轻量 FPN，输出 stride 8 特征，接 CenterHead。

为什么要它：ResNet8x 是从零训的小网络，只见过 2 万帧仿真图，学到的是仿真纹理；
纯仿真模型零样本到真实图上 2D 定位 85%、mavic2 只有 60%。ImageNet 预训练的特征对「真实照片」的
外观有先验，是 sim2real 里最基本的一步，用来检验「差是不是模型的原因」。

配置（MODEL.BACKBONE_2D）：
    NAME: TvResNet
    ARCH: resnet18 | resnet34
    PRETRAINED: true        # ImageNet1K 权重，缓存在 TORCH_HOME（缺省 @@TORCH_HOME@@，不往 C 盘放）
    BGR_INPUT: true         # 数据管线是 cv2 的 BGR，这里翻成 RGB 再进预训练网络
    OUT_CHANNELS: 512       # 与 CenterHead.INPUT_CHANNELS 一致
    FPN_CHANNELS: 128
    INPUT_CHANNELS: 3
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from uavdet3d.model.model_utils.mixstyle import build_mixstyle


class TvResNet(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
        import torchvision.models as tvm
        arch = str(model_cfg.get('ARCH', 'resnet34'))
        pretrained = bool(model_cfg.get('PRETRAINED', True))
        self.bgr_input = bool(model_cfg.get('BGR_INPUT', True))
        self.in_channels = int(model_cfg.get('INPUT_CHANNELS', 3))
        # INPUT_SLICE [a, b]（2026-09-25，IR 教师）：只取数据集吐出的第 a..b-1 通道喂骨干（如 MODALITIES [rgb, ir] 下 [3, 4] = 只看 IR），
        # 这样蒸馏时教师、学生吃同一个 batch，各取各的通道；缺省 None = 全部通道
        sl = model_cfg.get('INPUT_SLICE', None)
        self.input_slice = (int(sl[0]), int(sl[1])) if sl else None
        if self.input_slice is not None:
            assert self.input_slice[1] - self.input_slice[0] == self.in_channels, (self.input_slice, self.in_channels)
        self.out_channels = int(model_cfg.OUT_CHANNELS)
        fpn = int(model_cfg.get('FPN_CHANNELS', 128))
        weights = {'resnet18': tvm.ResNet18_Weights.IMAGENET1K_V1,
                   'resnet34': tvm.ResNet34_Weights.IMAGENET1K_V1,
                   'resnet50': tvm.ResNet50_Weights.IMAGENET1K_V2}[arch] if pretrained else None
        net = getattr(tvm, arch)(weights=weights)
        if self.in_channels != 3:
            # 多模态输入（2026-09-23，蒸馏教师 rgb+ir+depth+tag）：第一层卷积扩到 C 通道。前 3 通道沿用 ImageNet 权重，
            # 其余通道用 RGB 三通道滤波器的均值初始化（一张常数图进来时像一张灰度图，不会一上来冲掉预训练特征）。
            # BGR 交换只作用在前 3 通道（forward）。
            old = net.conv1
            conv = nn.Conv2d(self.in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            with torch.no_grad():
                conv.weight.zero_()
                if self.in_channels >= 3:
                    conv.weight[:, :3] = old.weight
                    if self.in_channels > 3:
                        conv.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
                else:
                    # 1~2 通道（如只看 IR）：每个通道都用 RGB 三通道滤波器的均值（= 把预训练网络当灰度网络用）
                    conv.weight[:] = old.weight.mean(dim=1, keepdim=True)
            net.conv1 = conv
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)   # stride 4
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        c8, c16, c32 = [m[-1].conv2.out_channels if hasattr(m[-1], 'conv2') and not hasattr(m[-1], 'conv3')
                        else m[-1].conv3.out_channels for m in (net.layer2, net.layer3, net.layer4)]
        self.lat8 = nn.Conv2d(c8, fpn, 1)
        self.lat16 = nn.Conv2d(c16, fpn, 1)
        self.lat32 = nn.Conv2d(c32, fpn, 1)
        self.out = nn.Sequential(
            nn.Conv2d(fpn, self.out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(self.out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True))
        self.arch, self.pretrained = arch, pretrained
        self.mixstyle, self.mixstyle_layers = build_mixstyle(model_cfg)
        # FREEZE_BN：把骨干的 BN 锁在 ImageNet 的 running statistics 上（train() 时也不更新、不学仿射）。
        # 动机同 MixStyle：BN 的 running stats 是域相关的，在仿真上微调会把它们拉回仿真分布，
        # 白白丢掉 ImageNet 那份「真实照片」的统计量。
        self.freeze_bn = bool(model_cfg.get('FREEZE_BN', False))
        # FREEZE_STAGES：前 n 级（stem、layer1、layer2 …）整级冻结在 ImageNet 权重上（卷积、BN 仿射和统计量都不动）。
        # 动机：低层滤波器（边缘、纹理）在真实照片上学成，仿真微调只会把它们拉向渲染图的统计；缺省 0 = 不冻。
        self.freeze_stages = int(model_cfg.get('FREEZE_STAGES', 0))
        for m in self._frozen_modules():
            for p in m.parameters():
                p.requires_grad_(False)

    def _frozen_modules(self):
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4][:getattr(self, 'freeze_stages', 0)]

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for m in self._frozen_modules():
                m.eval()
        if mode and getattr(self, 'freeze_bn', False):
            for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
                for mod in m.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.eval()
                        mod.weight.requires_grad_(False)
                        mod.bias.requires_grad_(False)
        return self

    def forward(self, batch_dict):
        x = batch_dict['image']
        B, K, C, H, W = x.shape
        x = x.reshape(B * K, C, H, W)
        if self.input_slice is not None:
            x = x[:, self.input_slice[0]:self.input_slice[1]]
            C = x.shape[1]
        if self.bgr_input and C >= 3:
            x = x[:, [2, 1, 0]] if C == 3 else torch.cat([x[:, [2, 1, 0]], x[:, 3:]], dim=1)
        x = self.stem(x)
        x = self.layer1(x)
        if self.mixstyle is not None and 1 in self.mixstyle_layers:
            x = self.mixstyle(x)
        f8 = self.layer2(x)
        if self.mixstyle is not None and 2 in self.mixstyle_layers:
            f8 = self.mixstyle(f8)
        f16 = self.layer3(f8)
        f32 = self.layer4(f16)
        p = self.lat8(f8)
        p = p + F.interpolate(self.lat16(f16), size=p.shape[-2:], mode='bilinear', align_corners=False)
        p = p + F.interpolate(self.lat32(f32), size=p.shape[-2:], mode='bilinear', align_corners=False)
        y = self.out(p)
        batch_dict['features_2d'] = y.reshape(B, K, y.shape[1], y.shape[2], y.shape[3])
        return batch_dict
