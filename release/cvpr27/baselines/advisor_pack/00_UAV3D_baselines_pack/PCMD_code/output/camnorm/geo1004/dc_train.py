# -*- coding: utf-8 -*-
"""「第二段直接修正深度」（PREREG_dc.md）第 2 步：训练。
网络 DCNet：18 层残差网络编码彩色裁窗（128×128）；5 个标量（log 视角宽、log 预测深度、log 预测长宽高，按训练集标准化）过小 MLP；
池化特征与标量拼接 → 两层 MLP → 深度修正量 δ̂ = log(真值深度 / 预测深度)。另有金字塔式解码器输出 128×128 剪影。
两个版本（结构、数据、子样本顺序、增广完全相同，只差剪影损失权重）：
    DCS  红外轮廓版：|δ̂ − δ| + 1.0 × (二值交叉熵 + 软骰子)（只在有红外剪影的样本上算）
    DC0  对照版：|δ̂ − δ|（剪影损失权重 0，解码器不受训练）
种子 j 用检测模型 P0_s{j} 在仿真训练集上的配对样本（dc/train_s{j}_*）。30 轮、批 128、AdamW 1e-3、单周期余弦。
增广：左右翻转（图与剪影一起翻，δ 不变）、分辨率退化 d ~ 对数均匀 [1, 2.5]（只作用于图）、亮度 / 对比度 ×U(0.75, 1.25)、噪声 σ 0.01。
写 dc/{DCS,DC0}_s{j}.pt（含标量标准化统计量）。"""
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

G = '@@ROOT@@/output/camnorm/geo1004'
O = G + '/dc'
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
EPOCHS, BS, LR = int(os.environ.get('DC_EPOCHS', '30')), 128, 1e-3
LIMIT = int(os.environ.get('DC_LIMIT', '0'))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class DCNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18()
        m.load_state_dict(torch.load(os.path.join(os.environ['TORCH_HOME'], 'hub/checkpoints/resnet18-f37072fd.pth'), map_location='cpu'))
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)
        self.pool = m.maxpool
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (64, 64, 128, 256, 512)])
        self.smooth = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.sil = nn.Conv2d(64, 1, 1)
        self.sc = nn.Sequential(nn.Linear(5, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(512 + 64, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        self.register_buffer('s_mean', torch.zeros(5))
        self.register_buffer('s_std', torch.ones(5))

    def forward(self, x, s):
        s0 = self.stem(x)
        c1 = self.l1(self.pool(s0))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        z = torch.cat([c4.mean(dim=(2, 3)), self.sc((s - self.s_mean) / self.s_std)], 1)
        delta = self.head(z).squeeze(1)
        y = self.lat[4](c4)
        for f, lat in ((c3, self.lat[3]), (c2, self.lat[2]), (c1, self.lat[1]), (s0, self.lat[0])):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        y = F.interpolate(self.smooth(y), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return delta, self.sil(y).squeeze(1)


def scalars(meta):
    return np.array([[m['lw'], m['lz']] + list(m['ldims']) for m in meta], np.float32)


def prep(imgs, rng, train):
    """imgs (n,128,128,3) BGR uint8 → 归一化张量；训练时做分辨率退化与光度扰动（翻转在外面做）"""
    out = []
    for v in imgs:
        if train:
            d = float(np.exp(rng.uniform(0, np.log(2.5))))
            if d > 1.05:
                n = max(8, int(round(128 / d)))
                v = cv2.resize(cv2.resize(v, (n, n), interpolation=cv2.INTER_AREA), (128, 128), interpolation=cv2.INTER_LINEAR)
        out.append(v)
    x = np.stack(out).astype(np.float32)[..., ::-1] / 255.0
    if train:
        n = len(x)
        g = rng.uniform(0.75, 1.25, (n, 1, 1, 1)).astype(np.float32)
        c = rng.uniform(0.75, 1.25, (n, 1, 1, 1)).astype(np.float32)
        m = x.mean(axis=(1, 2, 3), keepdims=True)
        x = np.clip(((x - m) * c + m) * g + rng.normal(0, 0.01, x.shape).astype(np.float32), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2)))


def train(arm, j, log):
    tag = 'train_s%d' % j
    rgb = np.load('%s/%s_rgb.npy' % (O, tag))
    sil = np.load('%s/%s_sil.npy' % (O, tag))
    meta = pickle.load(open('%s/%s_meta.pkl' % (O, tag), 'rb'))
    if LIMIT:
        rgb, sil, meta = rgb[:LIMIT], sil[:LIMIT], meta[:LIMIT]
    S = scalars(meta)
    dl = np.array([m['delta'] for m in meta], np.float32)
    hs = np.array([m['has_sil'] for m in meta], bool)
    torch.manual_seed(j)
    rng = np.random.default_rng(1000 + j)                    # 同种子两个版本：样本顺序、翻转、增广完全相同
    net = DCNet().cuda()
    net.s_mean.copy_(torch.from_numpy(S.mean(0)))
    net.s_std.copy_(torch.from_numpy(S.std(0) + 1e-6))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    N = len(dl)
    steps = EPOCHS * max(1, N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05, anneal_strategy='cos')
    w_sil = 1.0 if arm == 'DCS' else 0.0
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(N)
        for b in range(max(1, N // BS)):
            idx = order[b * BS:(b + 1) * BS]
            fl = rng.random(len(idx)) < 0.5
            im = np.stack([rgb[i][:, ::-1] if f else rgb[i] for i, f in zip(idx, fl)])
            mk = np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]).astype(np.float32) / 255.0
            x = prep(im, rng, True).cuda(non_blocking=True)
            s = torch.from_numpy(S[idx]).cuda()
            t = torch.from_numpy(dl[idx]).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dh, lg = net(x, s)
            loss = (dh.float() - t).abs().mean()
            if w_sil > 0:
                h = torch.from_numpy(hs[idx]).cuda()
                if bool(h.any()):
                    lgf, m = lg.float()[h], torch.from_numpy(mk).cuda()[h]
                    p = torch.sigmoid(lgf)
                    dice = 1 - (2 * (p * m).sum((1, 2)) + 1) / (p.sum((1, 2)) + m.sum((1, 2)) + 1)
                    loss = loss + w_sil * (F.binary_cross_entropy_with_logits(lgf, m) + dice.mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            log('%s_s%d 第 %d 轮 loss %.4f  %.0fs' % (arm, j, ep + 1, float(loss.detach()), time.time() - t0))
    net.eval()
    return net


def main():
    logf = open(O + '/train_log.txt', 'a', encoding='utf-8')

    def log(s):
        line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), s)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    for j in (0, 1, 2):
        for arm in ('DCS', 'DC0'):
            ck = '%s/%s_s%d%s.pt' % (O, arm, j, '_smoke' if LIMIT else '')
            if os.path.exists(ck):
                continue
            net = train(arm, j, log)
            torch.save(net.state_dict(), ck)
            del net
            torch.cuda.empty_cache()
    log('深度修正网络全部完成')


if __name__ == '__main__':
    main()
