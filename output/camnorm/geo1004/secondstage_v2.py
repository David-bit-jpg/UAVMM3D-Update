# -*- coding: utf-8 -*-
"""第二段 v2（PREREG_v2.md，sha256 19e95a39）训练。
基础网络（ImageNet 初始化，30 轮，softkd 配方，样本 softkd/train_s{j}，学生输入加外观增广）：
    V2A = UH 配方（拉普拉斯似然 + 红外剪影硬标签）+ 外观增广
    V2B = V2A + 红外教师边界蒸馏（预存教师 logits，0.5 等值线两侧 3 像素带内逐像素二值 KL，T = 2，× T²，权重 1）
               + ImageNet 特征蒸馏（冻结 ImageNet resnet18 的 c4，0.005 · 逐位置 L2 距离均值）
    V2C = V2B + 剪影几何支路（DCNetG：4 层步长 2 卷积编码剪影图进深度头；训练时每个样本 0.5 概率用红外剪影真值，否则用自己预测的剪影，截断梯度）
真实适应（从对应基础网络出发 2000 步，仿真批 64 用该臂自己的损失，真实无标签批 64）：
    V2N（PREREG_irablate）= V2A 去掉剪影损失（去红外对照，其余逐位相同）；V2Np 同理
    PREREG_ensconf：种子 j ≥ 5 时样本 / 真实裁窗 / 教师 logits 用 j mod 5，随机数用 j
    V2Rp（PREREG_ssst）= 从 V2Pp 出发，真实批改为拟合去偏伪目标 δ_伪（拉普拉斯似然）+ 剪影一致性，去掉特征对抗对齐
    V2G（PREREG_visgate）= V2P + 剪影交叉熵在红外边界带内按彩色可见度加权（0.25 + 0.75 v）
    V2Sp（PREREG_ssa）= V2Pp 的真实适应，特征对抗对齐换成回归专用子空间对齐（从 V2P 出发）
    V2K（PREREG_kpaste）= V2P + 被贴图样本以 0.5 概率随机改机身相对新背景的明暗（Δ ~ U(−70, 50)、k ~ U(0.6, 1.2)，独立随机数）
    V2P / V2Q（PREREG_xpaste）= V2A / V2N + 跨域贴图（仿真机身按红外软剪影贴到真实空背景窗，概率 0.5，独立随机数）
    V2L / V2Lp / V2Mp（PREREG_lidar_dc，10-12 用户：LiDAR 蒸馏先做测距）= V2P / V2Pp / V2Rp 同配方 + LiDAR 深度剪影头（DCNetL：
               逐像素 有无 LiDAR 点 + log(点深度/预测深度)，仿真监督；预测打点区平均修正与覆盖率喂进 δ 头；真实批加 LiDAR 图滑动平均一致性）
    V2H / V2Hp / V2Jp（PREREG_bgdepth，10-15 老师 BeyondRGB 拓展思路）= V2L / V2Lp / V2Mp 同配方，LiDAR 图加背景点（DCNetLB，4 通道：
               机身点有无 / log(机身点深度/预测深度) / 任意点有无 / log(任意点深度/预测深度)），摘要多喂背景平均深度比与背景覆盖率
    V2Ap / V2Bp / V2Cp = 滑动平均教师自训练（弱视图 → 学生强视图）+ c4 特征域对抗（梯度反转 0.1·ramp）+ 学生外观增广；保存滑动平均权重
用法：python secondstage_v2.py <臂,臂> <种子,种子>"""
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G)
import softkd_train as skt  # noqa: E402
from uda_train import GRL  # noqa: E402
from udast_train import appearance  # noqa: E402

dt = skt.dt
O = G + '/softkd'
EPOCHS = int(os.environ.get('V2_EPOCHS', '30'))
STEPS = int(os.environ.get('V2_STEPS', '2000'))
LIMIT = int(os.environ.get('V2_LIMIT', '0'))
SFX = '_smoke' if (LIMIT or EPOCHS < 30 or STEPS < 2000) else ''
NOADV = os.environ.get('V2_NOADV') == '1'   # 11-02 特征对齐消融：真实适应时去掉 c4 特征对抗对齐（其余不变），存档加后缀 _noadv
if NOADV:
    SFX += '_noadv'


class DCNetG(skt.DCNetU):
    """DCNetU + 剪影几何支路"""

    def __init__(self):
        super().__init__()
        blk = lambda i, o: [nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        self.genc = nn.Sequential(*blk(1, 32), *blk(32, 64), *blk(64, 128), *blk(128, 128))
        self.head = nn.Sequential(nn.Linear(512 + 64 + 3 + 128, 256), nn.ReLU(inplace=True), nn.Linear(256, 2))

    def forward(self, x, s, sil_gt=None, tf_mask=None):
        s0 = self.stem(x)
        c1 = self.l1(self.pool(s0))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        y = self.lat[4](c4)
        for f, lat in ((c3, self.lat[3]), (c2, self.lat[2]), (c1, self.lat[1]), (s0, self.lat[0])):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        lg = self.sil(F.interpolate(self.smooth(y), size=x.shape[-2:], mode='bilinear', align_corners=False)).squeeze(1)
        m = torch.sigmoid(lg.float()).detach()
        if sil_gt is not None:
            m = torch.where(tf_mask[:, None, None], sil_gt, m)
        g = self.genc(m.unsqueeze(1).to(c4.dtype)).mean(dim=(2, 3))
        z = torch.cat([c4.mean(dim=(2, 3)), self.sc((s - self.s_mean) / self.s_std), self.sil_feats(lg).to(c4.dtype), g], 1)
        o = self.head(z)
        return o[:, 0], o[:, 1].clamp(-6.0, 2.0), lg


class DCNetGG(DCNetG):
    """PREREG_v2d：DCNetG + 门控——几何特征乘 sigmoid(a·学生剪影平均熵 + b)（学生越不确定越少用几何支路）"""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(1, 1)
        nn.init.constant_(self.gate.weight, -1.0)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, x, s, sil_gt=None, tf_mask=None):
        s0 = self.stem(x)
        c1 = self.l1(self.pool(s0))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        y = self.lat[4](c4)
        for f, lat in ((c3, self.lat[3]), (c2, self.lat[2]), (c1, self.lat[1]), (s0, self.lat[0])):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        lg = self.sil(F.interpolate(self.smooth(y), size=x.shape[-2:], mode='bilinear', align_corners=False)).squeeze(1)
        p = torch.sigmoid(lg.float())
        m = p.detach()
        if sil_gt is not None:
            m = torch.where(tf_mask[:, None, None], sil_gt, m)
        g = self.genc(m.unsqueeze(1).to(c4.dtype)).mean(dim=(2, 3))
        ent = (-(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean((1, 2)) * 10.0).detach()
        g = g * torch.sigmoid(self.gate(ent.unsqueeze(1))).to(g.dtype)
        z = torch.cat([c4.mean(dim=(2, 3)), self.sc((s - self.s_mean) / self.s_std), self.sil_feats(lg).to(c4.dtype), g], 1)
        o = self.head(z)
        return o[:, 0], o[:, 1].clamp(-6.0, 2.0), lg


class DCNetL(skt.DCNetU):
    """PREREG_lidar_dc：DCNetU + LiDAR 深度剪影头（2 通道：有无点的 logit、log(点深度/预测深度)），其摘要（预测打点区的平均逐点修正、覆盖率）也喂进 δ 头。
    forward 仍返回 (δ̂, log b, 剪影 logit)，LiDAR 图存在 self._lid（与其他臂调用方式兼容）。"""

    def __init__(self):
        super().__init__()
        self.lid = nn.Conv2d(64, 2, 1)
        self.head = nn.Sequential(nn.Linear(512 + 64 + 3 + 2, 256), nn.ReLU(inplace=True), nn.Linear(256, 2))
        self._lid = None

    def forward(self, x, s):
        s0 = self.stem(x)
        c1 = self.l1(self.pool(s0))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        y = self.lat[4](c4)
        for f, lat in ((c3, self.lat[3]), (c2, self.lat[2]), (c1, self.lat[1]), (s0, self.lat[0])):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        y = F.interpolate(self.smooth(y), size=x.shape[-2:], mode='bilinear', align_corners=False)
        lg = self.sil(y).squeeze(1)
        ld = self.lid(y)
        self._lid = ld
        p = torch.sigmoid(ld[:, 0].float())
        dm = (p * ld[:, 1].float()).sum((1, 2)) / p.sum((1, 2)).clamp(min=1e-3)
        cov = torch.log(p.mean((1, 2)) * 10.0 + 1e-3)
        z = torch.cat([c4.mean(dim=(2, 3)), self.sc((s - self.s_mean) / self.s_std), self.sil_feats(lg).to(c4.dtype),
                       torch.stack([dm, cov], 1).to(c4.dtype)], 1)
        o = self.head(z)
        return o[:, 0], o[:, 1].clamp(-6.0, 2.0), lg


class DCNetLB(DCNetL):
    """PREREG_bgdepth：DCNetL 的 LiDAR 图从 2 通道扩到 4 通道（加"裁窗里任意 LiDAR 点"的有无与 log 深度比 = 背景比无人机远多少），
    δ 头多两个摘要：预测背景区（任意点且非机身）的平均 log 深度比、背景覆盖率。"""

    def __init__(self):
        super().__init__()
        self.lid = nn.Conv2d(64, 4, 1)
        self.head = nn.Sequential(nn.Linear(512 + 64 + 3 + 4, 256), nn.ReLU(inplace=True), nn.Linear(256, 2))

    def forward(self, x, s):
        s0 = self.stem(x)
        c1 = self.l1(self.pool(s0))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        y = self.lat[4](c4)
        for f, lat in ((c3, self.lat[3]), (c2, self.lat[2]), (c1, self.lat[1]), (s0, self.lat[0])):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        y = F.interpolate(self.smooth(y), size=x.shape[-2:], mode='bilinear', align_corners=False)
        lg = self.sil(y).squeeze(1)
        ld = self.lid(y)
        self._lid = ld
        p = torch.sigmoid(ld[:, 0].float())
        dm = (p * ld[:, 1].float()).sum((1, 2)) / p.sum((1, 2)).clamp(min=1e-3)
        cov = torch.log(p.mean((1, 2)) * 10.0 + 1e-3)
        pa = torch.sigmoid(ld[:, 2].float())
        wb = pa * (1 - p)
        bm = (wb * ld[:, 3].float()).sum((1, 2)) / wb.sum((1, 2)).clamp(min=1e-3)
        bcov = torch.log(wb.mean((1, 2)) * 10.0 + 1e-3)
        z = torch.cat([c4.mean(dim=(2, 3)), self.sc((s - self.s_mean) / self.s_std), self.sil_feats(lg).to(c4.dtype),
                       torch.stack([dm, cov, bm, bcov], 1).to(c4.dtype)], 1)
        o = self.head(z)
        return o[:, 0], o[:, 1].clamp(-6.0, 2.0), lg


def lid_loss_bg(ld, lt):
    """4 通道版：通道 0/1 同 lid_loss；通道 2 任意点有无（BCE）、通道 3 log(点深度/预测深度)（有点处 L1）"""
    ma = lt[:, 2]
    return (lid_loss(ld[:, :2], lt[:, :2]) + F.binary_cross_entropy_with_logits(ld[:, 2].float(), ma)
            + ((ld[:, 3].float() - lt[:, 3]).abs() * ma).sum() / ma.sum().clamp(min=1.0))


def lid_loss(ld, lt):
    """ld: 网络 LiDAR 图 (B,2,H,W)；lt: 仿真答案 (B,2,H,W)（通道 0 有无点、通道 1 log(点深度/预测深度)）"""
    m = lt[:, 0]
    return F.binary_cross_entropy_with_logits(ld[:, 0].float(), m) + ((ld[:, 1].float() - lt[:, 1]).abs() * m).sum() / m.sum().clamp(min=1.0)


def load_lid(arm, j):
    if arm[:3] in ('V2H', 'V2J'):
        return np.load('%s/train_s%d_lidbg.npy' % (O, j % 5))
    return np.load('%s/train_s%d_lid.npy' % (O, j % 5)) if arm[:3] in ('V2L', 'V2M') else None      # 整个读进内存（机械盘随机读太慢）


def lid_batch(lid, idx, fl):
    return torch.from_numpy(np.stack([lid[i][:, :, ::-1] if f else lid[i] for i, f in zip(idx, fl)]).astype(np.float32)).cuda()


def corrupt(m, gen):
    """(B,H,W) 概率图 → 随机腐蚀 / 膨胀（0–2 次 3×3）、高斯模糊（σ 0–1.5）、以 0.3 概率挖掉一个 16–32 像素方块"""
    B, H, W = m.shape
    out = []
    for i in range(B):
        x = m[i:i + 1, None]
        r = torch.rand(4, generator=gen).tolist()
        it = int(r[0] * 3)
        for _ in range(it):
            x = F.max_pool2d(x, 3, 1, 1) if r[1] < 0.5 else -F.max_pool2d(-x, 3, 1, 1)
        sig = r[2] * 1.5
        if sig > 0.2:
            k = torch.arange(-3, 4, dtype=torch.float32, device=x.device)
            k = torch.exp(-k ** 2 / (2 * sig ** 2))
            k = (k / k.sum())
            x = F.conv2d(F.pad(x, (3, 3, 0, 0), mode='replicate'), k.view(1, 1, 1, 7))
            x = F.conv2d(F.pad(x, (0, 0, 3, 3), mode='replicate'), k.view(1, 1, 7, 1))
        if r[3] < 0.3:
            sz = int(16 + 16 * torch.rand(1, generator=gen).item())
            y0 = int((H - sz) * torch.rand(1, generator=gen).item())
            x0 = int((W - sz) * torch.rand(1, generator=gen).item())
            x = x.clone()
            x[..., y0:y0 + sz, x0:x0 + sz] = 0
        out.append(x[0, 0])
    return torch.stack(out)


def recolor(im, sil, rng):
    """PREREG_v2e：红外剪影引导的外观随机化。im (N,128,128,3) BGR uint8，sil (N,128,128) uint8 软掩码（与 im 同翻转）"""
    out = im.astype(np.float32)
    for i in range(len(im)):
        if rng.random() >= 0.5:
            continue
        m = sil[i].astype(np.float32) / 255.0
        if m.sum() < 10:
            continue
        if rng.random() < 0.5:
            c = rng.uniform(225, 255, 3)
        else:
            c = rng.uniform(0, 255, 3)
        g = out[i].mean(-1)
        shade = np.clip(g / max(float((g * m).sum() / m.sum()), 1.0), 0.5, 1.5)
        a = rng.uniform(0.6, 1.0)
        new = c[None, None, :] * shade[..., None]
        mm = (a * m)[..., None]
        out[i] = out[i] * (1 - mm) + new * mm
        if rng.random() < 0.3:
            b = rng.uniform(0.3, 0.7)
            bg = (1 - m)[..., None]
            out[i] = out[i] * (1 - b * bg) + 220.0 * b * bg
    return np.clip(out, 0, 255).astype(np.uint8)


_BANK = []


def bank():
    if not _BANK:
        _BANK.append(np.load(G + '/xpaste/bank_bg.npy'))      # 10-12：整个读进内存（同数据，只改读法）
    return _BANK[0]


def recolor_body(body, b, a, rng):
    """PREREG_kpaste：机身新亮度 = 新背景机身周围平均亮度 + Δ + k ×（原机身 − 原机身平均亮度）"""
    m = a[..., 0] > 0.5
    if m.sum() < 10:
        return body
    import cv2
    near = (cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~m
    if near.sum() < 10:
        return body
    lum = lambda x: x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299
    d = float(rng.uniform(-70.0, 50.0))
    k = float(rng.uniform(0.6, 1.2))
    bm = float(lum(body)[m].mean())
    gm = float(lum(b)[near].mean())
    return np.clip(gm + d + k * (body - bm), 0, 255)


def xpaste(im, sil, bk, rng, rng2=None):
    """PREREG_xpaste：以 0.5 概率把仿真机身按红外软剪影贴到真实空背景窗上。im (N,128,128,3) uint8，sil (N,128,128) uint8（与 im 同翻转）
    rng2 不为 None 时（PREREG_kpaste）被贴图样本再以 0.5 概率随机改机身明暗；rng2 独立，贴图本身与 V2P 逐位相同"""
    out = im.astype(np.float32)
    for i in range(len(im)):
        if rng.random() < 0.5 and sil[i].max() > 128:
            b = np.asarray(bk[int(rng.integers(len(bk)))], np.float32)
            if rng.random() < 0.5:
                b = b[:, ::-1]
            a = sil[i].astype(np.float32)[..., None] / 255.0
            body = out[i]
            if rng2 is not None and rng2.random() < 0.5:
                body = recolor_body(body, b, a, rng2)
            out[i] = a * body + (1 - a) * b
    return np.clip(out, 0, 255).round().astype(np.uint8)


@torch.no_grad()
def ssa_stats(net, ctx, rgb, S, rng, n=4096, K=32):
    """PREREG_ssa：仿真池化特征的均值、前 K 主方向、方差、方向权重 α（距离修正量对池化特征的线性化梯度）"""
    was = net.training
    net.eval()
    hin = {}
    hk = net.head.register_forward_hook(lambda m, i, o: hin.__setitem__('z', i[0]))
    idx = rng.choice(len(rgb), min(n, len(rgb)), replace=False)
    Z, H = [], []
    for b in range(0, len(idx), 256):
        ii = idx[b:b + 256]
        x = dt.prep(np.stack([rgb[i] for i in ii]), None, False).cuda()
        net(x, torch.from_numpy(S[ii]).cuda())
        Z.append(ctx.feat['c4'].float().mean(dim=(2, 3)).cpu())
        H.append(hin['z'].float().cpu())
    hk.remove()
    Z, H = torch.cat(Z), torch.cat(H)
    mu = Z.mean(0)
    C = torch.cov((Z - mu).T)
    ev, evec = torch.linalg.eigh(C)
    lam, V = ev.flip(0)[:K].clamp(min=1e-6), evec.flip(1)[:, :K]
    W1, b1, W2 = net.head[0].weight.float().cpu(), net.head[0].bias.float().cpu(), net.head[2].weight.float().cpu()
    gate = ((H @ W1.T + b1) > 0).float()
    g = ((gate * W2[0]).mean(0) @ W1)[:512]
    gv = (g @ V).abs()
    alpha = 1 + gv / gv.max().clamp(min=1e-9)
    net.train(was)
    return mu.cuda(), V.cuda(), lam.cuda(), alpha.cuda()


def ssa_loss(z, ss):
    mu, V, lam, alpha = ss
    p = (z - mu) @ V
    m, s = p.mean(0), p.var(0).clamp(min=1e-6)
    kl = 0.5 * (s / lam + lam / s + m * m * (1 / lam + 1 / s)) - 1
    return (alpha * kl).sum() / alpha.sum()


@torch.no_grad()
def vis_weight(x, m, band=3):
    """PREREG_visgate：x (B,3,H,W) 学生输入，m (B,H,W) 红外剪影二值 → 交叉熵像素权重（边界带内 0.25 + 0.75 v，带外 1）"""
    g = x.float().mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    mag = torch.sqrt(F.conv2d(g, kx, padding=1) ** 2 + F.conv2d(g, kx.transpose(2, 3), padding=1) ** 2)
    mag = F.avg_pool2d(mag, 3, 1, 1)
    q = torch.quantile(mag.flatten(1), 0.9, dim=1).view(-1, 1, 1, 1)
    v = (mag / (q + 1e-6)).clamp(0, 1)
    mm = m.unsqueeze(1)
    k = 2 * band + 1
    bd = (F.max_pool2d(mm, k, 1, band) - (1 - F.max_pool2d(1 - mm, k, 1, band))) > 0.5
    w = torch.where(bd, 0.25 + 0.75 * v, torch.ones_like(v))
    return w.squeeze(1)


def make(arm):
    if arm.startswith('V2D'):
        return DCNetGG()
    if arm.startswith(('V2H', 'V2J')):
        return DCNetLB()
    if arm.startswith(('V2L', 'V2M')):
        return DCNetL()
    return DCNetG() if arm.startswith('V2C') else skt.DCNetU()


def boundary_kd(ls, lt, T=2.0, band=3):
    with torch.no_grad():
        m = (lt > 0).float().unsqueeze(1)
        k = 2 * band + 1
        bd = (F.max_pool2d(m, k, 1, band) - (1 - F.max_pool2d(1 - m, k, 1, band))).squeeze(1) > 0.5
        pt = torch.sigmoid(lt / T)
    ps = torch.sigmoid(ls / T)
    kl = pt * (torch.log(pt + 1e-6) - torch.log(ps + 1e-6)) + (1 - pt) * (torch.log(1 - pt + 1e-6) - torch.log(1 - ps + 1e-6))
    return (kl * bd).sum() / bd.sum().clamp(min=1) * T * T


class Ctx:
    """一个臂在仿真批次上的损失（基础训练与真实适应共用）"""

    def __init__(self, arm, j, net):
        self.base = arm.rstrip('p')
        self.kd = self.base in ('V2B', 'V2C', 'V2D', 'V2E')
        self.geo = self.base in ('V2C', 'V2D')
        self.nosil = self.base in ('V2N', 'V2Q')
        self.paste = self.base in ('V2P', 'V2Q', 'V2K', 'V2S', 'V2G', 'V2R', 'V2L', 'V2M', 'V2H', 'V2J')
        self.rps = self.base in ('V2R', 'V2M', 'V2J')
        self.lidar = self.base in ('V2L', 'V2M', 'V2H', 'V2J')
        self.lidbg = self.base in ('V2H', 'V2J')
        self.gate = self.base == 'V2G'
        self.ssa = self.base == 'V2S'
        self.kc = self.base == 'V2K'
        self.gen = torch.Generator().manual_seed(7000 + j)
        self.feat = {}
        net.l4.register_forward_hook(lambda m, i, o: self.feat.__setitem__('c4', o))
        if self.kd:
            self.tl = np.load('%s/teacher_logits_s%d.npy' % (O, j % 5), mmap_mode='r')
            self.inet = skt.DCNetU().cuda().eval()          # ImageNet 初始化的骨干（只用 c4）
            for p in self.inet.parameters():
                p.requires_grad_(False)

    def inet_c4(self, x):
        n = self.inet
        return n.l4(n.l3(n.l2(n.l1(n.pool(n.stem(x))))))

    def sim_loss(self, net, x, s, t, mk, idx, fl, rng_tf, lid=None):
        lt = torch.from_numpy(np.stack([self.tl[i][:, ::-1] if f else self.tl[i] for i, f in zip(idx, fl)]).astype(np.float32)).cuda() if self.kd else None
        if self.geo:
            tf = torch.from_numpy(rng_tf.random(len(idx)) < 0.5).cuda()
            forced = corrupt(torch.sigmoid(lt / 2.0), self.gen) if self.base == 'V2D' else mk
            dh, lb, lg = net(x, s, forced, tf)
        else:
            dh, lb, lg = net(x, s)
        c4 = self.feat['c4']
        loss = skt.nll(dh, lb, t)
        if self.lidar:
            loss = loss + (lid_loss_bg(net._lid, lid) if self.lidbg else lid_loss(net._lid, lid))
        if self.gate:
            m = (mk > 0.5).float()
            loss = loss + F.binary_cross_entropy_with_logits(lg.float(), m, weight=vis_weight(x, m)) + skt.bce_dice(lg.float(), m) - F.binary_cross_entropy_with_logits(lg.float(), m)
        elif not self.nosil:
            loss = loss + skt.bce_dice(lg.float(), (mk > 0.5).float())
        lk = lf = torch.zeros((), device='cuda')
        if self.kd:
            lk = boundary_kd(lg.float(), lt)
            with torch.no_grad():
                ci = self.inet_c4(x)
            lf = 0.005 * (c4.float() - ci.float()).pow(2).sum(1).sqrt().mean()
            loss = loss + lk + lf
        return loss, dh, c4, float(lk.detach()), float(lf.detach())


def base_train(arm, j, log):
    rgb, _, sil, meta = skt.load(j % 5)
    lid = load_lid(arm, j)
    if LIMIT:
        rgb, sil, meta = rgb[:LIMIT], sil[:LIMIT], meta[:LIMIT]
    S = dt.scalars(meta)
    dl = np.array([m['delta'] for m in meta], np.float32)
    torch.manual_seed(j)
    rng = np.random.default_rng(1000 + j)
    rng2 = np.random.default_rng(3000 + j)
    rng_tf = np.random.default_rng(4000 + j)
    rng_rc = np.random.default_rng(8000 + j)
    rng_xp = np.random.default_rng(10000 + j)
    rng_kc = np.random.default_rng(12000 + j)
    net = make(arm).cuda()
    net.s_mean.copy_(torch.from_numpy(S.mean(0)))
    net.s_std.copy_(torch.from_numpy(S.std(0) + 1e-6))
    ctx = Ctx(arm, j, net)
    opt = torch.optim.AdamW(net.parameters(), lr=skt.LR, weight_decay=1e-4)
    N = len(dl)
    nb = max(1, N // skt.BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=skt.LR, total_steps=EPOCHS * nb, pct_start=0.05, anneal_strategy='cos')
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(N)
        acc = []
        for b in range(nb):
            idx = order[b * skt.BS:(b + 1) * skt.BS]
            fl = rng.random(len(idx)) < 0.5
            im = np.stack([rgb[i][:, ::-1] if f else rgb[i] for i, f in zip(idx, fl)])
            mk = torch.from_numpy(np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]).astype(np.float32) / 255.0).cuda()
            if arm.startswith('V2E'):
                im = recolor(im, np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]), rng_rc)
            if ctx.paste:
                im = xpaste(im, np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]), bank(), rng_xp, rng_kc if ctx.kc else None)
            x = dt.prep(appearance(im, rng2), rng, True).cuda(non_blocking=True)
            s = torch.from_numpy(S[idx]).cuda()
            t = torch.from_numpy(dl[idx]).cuda()
            lt = lid_batch(lid, idx, fl) if ctx.lidar else None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, dh, _, lk, lf = ctx.sim_loss(net, x, s, t, mk, idx, fl, rng_tf, lt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc.append([float((dh.float() - t).abs().mean().detach()), lk, lf])
        if (ep + 1) % 5 == 0 or ep == 0:
            a = np.mean(acc, 0)
            log('%s_s%d 第 %d 轮 |δ̂−δ| %.4f 边界蒸馏 %.4f ImageNet 特征蒸馏 %.4f  %.0fs' % (arm, j, ep + 1, a[0], a[1], a[2], time.time() - t0))
    net.eval()
    return net


def adapt(arm, j, log):
    base = arm.rstrip('p')
    rgb, _, sil, meta = skt.load(j % 5)
    S = dt.scalars(meta)
    dl = np.array([m['delta'] for m in meta], np.float32)
    real = np.load('%s/selftrain/real_s%d_rgb.npy' % (G, j % 5))      # 10-12：整个读进内存（同数据，只改读法）
    rs = np.load('%s/selftrain/real_s%d_scal.npy' % (G, j % 5))
    dps = np.load('%s/selftrain/real_s%d_dpseudo.npy' % (G, j % 5)) if base in ('V2R', 'V2M', 'V2J') else None
    lid = load_lid(arm, j)
    torch.manual_seed(j)
    rng = np.random.default_rng(2000 + j)
    rng2 = np.random.default_rng(5000 + j)
    rng_tf = np.random.default_rng(6000 + j)
    rng_rc = np.random.default_rng(9000 + j)
    rng_xp = np.random.default_rng(11000 + j)
    rng_kc = np.random.default_rng(13000 + j)
    net = make(arm).cuda()
    net.load_state_dict(torch.load('%s/%s_s%d%s.pt' % (O, {'V2S': 'V2P', 'V2R': 'V2Pp', 'V2M': 'V2Lp', 'V2J': 'V2Hp'}.get(base, base), j, SFX), map_location='cuda'))
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    net.train()
    ctx = Ctx(arm, j, net)
    dfeat = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)).cuda()
    if ctx.ssa:
        ss = ssa_stats(net, ctx, rgb, S, np.random.default_rng(14000 + j))
        log('%s_s%d 子空间：K=32，λ 范围 %.3g–%.3g，α 范围 %.2f–%.2f' % (arm, j, float(ss[2].min()), float(ss[2].max()), float(ss[3].min()), float(ss[3].max())))
    opt = torch.optim.AdamW(list(net.parameters()) + list(dfeat.parameters()), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    BH = 64
    lab = torch.cat([torch.zeros(BH), torch.ones(BH)]).cuda()
    t0 = time.time()
    acc = []
    for step in range(STEPS):
        idx = rng.integers(0, len(dl), BH)
        fl = rng.random(BH) < 0.5
        im = np.stack([rgb[i][:, ::-1] if f else rgb[i] for i, f in zip(idx, fl)])
        mk = torch.from_numpy(np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]).astype(np.float32) / 255.0).cuda()
        if arm.startswith('V2E'):
            im = recolor(im, np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]), rng_rc)
        if ctx.paste:
            im = xpaste(im, np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]), bank(), rng_xp, rng_kc if ctx.kc else None)
        x = dt.prep(appearance(im, rng2), rng, True).cuda(non_blocking=True)
        s = torch.from_numpy(S[idx]).cuda()
        t = torch.from_numpy(dl[idx]).cuda()
        ridx = rng.integers(0, len(rs), BH)
        rfl = rng.random(BH) < 0.5
        rim = np.stack([real[i][:, ::-1] if f else real[i] for i, f in zip(ridx, rfl)])
        rstrong = dt.prep(appearance(rim, rng2), rng, True).cuda(non_blocking=True)
        rweak = dt.prep(rim, None, False).cuda(non_blocking=True)
        rsc = torch.from_numpy(rs[ridx]).cuda()
        ramp = 2.0 / (1.0 + np.exp(-10.0 * step / STEPS)) - 1.0
        lt = lid_batch(lid, idx, fl) if ctx.lidar else None
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            dE, lbE, lgE = ema(rweak, rsc)
            ldE = ema._lid if ctx.lidar else None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, dh, c4s, _, _ = ctx.sim_loss(net, x, s, t, mk, idx, fl, rng_tf, lt)
            fs = c4s.mean(dim=(2, 3))
            dS, lbS, lgS = net(rstrong, rsc)
            ldS = net._lid if ctx.lidar else None
            fr = ctx.feat['c4'].mean(dim=(2, 3))
            lfd = dfeat(GRL.apply(torch.cat([fs, fr], 0).float(), 0.1 * ramp)).squeeze(1)
        if ctx.rps:
            tr = torch.from_numpy(dps[ridx]).cuda()
            mm = torch.isfinite(tr)
            lreg = skt.nll(dS.float()[mm], lbS.float()[mm], tr[mm]) if bool(mm.any()) else torch.zeros((), device='cuda')
            lcons = lreg + F.binary_cross_entropy_with_logits(lgS.float(), torch.sigmoid(lgE.float()))
            if ctx.lidar:
                pE = torch.sigmoid(ldE[:, 0].float())
                lcons = lcons + F.binary_cross_entropy_with_logits(ldS[:, 0].float(), pE) + ((ldS[:, 1].float() - ldE[:, 1].float()).abs() * pE).sum() / pE.sum().clamp(min=1.0)
            ldf = torch.zeros((), device='cuda')
            loss = loss + lcons
        else:
            lcons = ((dS.float() - dE.float()).abs() * torch.exp(-lbE.float())).mean() + F.binary_cross_entropy_with_logits(lgS.float(), torch.sigmoid(lgE.float()))
            if ctx.lidar:
                pE = torch.sigmoid(ldE[:, 0].float())
                lcons = lcons + F.binary_cross_entropy_with_logits(ldS[:, 0].float(), pE) + ((ldS[:, 1].float() - ldE[:, 1].float()).abs() * pE).sum() / pE.sum().clamp(min=1.0)
            ldf = ssa_loss(fr.float(), ss) if ctx.ssa else F.binary_cross_entropy_with_logits(lfd.float(), lab)
            loss = loss + lcons + (0.1 * ldf if ctx.ssa else (0.0 if NOADV else 1.0) * ldf)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe_, pn in zip(ema.parameters(), net.parameters()):
                pe_.mul_(0.999).add_(pn.detach(), alpha=0.001)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        acc.append([float((dh.float() - t).abs().mean().detach()), float(lcons.detach()), float(ldf.detach())])
        if (step + 1) % 250 == 0 or step == 0:
            a = np.mean(acc, 0)
            acc = []
            log('%s_s%d 第 %d 步 仿真|δ̂−δ| %.4f 一致性 %.4f 特征域 %.4f  %.0fs' % (arm, j, step + 1, a[0], a[1], a[2], time.time() - t0))
    return ema


def main():
    arms = sys.argv[1].split(',')
    seeds = [int(x) for x in sys.argv[2].split(',')]
    assert all(a in ('V2A', 'V2B', 'V2C', 'V2D', 'V2E', 'V2N', 'V2P', 'V2Q', 'V2K', 'V2G', 'V2Ap', 'V2Sp', 'V2Gp', 'V2Rp', 'V2Bp', 'V2Cp', 'V2Dp', 'V2Ep', 'V2Np', 'V2Pp', 'V2Qp', 'V2Kp', 'V2L', 'V2Lp', 'V2Mp', 'V2H', 'V2Hp', 'V2Jp') for a in arms)
    logf = open(O + '/v2_log.txt', 'a', encoding='utf-8')

    def log(s):
        line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), s)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    for j in seeds:
        for arm in arms:
            ck = '%s/%s_s%d%s.pt' % (O, arm, j, SFX)
            if os.path.exists(ck):
                continue
            net = adapt(arm, j, log) if arm.endswith('p') else base_train(arm, j, log)
            torch.save(net.state_dict(), ck)
            del net
            torch.cuda.empty_cache()
    log('完成 %s × %s' % (arms, seeds))


if __name__ == '__main__':
    main()
