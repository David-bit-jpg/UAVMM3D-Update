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
if os.environ.get('V2_TAG'):                  # 11-08：小规模试验的存档后缀（基础网络与真实适应同后缀，互相能找到）
    SFX += '_' + os.environ['V2_TAG']
XP_PROB = float(os.environ.get('V2_XP_PROB', '0.5'))    # 11-08：跨域贴图比例（默认 0.5 = 原配方）
XP_REAL = os.environ.get('V2_XP_REAL') == '1'           # 11-08：贴图真实感——机身颜色统计换成随机真实无人机裁窗中心的统计 + 剪影边缘按随机 σ 柔化
PW = os.environ.get('V2_PW') == '1'          # 11-08：原型密度比加权的贴图真值损失（只在真实适应段；RGB 原型 = 滑动平均网络读数头 256 维隐层）
PW_K = int(os.environ.get('V2_PW_K', '32'))
PW_CLIP = float(os.environ.get('V2_PW_CLIP', '4'))
PW_EVERY = int(os.environ.get('V2_PW_EVERY', '250'))
PW_GATE = os.environ.get('V2_GATE') == '1'   # 11-08：红外⇄LiDAR 双向体检守门权重 gproto/wf16/synthesis_matte/gate_s{j%5}.npy（1/0），只在 V2_PW=1 时生效
TP = float(os.environ.get('V2_TP', '0'))     # 11-08：真实外观移植——被贴图样本以此概率把机身颜色场换成真实无人机机身（供体库 selftrain/real_s{j}_tpbank.npz，红外蒸馏剪影头抠出），几何与标签仍是仿真；需 V2_XP_REAL=1
FG_PC = float(os.environ.get('V2_FG_COL', '0'))         # wf23 前景中间域：贴图样本以此概率按教师统计的真实机身亮度重上色（色彩偏离压到 0.3）
FG_PB = float(os.environ.get('V2_FG_BLUR', '0'))        # wf23 前景中间域：贴图样本以此概率加运动模糊（机身层与剪影一起拖影）
FG_L = tuple(int(v) for v in os.environ.get('V2_FG_L', '3,12').split(','))
_FG = {}


def fg_init(j):
    """前景中间域的独立随机数与教师统计（gproto/wf23/fgstats_kz.npz：教师从过滤通过的真实帧里抠出的机身 RGB 均值/标准差 → 换成 BGR）"""
    _FG.clear()
    if FG_PC > 0 or FG_PB > 0:
        _FG['rng'] = np.random.default_rng(50000 + j)
        _FG['col'] = np.load(G + '/gproto/wf23/fgstats_kz.npz')['col_real'][:, [2, 1, 0, 5, 4, 3]].astype(np.float32)


def _mker(L, ang):
    k = np.zeros((L, L), np.float32)
    c = (L - 1) / 2.0
    for t in np.linspace(-c, c, 4 * L):
        x, y = int(round(c + t * np.cos(ang))), int(round(c + t * np.sin(ang)))
        k[min(max(y, 0), L - 1), min(max(x, 0), L - 1)] = 1
    return k / k.sum()


TDP = os.environ.get('V2_TDP') == '1'      # wf23：教师分歧引导的原型再分配——真实裁窗按 RGB 原型聚类，原型内"红外教师 vs LiDAR 教师"距离分歧的稳健散布 s_p 定可靠度 ρ_p=(s_全/s_p)²；
#   真实伪目标损失 × ρ_p（教师吵得凶的外观少信伪目标），中间域（贴图）样本的精确标签损失 × 1/ρ_p（那些外观多靠精确标签）；需 V2_DR（逐裁窗分歧文件）
DEP = float(os.environ.get('V2_DEPLOY', '0'))          # 11-08：部署口径标签输运——档内且被贴图的样本，标量换成真实部署口径（常数尺寸、按常数尺寸重算 z0），标签 δ 按同一参照点换算（真值距离不变）
DEP_BAND = float(os.environ.get('V2_DEPLOY_BAND', '0.22314'))   # |尺寸项| ≤ log 1.25 才输运
_TPB = {}
_UV = np.meshgrid(np.arange(128, dtype=np.float32), np.arange(128, dtype=np.float32))


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
        _BANK.append(np.load(G + {'sf': '/xpaste/bank_bg_sf.npy', 'sf2': '/xpaste/bank_bg_sf2.npy'}.get(os.environ.get('V2_BANK'), '/xpaste/bank_bg.npy')))      # 10-12：整个读进内存；11-08 V2_BANK=sf → 单帧背景库（同帧平移窗，不跨帧）；V2_BANK=sf2 → 单帧检测器选窗的单帧背景库（wf21/strict，xpaste_bank_sf2.py）
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


_REALC = []


def real_centers():
    """11-08：真实无人机外观样本——真实训练小图（纯仿真检测器裁窗，无人机居中）中心 24×24 块的逐通道均值/标准差"""
    if not _REALC:
        r = np.load('%s/selftrain/real_s0_rgb.npy' % G, mmap_mode='r')
        c = np.asarray(r[:, 52:76, 52:76], np.float32).reshape(len(r), -1, 3)
        _REALC.append((c.mean(1), c.std(1) + 1e-3))
    return _REALC[0]


def deploy_conv(S, dl, j):
    """11-08 部署口径标签输运：真实帧的尺寸标量是自标定常数 logSc、z0 按常数尺寸算；仿真样本用检测器逐帧预测尺寸。
    Δ = 预测尺寸横向两轴 log 均值 − 常数尺寸横向两轴 log 均值；标量 (lw, lz, ldims) → (lw, lz − Δ, logSc)，标签 δ → δ + Δ（z* = exp(lz+δ) 不变）"""
    rs = np.load('%s/selftrain/real_s%d_scal.npy' % (G, j % 5))
    assert float(rs[:, 2:5].std(0).max()) < 1e-3
    logSc = rs[0, 2:5].astype(np.float64)
    dsz = S[:, 2:4].astype(np.float64).mean(1) - logSc[:2].mean()
    Sd = S.astype(np.float64).copy()
    Sd[:, 2:5] = logSc
    Sd[:, 1] = S[:, 1] - dsz
    dd = dl.astype(np.float64) + dsz
    return Sd.astype(np.float32), dd.astype(np.float32), np.abs(dsz) <= DEP_BAND


def tp_bank(j):
    if j % 5 not in _TPB:
        z = np.load('%s/selftrain/real_s%d_tpbank.npz' % (G, j % 5))
        _TPB[j % 5] = dict(tex=z['tex'], mom=z['mom'], logw=z['logw'])
    return _TPB[j % 5]


def tp_body(m, w, B, rng):
    """11-08 真实外观移植：从分辨率可接纳（|Δlog w|≤0.2，不足 15 个取最近 15）的真实供体里随机取一个（50% 镜像），
    按剪影矩（质心 + 两轴标准差）对齐，把真实机身颜色场搬进仿真剪影 m 的坐标系"""
    import cv2
    ys, xs = np.nonzero(m)
    cxi, cyi, sxi, syi = xs.mean(), ys.mean(), xs.std() + 0.5, ys.std() + 0.5
    L = B['logw']
    lw = np.log(max(w, 1e-3))
    lo, hi = int(np.searchsorted(L, lw - 0.2)), int(np.searchsorted(L, lw + 0.2))
    if hi - lo < 15:
        c = int(np.searchsorted(L, lw))
        hi = min(len(L), max(c + 8, 15))
        lo = max(0, hi - 15)
    k = int(rng.integers(lo, hi))
    cxj, cyj, sxj, syj = [float(t) for t in B['mom'][k]]
    sg = -1.0 if rng.random() < 0.5 else 1.0
    u, v = _UV
    mx = (cxj + (sxj / sxi) * (u - cxi) * sg).astype(np.float32)
    my = (cyj + (syj / syi) * (v - cyi)).astype(np.float32)
    return cv2.remap(B['tex'][k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


def xpaste(im, sil, bk, rng, rng2=None, ws=None, tpj=None, rtp=None, pm=None):
    """PREREG_xpaste：以 0.5 概率把仿真机身按红外软剪影贴到真实空背景窗上。im (N,128,128,3) uint8，sil (N,128,128) uint8（与 im 同翻转）
    rng2 不为 None 时（PREREG_kpaste）被贴图样本再以 0.5 概率随机改机身明暗；rng2 独立，贴图本身与 V2P 逐位相同
    11-08：比例由 V2_XP_PROB 控制（默认 0.5）；V2_XP_REAL=1 时机身颜色统计迁移到随机真实无人机中心块、剪影 alpha 按 σ~U(0.5,1.5) 高斯柔化"""
    import cv2
    out = im.astype(np.float32)
    for i in range(len(im)):
        hit = rng.random() < XP_PROB and sil[i].max() > 128
        if pm is not None:
            pm.append(bool(hit))
        if hit:
            b = np.asarray(bk[int(rng.integers(len(bk)))], np.float32)
            if rng.random() < 0.5:
                b = b[:, ::-1]
            a = sil[i].astype(np.float32)[..., None] / 255.0
            body = out[i]
            if rng2 is not None and rng2.random() < 0.5:
                body = recolor_body(body, b, a, rng2)
            if XP_REAL:
                mu_r, sd_r = real_centers(); k = int(rng.integers(len(mu_r)))
                m = a[..., 0] > 0.5
                if TP > 0 and rtp is not None and m.sum() > 10 and rtp.random() < TP:   # 11-08：真实外观移植（不动主随机数流 rng）
                    body = tp_body(m, float(ws[i]), tp_bank(tpj), rtp)
                elif m.sum() > 10:
                    mu_s = body[m].mean(0); sd_s = body[m].std(0) + 1e-3
                    body = np.clip((body - mu_s) / sd_s * sd_r[k] + mu_r[k], 0, 255)
                a = cv2.GaussianBlur(a[..., 0], (0, 0), float(rng.uniform(0.5, 1.5)))[..., None]
            if _FG:                                                       # wf23 前景中间域（独立随机数，不动主随机数流）
                r_ = _FG['rng']; m_ = a[..., 0] > 0.5
                if FG_PC > 0 and r_.random() < FG_PC and m_.sum() > 10:
                    t_ = _FG['col'][int(r_.integers(len(_FG['col'])))]
                    Y_ = body.mean(2, keepdims=True)
                    sc_ = float(np.clip(np.mean(t_[3:6]) / (float(Y_[m_].std()) + 1e-3), 0.3, 3.0))
                    body = np.clip(t_[:3] + sc_ * (Y_ - float(Y_[m_].mean())) + 0.3 * ((body - Y_) - (body[m_] - Y_[m_]).mean(0)), 0, 255)
                if FG_PB > 0 and r_.random() < FG_PB:
                    k_ = _mker(int(r_.integers(FG_L[0], FG_L[1] + 1)), float(r_.uniform(0, np.pi)))
                    ba_ = cv2.filter2D(body * a, -1, k_)
                    a = cv2.filter2D(a[..., 0], -1, k_)[..., None]
                    out[i] = ba_ + (1 - a) * b
                    continue
            out[i] = a * body + (1 - a) * b
    return np.clip(out, 0, 255).round().astype(np.uint8)


def _hid(net, x, s):
    """读数头 ReLU 之后的 256 维隐层（RGB 原型空间）"""
    box = {}
    h = net.head[1].register_forward_hook(lambda m, i, o: box.__setitem__('h', o))
    net(x, s)
    h.remove()
    return box['h'].float()


@torch.no_grad()
def proto_rel(ema, real, rs, dr, rng, K=32, clip=4.0):
    """wf23 TDP：RGB 原型 = 滑动平均网络读数头 256 维隐层（标准化 + PCA 32 维）在全部真实裁窗上的 k 均值（K 个）。
    每个原型内两教师分歧 dr（去偏后的 log z_kp − log z_伪，缺失为 NaN）的稳健散布 s_p；ρ_p = (s_全/s_p)²，截在 [1/clip, clip]；成员 < 8 的原型 ρ=1。
    返回 (mu, sd, P, C, rho, 每个真实裁窗的原型号)。"""
    feats = []
    for b in range(0, len(real), 256):
        ii = np.arange(b, min(b + 256, len(real)))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feats.append(_hid(ema, dt.prep(np.stack([real[i] for i in ii]), None, False).cuda(), torch.from_numpy(rs[ii]).cuda()))
    Fz = torch.cat(feats)
    mu = Fz.mean(0)
    sd = Fz.std(0) + 1e-4
    Z = (Fz - mu) / sd
    _, _, V = torch.linalg.svd(Z, full_matrices=False)
    P = V[:32].T.contiguous()
    Y = Z @ P
    C = Y[torch.from_numpy(rng.choice(len(Y), K, replace=False)).cuda()].clone()
    for _ in range(30):
        a = torch.cdist(Y, C).argmin(1)
        for k in range(K):
            m = a == k
            if bool(m.any()):
                C[k] = Y[m].mean(0)
    a = torch.cdist(Y, C).argmin(1).cpu().numpy()
    fin = np.isfinite(dr)
    mad = lambda x: 1.4826 * float(np.median(np.abs(x - np.median(x))))
    sg = mad(dr[fin])
    rho = np.ones(K, np.float32)
    for k in range(K):
        x = dr[(a == k) & fin]
        if len(x) >= 8:
            rho[k] = float(np.clip((sg / max(mad(x), 1e-4)) ** 2, 1.0 / clip, clip))
    return mu, sd, P, C, torch.from_numpy(rho).cuda(), a


@torch.no_grad()
def proto_ratio(ema, rgb, sil, S, real, rs, rng_pw, K=32, clip=4.0, n=4096):
    """11-08：原型密度比 ρ_k = 真实帧占比 / 仿真批（按当前贴图配方渲染）占比，截在 [1/clip, clip]。
    返回 (mu, sd, P, C, rho)：特征标准化参数、PCA 主方向 (256,32)、k 均值中心 (K,32)、ρ̃ (K,)。只用独立随机数流 rng_pw。"""
    feats = []
    si = rng_pw.integers(0, len(rgb), n)
    for b in range(0, n, 256):
        ii = si[b:b + 256]
        im = xpaste(np.stack([rgb[i] for i in ii]), np.stack([sil[i] for i in ii]), bank(), rng_pw)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feats.append(_hid(ema, dt.prep(im, None, False).cuda(), torch.from_numpy(S[ii]).cuda()))
    ri = rng_pw.integers(0, len(rs), n)
    for b in range(0, n, 256):
        ii = ri[b:b + 256]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feats.append(_hid(ema, dt.prep(np.stack([real[i] for i in ii]), None, False).cuda(), torch.from_numpy(rs[ii]).cuda()))
    Fz = torch.cat(feats)
    mu = Fz.mean(0)
    sd = Fz.std(0) + 1e-4
    Z = (Fz - mu) / sd
    _, _, V = torch.linalg.svd(Z, full_matrices=False)
    P = V[:32].T.contiguous()
    Y = Z @ P
    C = Y[torch.from_numpy(rng_pw.choice(len(Y), K, replace=False)).cuda()].clone()
    for _ in range(30):
        a = torch.cdist(Y, C).argmin(1)
        for k in range(K):
            m = a == k
            if bool(m.any()):
                C[k] = Y[m].mean(0)
    a = torch.cdist(Y, C).argmin(1)
    nP = torch.bincount(a[:n], minlength=K).float()
    nR = torch.bincount(a[n:], minlength=K).float()
    rho = ((nR / n) / (nP / n).clamp(min=1e-9)).clamp(1.0 / clip, clip)
    return mu, sd, P, C, rho


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

    def sim_loss(self, net, x, s, t, mk, idx, fl, rng_tf, lid=None, wv=None):
        lt = torch.from_numpy(np.stack([self.tl[i][:, ::-1] if f else self.tl[i] for i, f in zip(idx, fl)]).astype(np.float32)).cuda() if self.kd else None
        if self.geo:
            tf = torch.from_numpy(rng_tf.random(len(idx)) < 0.5).cuda()
            forced = corrupt(torch.sigmoid(lt / 2.0), self.gen) if self.base == 'V2D' else mk
            dh, lb, lg = net(x, s, forced, tf)
        else:
            dh, lb, lg = net(x, s)
        c4 = self.feat['c4']
        if wv is None:
            loss = skt.nll(dh, lb, t)
        else:                                   # 11-08 V2_PW：逐样本加权的拉普拉斯似然（wv 已在批内归一到均值 1）
            loss = (((dh.float() - t).abs() * torch.exp(-lb.float()) + lb.float()) * wv).sum() / wv.sum().clamp(min=1e-6)
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
    fg_init(j)
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
    rng_tp = np.random.default_rng(16000 + j) if TP > 0 else None     # 11-08 真实外观移植的独立随机数
    w_s = np.array([m['w'] for m in meta], np.float32)
    if DEP > 0:                                                         # 11-08 部署口径标签输运
        Sd, dd, elig = deploy_conv(S, dl, j)
        rng_dep = np.random.default_rng(18000 + j)
        dep_acc = []
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
            pmk = [] if DEP > 0 else None
            if ctx.paste:
                im = xpaste(im, np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]), bank(), rng_xp, rng_kc if ctx.kc else None, ws=w_s[idx], tpj=j, rtp=rng_tp, pm=pmk)
            x = dt.prep(appearance(im, rng2), rng, True).cuda(non_blocking=True)
            if DEP > 0:
                cv = (np.array(pmk, bool) if pmk else np.zeros(len(idx), bool)) & elig[idx] & (rng_dep.random(len(idx)) < DEP)
                dep_acc.append(float(cv.mean()))
                s = torch.from_numpy(np.where(cv[:, None], Sd[idx], S[idx])).cuda()
                t = torch.from_numpy(np.where(cv, dd[idx], dl[idx])).cuda()
            else:
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
            if DEP > 0:
                log('%s_s%d 部署口径输运比例 %.3f' % (arm, j, float(np.mean(dep_acc[-nb:]))))
    net.eval()
    return net


def adapt(arm, j, log):
    fg_init(j)
    base = arm.rstrip('p')
    rgb, _, sil, meta = skt.load(j % 5)
    S = dt.scalars(meta)
    dl = np.array([m['delta'] for m in meta], np.float32)
    RSX = os.environ.get('V2_REAL', '')                                   # 11-08 wf21/strict：V2_REAL=nb → 真实裁窗换成单帧检测（P0 不开热图时序背景）版 real_s{j}_nb{rgb,scal}.npy；缺省不变
    real = np.load('%s/selftrain/real_s%d_%srgb.npy' % (G, j % 5, RSX))      # 10-12：整个读进内存（同数据，只改读法）
    rs = np.load('%s/selftrain/real_s%d_%sscal.npy' % (G, j % 5, RSX))
    dps = np.load('%s/selftrain/real_s%d_%s.npy' % (G, j % 5, os.environ.get('V2_DPS', 'dpseudo'))) if base in ('V2R', 'V2M', 'V2J') else None   # 11-08：V2_DPS 换伪目标文件（默认原 dpseudo）
    dpw = np.load('%s/selftrain/real_s%d_%s.npy' % (G, j % 5, os.environ['V2_DPW'])).astype(np.float32) if (os.environ.get('V2_DPW') and dps is not None) else None   # 11-08：伪目标逐样本权重（导师打分器软加权），缺省不用
    lid = load_lid(arm, j)
    if TDP:
        assert not PW, 'TDP 与 PW 互斥'
        dr = np.load('%s/selftrain/real_s%d_%s.npy' % (G, j % 5, os.environ.get('V2_DR', 'nbmdr_fz'))).astype(np.float32)
        assert len(dr) == len(real), (len(dr), len(real))
        rng_td = np.random.default_rng(23000 + j)
        td, td_acc = None, []
    RCP = float(os.environ.get('V2_RC', '0'))                           # wf23：真实机身换景（学生看换景图，教师/伪目标对应原图；只用红外⇄LiDAR 互验过滤通过的裁窗）
    if RCP > 0:
        assert RSX == 'nbm', '换景蒙版按 real_s*_nbmrgb 计算'
        rcA = np.load('%s/gproto/wf23/realcut/masks_dist_s%d.npy' % (G, j % 5))
        rcok = np.load('%s/gproto/wf23/realcut/q_dist_s%d.npz' % (G, j % 5))['passed']
        assert len(rcA) == len(real) and len(rcok) == len(real)
        rng_rcx = np.random.default_rng(21000 + j)
        rc_acc = []
    torch.manual_seed(j)
    rng = np.random.default_rng(2000 + j)
    rng2 = np.random.default_rng(5000 + j)
    rng_tf = np.random.default_rng(6000 + j)
    rng_rc = np.random.default_rng(9000 + j)
    rng_xp = np.random.default_rng(11000 + j)
    rng_kc = np.random.default_rng(13000 + j)
    rng_tp = np.random.default_rng(17000 + j) if TP > 0 else None
    w_s = np.array([m['w'] for m in meta], np.float32)
    if DEP > 0:
        Sd, dd, elig = deploy_conv(S, dl, j)
        rng_dep = np.random.default_rng(19000 + j)
        dep_acc = []
    net = make(arm).cuda()
    net.load_state_dict(torch.load('%s/%s_s%d%s.pt' % (O, {'V2S': 'V2P', 'V2R': 'V2Pp', 'V2M': 'V2Lp', 'V2J': 'V2Hp'}.get(base, base), j, SFX), map_location='cuda'))
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    net.train()
    ctx = Ctx(arm, j, net)
    if PW:
        rng_pw = np.random.default_rng(15000 + j)
        gate = np.load('%s/gproto/wf16/synthesis_matte/gate_s%d.npy' % (G, j % 5)) if PW_GATE else None
        if gate is not None:
            assert len(gate) == len(dl), (len(gate), len(dl))
        pw, pw_acc = None, []
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
        pmk = [] if (DEP > 0 or TDP) else None
        if ctx.paste:
            im = xpaste(im, np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]), bank(), rng_xp, rng_kc if ctx.kc else None, ws=w_s[idx], tpj=j, rtp=rng_tp, pm=pmk)
        x = dt.prep(appearance(im, rng2), rng, True).cuda(non_blocking=True)
        if DEP > 0:
            cv = (np.array(pmk, bool) if pmk else np.zeros(len(idx), bool)) & elig[idx] & (rng_dep.random(len(idx)) < DEP)
            dep_acc.append(float(cv.mean()))
            s = torch.from_numpy(np.where(cv[:, None], Sd[idx], S[idx])).cuda()
            t = torch.from_numpy(np.where(cv, dd[idx], dl[idx])).cuda()
        else:
            s = torch.from_numpy(S[idx]).cuda()
            t = torch.from_numpy(dl[idx]).cuda()
        ridx = rng.integers(0, len(rs), BH)
        rfl = rng.random(BH) < 0.5
        rim = np.stack([real[i][:, ::-1] if f else real[i] for i, f in zip(ridx, rfl)])
        if RCP > 0:
            rim_s = rim.copy()
            nrc = 0
            for k_, (i_, f_) in enumerate(zip(ridx, rfl)):
                if rcok[i_] and rng_rcx.random() < RCP:
                    a_ = (rcA[i_][:, ::-1] if f_ else rcA[i_]).astype(np.float32)[..., None] / 255.0
                    bg_ = bank()[int(rng_rcx.integers(len(bank())))]
                    if rng_rcx.random() < 0.5:
                        bg_ = bg_[:, ::-1]
                    rim_s[k_] = np.clip(a_ * rim[k_].astype(np.float32) + (1 - a_) * bg_.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
                    nrc += 1
            rc_acc.append(nrc / BH)
            rstrong = dt.prep(appearance(rim_s, rng2), rng, True).cuda(non_blocking=True)
        else:
            rstrong = dt.prep(appearance(rim, rng2), rng, True).cuda(non_blocking=True)
        rweak = dt.prep(rim, None, False).cuda(non_blocking=True)
        rsc = torch.from_numpy(rs[ridx]).cuda()
        ramp = 2.0 / (1.0 + np.exp(-10.0 * step / STEPS)) - 1.0
        lt = lid_batch(lid, idx, fl) if ctx.lidar else None
        wv = None
        if TDP:
            if step % PW_EVERY == 0:
                td = proto_rel(ema, real, rs, dr, rng_td)
                log('%s_s%d 第 %d 步 教师分歧原型 ρ 范围 %.2f–%.2f，截顶/截底 %d/%d，真实裁窗平均 ρ %.2f' % (arm, j, step, float(td[4].min()), float(td[4].max()), int((td[4] >= 4.0).sum()), int((td[4] <= 0.25).sum()), float(td[4][torch.from_numpy(td[5]).cuda()].mean())))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                hb = _hid(ema, dt.prep(im, None, False).cuda(non_blocking=True), s)
            pa_ = torch.cdist(((hb - td[0]) / td[1]) @ td[2], td[3]).argmin(1)
            pmt = torch.from_numpy(np.array(pmk, bool) if pmk else np.zeros(len(idx), bool)).cuda()
            wv = torch.where(pmt, 1.0 / td[4][pa_], torch.ones_like(td[4][pa_]))
            wv = wv / wv.mean().clamp(min=1e-6)
            td_acc.append([float(pmt.float().mean()), float(wv[pmt].mean()) if bool(pmt.any()) else 1.0])
        if PW:
            if step % PW_EVERY == 0:
                pw = proto_ratio(ema, rgb, sil, S, real, rs, rng_pw, PW_K, PW_CLIP)
                log('%s_s%d 第 %d 步 原型密度比 ρ̃ 范围 %.2f–%.2f，截顶/截底原型 %d/%d' % (arm, j, step, float(pw[4].min()), float(pw[4].max()), int((pw[4] >= PW_CLIP).sum()), int((pw[4] <= 1.0 / PW_CLIP).sum())))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                hb = _hid(ema, dt.prep(im, None, False).cuda(non_blocking=True), s)
            mu_, sd_, P_, C_, rho_ = pw
            wv = rho_[torch.cdist(((hb - mu_) / sd_) @ P_, C_).argmin(1)]
            if gate is not None:
                wv = wv * torch.from_numpy(gate[idx]).cuda()
            wv = wv / wv.mean().clamp(min=1e-6)
            pw_acc.append([float(wv.sum() ** 2 / (wv.pow(2).sum() * len(wv)).clamp(min=1e-9)), float((wv == 0).float().mean())])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            dE, lbE, lgE = ema(rweak, rsc)
            ldE = ema._lid if ctx.lidar else None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, dh, c4s, _, _ = ctx.sim_loss(net, x, s, t, mk, idx, fl, rng_tf, lt, wv=wv)
            fs = c4s.mean(dim=(2, 3))
            dS, lbS, lgS = net(rstrong, rsc)
            ldS = net._lid if ctx.lidar else None
            fr = ctx.feat['c4'].mean(dim=(2, 3))
            lfd = dfeat(GRL.apply(torch.cat([fs, fr], 0).float(), 0.1 * ramp)).squeeze(1)
        if ctx.rps:
            tr = torch.from_numpy(dps[ridx]).cuda()
            mm = torch.isfinite(tr)
            if (dpw is not None or TDP) and bool(mm.any()):
                wr = torch.ones(len(ridx), device='cuda')
                if dpw is not None:
                    wr = wr * torch.from_numpy(dpw[ridx]).cuda()
                if TDP:
                    wr = wr * td[4][torch.from_numpy(td[5][ridx]).cuda()]
                wr = wr[mm]
                lreg = (((dS.float()[mm] - tr[mm]).abs() * torch.exp(-lbS.float()[mm]) + lbS.float()[mm]) * wr).sum() / wr.sum().clamp(min=1e-6)
            else:
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
            if RCP > 0:
                log('%s_s%d 真实机身换景比例 %.3f' % (arm, j, float(np.mean(rc_acc[-250:]))))
            if DEP > 0:
                log('%s_s%d 部署口径输运比例 %.3f' % (arm, j, float(np.mean(dep_acc[-250:]))))
            if TDP and td_acc:
                b_ = np.mean(td_acc, 0)
                td_acc = []
                log('%s_s%d 中间域样本比例 %.3f，其精确标签损失平均权重 %.2f' % (arm, j, b_[0], b_[1]))
            if PW and pw_acc:
                b_ = np.mean(pw_acc, 0)
                pw_acc = []
                log('%s_s%d 第 %d 步 加权有效样本比 %.3f，守门剔除 %.3f' % (arm, j, step + 1, b_[0], b_[1]))
        se = int(os.environ.get('V2_SAVE_EVERY', '0'))          # 11-08：早期学习检验——每 se 步存一次滑动平均权重
        if se and (step + 1) % se == 0 and (step + 1) < STEPS:
            torch.save(ema.state_dict(), '%s/%s_s%d%s_step%d.pt' % (O, os.environ.get('V2_OUTARM', arm), j, SFX, step + 1))
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
            ck = '%s/%s_s%d%s.pt' % (O, os.environ.get('V2_OUTARM', arm), j, SFX)   # 11-08：V2_OUTARM 只改存档名（训练逻辑仍按 arm）
            if os.path.exists(ck):
                continue
            net = adapt(arm, j, log) if arm.endswith('p') else base_train(arm, j, log)
            torch.save(net.state_dict(), ck)
            del net
            torch.cuda.empty_cache()
    log('完成 %s × %s' % (arms, seeds))


if __name__ == '__main__':
    main()
