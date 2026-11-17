# -*- coding: utf-8 -*-
"""软剪影蒸馏（PREREG_softkd.md）第 2 步：训练红外教师与四个彩色学生。全部是同一个网络 DCNetU：
    18 层残差网络编码 128×128 裁窗 + 5 个标量 MLP + 金字塔解码器出剪影 logits；
    从剪影概率 p 算 3 个可微特征（剪影宽 / 预测框宽 的对数、面积比的对数、平均二值熵），与池化特征、标量一起进深度头；
    深度头输出 δ̂ 与不确定度 log b（拉普拉斯似然：|δ̂ − δ|·exp(−log b) + log b）。
教师 TIR：输入红外裁窗（灰度复制成 3 通道），损失 = 似然 + (二值交叉熵 + 软骰子)，剪影答案 = 红外剪影（连续值）。
学生（输入彩色裁窗，结构、数据、样本顺序、增广完全相同，只差剪影损失）：
    U0  不监督剪影
    UH  硬标签：红外剪影二值化（> 0.5）后 二值交叉熵 + 软骰子
    UC  覆盖率软标签：红外剪影连续值（取窗插值得到的 0–1）二值交叉熵 + 软骰子（= 之前 DCS 的做法）
    UT  教师软剪影蒸馏（不看剪影标签）：边界带（教师 0.5 等值线两侧 3 像素）逐像素 KL（T = 2）+ 整体空间分布 KL（每张图在像素上做 softmax，T = 4），各乘 T 平方
所有剪影损失权重 1。种子 j：样本与 dc 同批（softkd/train_s{j}_*），torch 种子 j、numpy 种子 1000 + j。30 轮、批 128、AdamW 1e-3、单周期余弦。
用法：python softkd_train.py <臂,臂> <种子,种子>"""
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G)
import dc_train as dt  # noqa: E402

O = G + '/softkd'
EPOCHS, BS, LR = int(os.environ.get('SK_EPOCHS', '30')), 128, 1e-3
LIMIT = int(os.environ.get('SK_LIMIT', '0'))
SFX = '_smoke' if LIMIT else ''
S0 = float(np.sqrt(3.0))


class DCNetU(dt.DCNet):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(512 + 64 + 3, 256), nn.ReLU(inplace=True), nn.Linear(256, 2))

    @staticmethod
    def sil_feats(lg):
        p = torch.sigmoid(lg.float())
        ext = p.amax(dim=1).sum(-1) / p.shape[-1] * S0            # 剪影宽 / 预测框宽
        area = p.mean((1, 2)) * 3.0
        ent = -(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6)).mean((1, 2)) * 10.0
        return torch.stack([torch.log(ext + 1e-3), torch.log(area + 1e-3), ent], 1)

    def forward(self, x, s):
        s0 = self.stem(x)
        c1 = self.l1(self.pool(s0))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        y = self.lat[4](c4)
        for f, lat in ((c3, self.lat[3]), (c2, self.lat[2]), (c1, self.lat[1]), (s0, self.lat[0])):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        lg = self.sil(F.interpolate(self.smooth(y), size=x.shape[-2:], mode='bilinear', align_corners=False)).squeeze(1)
        z = torch.cat([c4.mean(dim=(2, 3)), self.sc((s - self.s_mean) / self.s_std), self.sil_feats(lg).to(c4.dtype)], 1)
        o = self.head(z)
        return o[:, 0], o[:, 1].clamp(-6.0, 2.0), lg


def nll(dh, lb, t):
    return ((dh.float() - t).abs() * torch.exp(-lb.float()) + lb.float()).mean()


def bce_dice(lg, m):
    p = torch.sigmoid(lg)
    dice = 1 - (2 * (p * m).sum((1, 2)) + 1) / (p.sum((1, 2)) + m.sum((1, 2)) + 1)
    return F.binary_cross_entropy_with_logits(lg, m) + dice.mean()


def kd_loss(ls, lt, Tb=2.0, Tc=4.0, band=3):
    """边界带逐像素二值 KL（温度 Tb）+ 整体空间分布 KL（温度 Tc）；ls 学生 logits、lt 教师 logits，(B,H,W)"""
    with torch.no_grad():
        m = (lt > 0).float().unsqueeze(1)
        k = 2 * band + 1
        dil = F.max_pool2d(m, k, 1, band)
        ero = 1 - F.max_pool2d(1 - m, k, 1, band)
        bd = (dil - ero).squeeze(1) > 0.5
        pt = torch.sigmoid(lt / Tb)
    ps = torch.sigmoid(ls / Tb)
    kl = pt * (torch.log(pt + 1e-6) - torch.log(ps + 1e-6)) + (1 - pt) * (torch.log(1 - pt + 1e-6) - torch.log(1 - ps + 1e-6))
    lb = (kl * bd).sum() / bd.sum().clamp(min=1) * Tb * Tb
    B = ls.shape[0]
    qt = F.softmax(lt.reshape(B, -1) / Tc, 1)
    lc = (qt * (torch.log(qt + 1e-9) - F.log_softmax(ls.reshape(B, -1) / Tc, 1))).sum(1).mean() * Tc * Tc
    return lb + lc


def load(j):
    tag = 'train_s%d' % j
    rgb = np.load('%s/%s_rgb.npy' % (O, tag), mmap_mode='r')
    ir = np.load('%s/%s_ir.npy' % (O, tag), mmap_mode='r')
    sil = np.load('%s/%s_sil.npy' % (O, tag), mmap_mode='r')
    meta = pickle.load(open('%s/%s_meta.pkl' % (O, tag), 'rb'))
    n = LIMIT or len(meta)
    return np.ascontiguousarray(rgb[:n]), np.ascontiguousarray(ir[:n]), np.ascontiguousarray(sil[:n]), meta[:n]


def gray3(a):
    return np.repeat(np.asarray(a)[..., None], 3, axis=-1)


def train(arm, j, log):
    rgb, ir, sil, meta = load(j)
    S = dt.scalars(meta)
    dl = np.array([m['delta'] for m in meta], np.float32)
    tl = np.load('%s/teacher_logits_s%d%s.npy' % (O, j, SFX), mmap_mode='r') if arm == 'UT' else None
    torch.manual_seed(j)
    rng = np.random.default_rng(1000 + j)
    net = DCNetU().cuda()
    net.s_mean.copy_(torch.from_numpy(S.mean(0)))
    net.s_std.copy_(torch.from_numpy(S.std(0) + 1e-6))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    N = len(dl)
    nb = max(1, N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * nb, pct_start=0.05, anneal_strategy='cos')
    src = ir if arm in ('TIR', 'TIRINV') else (sil if arm == 'SILIN' else rgb)   # SILIN：输入 = 红外剪影真值；TIRINV：红外裁窗一半黑白反转（PREREG_uda）
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(N)
        acc = []
        for b in range(nb):
            idx = order[b * BS:(b + 1) * BS]
            fl = rng.random(len(idx)) < 0.5
            im = np.stack([src[i][:, ::-1] if f else src[i] for i, f in zip(idx, fl)])
            if arm == 'TIRINV':
                inv = rng.random(len(idx)) < 0.5
                im = np.where(inv[:, None, None], 255 - im, im).astype(np.uint8)
            if arm in ('TIR', 'SILIN', 'TIRINV'):
                im = gray3(im)
            mk = torch.from_numpy(np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]).astype(np.float32) / 255.0).cuda()
            x = dt.prep(im, rng, True).cuda(non_blocking=True)
            s = torch.from_numpy(S[idx]).cuda()
            t = torch.from_numpy(dl[idx]).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dh, lb, lg = net(x, s)
            loss = nll(dh, lb, t)
            extra = torch.zeros((), device='cuda')
            if arm in ('TIR', 'UC', 'SILIN', 'TIRINV'):
                extra = bce_dice(lg.float(), mk)
            elif arm == 'UH':
                extra = bce_dice(lg.float(), (mk > 0.5).float())
            elif arm == 'UT':
                lt = torch.from_numpy(np.stack([tl[i][:, ::-1] if f else tl[i] for i, f in zip(idx, fl)]).astype(np.float32)).cuda()
                extra = kd_loss(lg.float(), lt)
            loss = loss + extra
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc.append([float((dh.float() - t).abs().mean()), float(extra.detach())])
        if (ep + 1) % 5 == 0 or ep == 0:
            a = np.mean(acc, 0)
            log('%s_s%d 第 %d 轮 |δ̂−δ| %.4f 剪影损失 %.4f  %.0fs' % (arm, j, ep + 1, a[0], a[1], time.time() - t0))
    net.eval()
    return net


@torch.no_grad()
def teacher_logits(net, j):
    _, ir, sil, meta = load(j)
    S = torch.from_numpy(dt.scalars(meta))
    out = np.zeros((len(meta), 128, 128), np.float16)
    for b in range(0, len(meta), 256):
        x = dt.prep(gray3(ir[b:b + 256]), None, False).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, _, lg = net(x, S[b:b + 256].cuda())
        out[b:b + 256] = lg.float().clamp(-30, 30).cpu().numpy().astype(np.float16)
    np.save('%s/teacher_logits_s%d%s.npy' % (O, j, SFX), out)
    m = sil.astype(np.float32) / 255.0 > 0.5
    lo = out.astype(np.float32)
    p = lo > 0
    iou = (p & m).sum((1, 2)) / np.maximum((p | m).sum((1, 2)), 1)
    q = 1 / (1 + np.exp(-lo))
    return float(np.median(iou)), float((q * (1 - q)).mean())


def main():
    arms = sys.argv[1].split(',')
    seeds = [int(x) for x in sys.argv[2].split(',')]
    logf = open(O + '/train_log.txt', 'a', encoding='utf-8')

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
            if arm == 'UT' and not os.path.exists('%s/teacher_logits_s%d%s.npy' % (O, j, SFX)):
                log('!! 缺教师 logits s%d，停下' % j)
                raise SystemExit(4)
            net = train(arm, j, log)
            torch.save(net.state_dict(), ck)
            if arm == 'TIR':
                iou, soft = teacher_logits(net, j)
                log('教师 s%d 剪影与红外剪影标签 IoU 中位 %.3f；平均 p(1−p) %.4f（越大越软）' % (j, iou, soft))
            del net
            torch.cuda.empty_cache()
    log('完成 %s × %s' % (arms, seeds))


if __name__ == '__main__':
    main()
