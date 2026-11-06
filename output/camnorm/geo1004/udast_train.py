# -*- coding: utf-8 -*-
"""PREREG_udast.md（sha256 811f6b92）训练。从 softkd/UH_s{j} 出发 2000 步，仿真有标签 64 + 真实无标签 64（抽样序列与 UHFT / UHST / UHDA 相同）。
    UHDAST = UH 配方 + 真实一致性（同 UHST）+ c4 特征域对抗（同 UHDA）
    UHDASO = UHDAST + 剪影输出域对抗（64×64 概率图，小卷积判别器，梯度反转 0.05·ramp）
    UHX    = UHDASO + 学生输入外观增广（灰度 0.2 / 反转 0.15 / 运动模糊 0.3 / JPEG 0.3，独立随机数）
保存滑动平均权重 → softkd/{臂}_s{j}.pt。用法：python udast_train.py <臂,臂> <种子,种子>"""
import copy
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G)
import softkd_train as skt  # noqa: E402
from uda_train import GRL  # noqa: E402

dt = skt.dt
O = G + '/softkd'
STEPS = int(os.environ.get('ST_STEPS', '2000'))
BH, LR, DECAY = 64, 1e-4, 0.999


def appearance(batch, rng):
    """uint8 (N,128,128,3) BGR → 随机灰度 / 反转 / 运动模糊 / JPEG"""
    out = []
    for v in batch:
        v = np.ascontiguousarray(v)
        if rng.random() < 0.2:
            v = np.repeat(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)[..., None], 3, -1)
        if rng.random() < 0.15:
            v = 255 - v
        if rng.random() < 0.3:
            L = int(rng.integers(3, 8))
            k = np.zeros((L, L), np.float32)
            k[L // 2, :] = 1.0 / L
            M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
            k = cv2.warpAffine(k, M, (L, L))
            k /= max(k.sum(), 1e-6)
            v = cv2.filter2D(v, -1, k)
        if rng.random() < 0.3:
            ok, enc = cv2.imencode('.jpg', v, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 71))])
            v = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        out.append(v)
    return np.stack(out)


def out_disc():
    return nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
                         nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(128, 1, 4, 1, 1))


def train(arm, j, log):
    rgb, _, sil, meta = skt.load(j)
    S = dt.scalars(meta)
    dl = np.array([m['delta'] for m in meta], np.float32)
    real = np.load('%s/selftrain/real_s%d_rgb.npy' % (G, j), mmap_mode='r')
    rs = np.load('%s/selftrain/real_s%d_scal.npy' % (G, j))
    use_o, use_a = arm in ('UHDASO', 'UHX'), arm == 'UHX'
    torch.manual_seed(j)
    rng = np.random.default_rng(2000 + j)
    rng2 = np.random.default_rng(3000 + j)
    net = skt.DCNetU().cuda()
    net.load_state_dict(torch.load('%s/UH_s%d.pt' % (O, j), map_location='cuda'))
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    net.train()
    feat = {}
    net.l4.register_forward_hook(lambda m, i, o: feat.__setitem__('c4', o))
    dfeat = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)).cuda()
    dout = out_disc().cuda()
    params = list(net.parameters()) + list(dfeat.parameters()) + (list(dout.parameters()) if use_o else [])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    t0 = time.time()
    acc = []
    lab = torch.cat([torch.zeros(BH), torch.ones(BH)]).cuda()
    for step in range(STEPS):
        idx = rng.integers(0, len(dl), BH)
        fl = rng.random(BH) < 0.5
        im = np.stack([rgb[i][:, ::-1] if f else rgb[i] for i, f in zip(idx, fl)])
        mk = torch.from_numpy(np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]).astype(np.float32) / 255.0).cuda()
        if use_a:
            im = appearance(im, rng2)
        x = dt.prep(im, rng, True).cuda(non_blocking=True)
        s = torch.from_numpy(S[idx]).cuda()
        t = torch.from_numpy(dl[idx]).cuda()
        ridx = rng.integers(0, len(rs), BH)
        rfl = rng.random(BH) < 0.5
        rim = np.stack([real[i][:, ::-1] if f else real[i] for i, f in zip(ridx, rfl)])
        rim_s = appearance(rim, rng2) if use_a else rim
        rstrong = dt.prep(rim_s, rng, True).cuda(non_blocking=True)
        rweak = dt.prep(rim, None, False).cuda(non_blocking=True)
        rsc = torch.from_numpy(rs[ridx]).cuda()
        ramp = 2.0 / (1.0 + np.exp(-10.0 * step / STEPS)) - 1.0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            dE, lbE, lgE = ema(rweak, rsc)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dh, lb, lg = net(x, s)
            fs = feat['c4'].mean(dim=(2, 3))
            dS, _, lgS = net(rstrong, rsc)
            fr = feat['c4'].mean(dim=(2, 3))
            lf = dfeat(GRL.apply(torch.cat([fs, fr], 0).float(), 0.1 * ramp)).squeeze(1)
        loss = skt.nll(dh, lb, t) + skt.bce_dice(lg.float(), (mk > 0.5).float())
        lcons = ((dS.float() - dE.float()).abs() * torch.exp(-lbE.float())).mean() + F.binary_cross_entropy_with_logits(lgS.float(), torch.sigmoid(lgE.float()))
        ldf = F.binary_cross_entropy_with_logits(lf.float(), lab)
        loss = loss + lcons + ldf
        ldo = torch.zeros((), device='cuda')
        if use_o:
            maps = F.avg_pool2d(torch.sigmoid(torch.cat([lg, lgS], 0).float()).unsqueeze(1), 2)
            lo = dout(GRL.apply(maps, 0.05 * ramp))
            ldo = F.binary_cross_entropy_with_logits(lo, lab.view(-1, 1, 1, 1).expand_as(lo))
            loss = loss + ldo
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe_, pn in zip(ema.parameters(), net.parameters()):
                pe_.mul_(DECAY).add_(pn.detach(), alpha=1 - DECAY)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        acc.append([float((dh.float() - t).abs().mean().detach()), float(lcons.detach()), float(ldf.detach()), float(ldo.detach())])
        if (step + 1) % 250 == 0 or step == 0:
            a = np.mean(acc, 0)
            acc = []
            log('%s_s%d 第 %d 步 仿真|δ̂−δ| %.4f 一致性 %.4f 特征域 %.4f 剪影域 %.4f  %.0fs' % (arm, j, step + 1, a[0], a[1], a[2], a[3], time.time() - t0))
    return ema


def main():
    arms = sys.argv[1].split(',')
    seeds = [int(x) for x in sys.argv[2].split(',')]
    assert all(a in ('UHDAST', 'UHDASO', 'UHX') for a in arms)
    sfx = '_smoke' if STEPS < 2000 else ''
    logf = open(O + '/udast_log.txt', 'a', encoding='utf-8')

    def log(s):
        line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), s)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    for j in seeds:
        for arm in arms:
            ck = '%s/%s_s%d%s.pt' % (O, arm, j, sfx)
            if os.path.exists(ck):
                continue
            ema = train(arm, j, log)
            torch.save(ema.state_dict(), ck)
            del ema
            torch.cuda.empty_cache()
    log('完成 %s × %s' % (arms, seeds))


if __name__ == '__main__':
    main()
