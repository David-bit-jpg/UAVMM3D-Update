# -*- coding: utf-8 -*-
"""PREREG_uda.md 臂 UHDA（简单 UDA：对抗域对齐）。从 softkd/UH_s{j} 出发 2000 步，仿真有标签批次与 selftrain 的 UHFT 完全相同（同一随机数序列）；
另加域分类器（512→256→1）判 c4 池化特征来自仿真还是真实，经梯度反转，系数 0.1·(2/(1+e^{−10p})−1)，p = 步数进度。
真实批次 = selftrain/real_s{j} 的强视图（与 UHFT 消耗同样的随机数）。保存滑动平均权重（0.999，同 UHFT）→ softkd/UHDA_s{j}.pt。"""
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

dt = skt.dt
O = G + '/softkd'
STEPS = int(os.environ.get('ST_STEPS', '2000'))
BH, LR, DECAY = 64, 1e-4, 0.999


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def train(j, log):
    rgb, _, sil, meta = skt.load(j)
    S = dt.scalars(meta)
    dl = np.array([m['delta'] for m in meta], np.float32)
    real = np.load('%s/selftrain/real_s%d_rgb.npy' % (G, j), mmap_mode='r')
    rs = np.load('%s/selftrain/real_s%d_scal.npy' % (G, j))
    torch.manual_seed(j)
    rng = np.random.default_rng(2000 + j)
    net = skt.DCNetU().cuda()
    net.load_state_dict(torch.load('%s/UH_s%d.pt' % (O, j), map_location='cuda'))
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    net.train()
    feat = {}
    net.l4.register_forward_hook(lambda m, i, o: feat.__setitem__('c4', o))
    disc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)).cuda()
    opt = torch.optim.AdamW(list(net.parameters()) + list(disc.parameters()), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    t0 = time.time()
    acc = []
    for step in range(STEPS):
        idx = rng.integers(0, len(dl), BH)
        fl = rng.random(BH) < 0.5
        im = np.stack([rgb[i][:, ::-1] if f else rgb[i] for i, f in zip(idx, fl)])
        mk = torch.from_numpy(np.stack([sil[i][:, ::-1] if f else sil[i] for i, f in zip(idx, fl)]).astype(np.float32) / 255.0).cuda()
        x = dt.prep(im, rng, True).cuda(non_blocking=True)
        s = torch.from_numpy(S[idx]).cuda()
        t = torch.from_numpy(dl[idx]).cuda()
        ridx = rng.integers(0, len(rs), BH)
        rfl = rng.random(BH) < 0.5
        rim = np.stack([real[i][:, ::-1] if f else real[i] for i, f in zip(ridx, rfl)])
        rstrong = dt.prep(rim, rng, True).cuda(non_blocking=True)
        lam = 0.1 * (2.0 / (1.0 + np.exp(-10.0 * step / STEPS)) - 1.0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dh, lb, lg = net(x, s)
            fs = feat['c4'].mean(dim=(2, 3))
            net(rstrong, torch.from_numpy(rs[ridx]).cuda())
            fr = feat['c4'].mean(dim=(2, 3))
            logit = disc(GRL.apply(torch.cat([fs, fr], 0).float(), lam)).squeeze(1)
        lab = torch.cat([torch.zeros(BH), torch.ones(BH)]).cuda()
        ldom = F.binary_cross_entropy_with_logits(logit.float(), lab)
        loss = skt.nll(dh, lb, t) + skt.bce_dice(lg.float(), (mk > 0.5).float()) + ldom
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe_, pn in zip(ema.parameters(), net.parameters()):
                pe_.mul_(DECAY).add_(pn.detach(), alpha=1 - DECAY)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
            accd = float(((logit.float() > 0).float() == lab).float().mean())
        acc.append([float((dh.float() - t).abs().mean().detach()), float(ldom.detach()), accd])
        if (step + 1) % 250 == 0 or step == 0:
            a = np.mean(acc, 0)
            acc = []
            log('UHDA_s%d 第 %d 步 仿真|δ̂−δ| %.4f 域分类损失 %.4f 域分类准确率 %.2f 反转系数 %.3f  %.0fs' % (j, step + 1, a[0], a[1], a[2], lam, time.time() - t0))
    return ema


def main():
    seeds = [int(x) for x in sys.argv[1].split(',')]
    sfx = '_smoke' if STEPS < 2000 else ''
    logf = open(O + '/uda_log.txt', 'a', encoding='utf-8')

    def log(s):
        line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), s)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    for j in seeds:
        ck = '%s/UHDA_s%d%s.pt' % (O, j, sfx)
        if os.path.exists(ck):
            continue
        ema = train(j, log)
        torch.save(ema.state_dict(), ck)
        del ema
        torch.cuda.empty_cache()
    log('完成 UHDA × %s' % seeds)


if __name__ == '__main__':
    main()
