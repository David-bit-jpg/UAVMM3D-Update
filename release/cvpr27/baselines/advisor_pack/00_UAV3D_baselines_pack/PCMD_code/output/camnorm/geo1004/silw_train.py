# -*- coding: utf-8 -*-
"""剪影测宽实验（PREREG_silw.md）第 2 步：训练 + 评测。设计 A 的关键一步——宽度「从逐像素剪影量出来」比「直接回归一个数」准不准。
子窗口：边长 s × 机框宽（s ~ 对数均匀 [1.2, 2.5]），中心抖动 ±0.1 机框宽，输出 128×128。
统一答案：子窗口里红外软剪影的宽度——逐列取最大值得到列剖面，左右两侧第一次过 0.5 的位置（线性插值，亚像素）之差；t = log(宽 / 128)。
三组（各 3 个种子；同种子三组看到完全相同的子窗口序列）：
    R  彩色图 → 18 层残差网络 → 回归 t（给完美的数）
    M  彩色图 → 18 层残差网络 + 轻量上采样解码器 → 逐像素剪影（二值交叉熵 + 软骰子，对红外软剪影）→ 用同一个过 0.5 规则量出宽度（给逐像素答案，即设计 A）
    T  红外图 → 18 层残差网络 → 回归 t（参照上限）
训练增广（三组相同）：分辨率退化 d ~ 对数均匀 [1, 2.5]（只作用于输入图，不作用于答案）、左右翻转、亮度 / 对比度 ×U(0.75, 1.25)、噪声 σ 0.01。
20 轮、批 128、AdamW 1e-3、单周期余弦。评测：仿真 test 固定种子子窗口（d = 1 与 d = 2.5），不翻转不加光度；
指标 |t̂ − t| × 100 的中位数；M 量不出宽度（预测剪影没有过 0.5 的列）记作误差 100。真实 test 没有剪影答案，只存 M 的预测剪影图看效果。"""
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

D = '@@ROOT@@/output/camnorm/geo1004'
C = D + '/silw'
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
P, WP, OUT = 256, 256 / 3.0, 128
EPOCHS, BS, LR = int(os.environ.get('SILW_EPOCHS', '20')), 128, 1e-3
SEEDS = [0, 1, 2]
LIMIT = int(os.environ.get('SILW_LIMIT', '0'))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
POOL = ThreadPoolExecutor(8)


def affine(s, jx, jy, n_out):
    Lp = s * WP
    cx, cy = P / 2 + jx * WP, P / 2 + jy * WP
    A = n_out / Lp
    return np.array([[A, 0, (0.5 - cx + Lp / 2) * A - 0.5], [0, A, (0.5 - cy + Lp / 2) * A - 0.5]], np.float64), A


def view_img(patch, s, jx, jy, d, w_orig, flip):
    n_out = OUT
    if d > 1.0:
        n_out = int(max(8, min(OUT, round(s * min(w_orig, WP) / d))))
    M, A = affine(s, jx, jy, n_out)
    src = cv2.GaussianBlur(patch, (0, 0), 0.4 / A) if A < 0.9 else patch
    v = cv2.warpAffine(src, M, (n_out, n_out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if n_out != OUT:
        v = cv2.resize(v, (OUT, OUT), interpolation=cv2.INTER_LINEAR)
    return v[:, ::-1] if flip else v


def view_mask(soft, s, jx, jy, flip):
    M, _ = affine(s, jx, jy, OUT)
    v = cv2.warpAffine(soft, M, (OUT, OUT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    v = v.astype(np.float32) / 255.0
    return v[:, ::-1] if flip else v


def extent(m, thr=0.5):
    """m (H,W) [0,1] → 逐列最大值剖面左右第一次过 thr 的亚像素位置之差；量不出返回 nan。"""
    col = m.max(axis=0)
    idx = np.nonzero(col >= thr)[0]
    if len(idx) == 0:
        return np.nan
    i, j = idx[0], idx[-1]
    left = i - 1 + (thr - col[i - 1]) / max(col[i] - col[i - 1], 1e-6) if i > 0 else 0.0
    right = j + (col[j] - thr) / max(col[j] - col[j + 1], 1e-6) if j < len(col) - 1 else float(len(col) - 1)
    return max(right - left, 1e-3)


def to_tensor(vs, ir, photo, rng):
    x = np.stack(vs).astype(np.float32) / 255.0
    x = np.repeat(x[..., None], 3, axis=-1) if ir else x[..., ::-1]
    if photo:
        n = len(x)
        g = rng.uniform(0.75, 1.25, (n, 1, 1, 1)).astype(np.float32)
        c = rng.uniform(0.75, 1.25, (n, 1, 1, 1)).astype(np.float32)
        m = x.mean(axis=(1, 2, 3), keepdims=True)
        x = np.clip(((x - m) * c + m) * g + rng.normal(0, 0.01, x.shape).astype(np.float32), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2)))


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18()
        m.load_state_dict(torch.load(os.path.join(os.environ['TORCH_HOME'], 'hub/checkpoints/resnet18-f37072fd.pth'), map_location='cpu'))
        self.body = nn.Sequential(*list(m.children())[:-2])
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.fc(self.body(x).mean(dim=(2, 3))).squeeze(1)


class MaskNet(nn.Module):
    """18 层残差网络编码 + 特征金字塔式解码，输出 128×128 剪影概率的对数几率。"""
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18()
        m.load_state_dict(torch.load(os.path.join(os.environ['TORCH_HOME'], 'hub/checkpoints/resnet18-f37072fd.pth'), map_location='cpu'))
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)          # 64×64
        self.pool = m.maxpool
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (64, 64, 128, 256, 512)])
        self.smooth = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)                  # 64×64
        c1 = self.l1(self.pool(s0))        # 32×32
        c2 = self.l2(c1)                   # 16
        c3 = self.l3(c2)                   # 8
        c4 = self.l4(c3)                   # 4
        y = self.lat[4](c4)
        for f, lat in ((c3, self.lat[3]), (c2, self.lat[2]), (c1, self.lat[1]), (s0, self.lat[0])):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        y = self.smooth(y)
        y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(y).squeeze(1)


def params(n, rng, train):
    s = np.exp(rng.uniform(np.log(1.2), np.log(2.5), n))
    jx, jy = rng.uniform(-0.1, 0.1, n), rng.uniform(-0.1, 0.1, n)
    d = np.exp(rng.uniform(0, np.log(2.5), n)) if train else np.ones(n)
    fl = rng.random(n) < 0.5 if train else np.zeros(n, bool)
    return s, jx, jy, d, fl


def make_batch(data, idx, prm, use_ir, photo, rng, d_fixed=None):
    s, jx, jy, d, fl = prm
    dd = d if d_fixed is None else np.full(len(idx), d_fixed)
    arr = data['ir'] if use_ir else data['rgb']
    imgs = list(POOL.map(lambda j: view_img(arr[idx[j]], s[j], jx[j], jy[j], dd[j], data['w'][idx[j]], fl[j]), range(len(idx))))
    masks = list(POOL.map(lambda j: view_mask(data['soft'][idx[j]], s[j], jx[j], jy[j], fl[j]), range(len(idx)))) if data.get('soft') is not None else None
    x = to_tensor(imgs, use_ir, photo, rng)
    if masks is None:
        return x, None, None
    m = np.stack(masks)
    t = np.log(np.array([extent(mm) for mm in m]) / OUT)
    return x, torch.from_numpy(m), t


def train(arm, seed, data, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    prng = np.random.default_rng(2000 + seed)
    N = len(data['w'])
    net = (MaskNet() if arm == 'M' else Scalar()).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * (N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05, anneal_strategy='cos')
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(N)
        for b in range(N // BS):
            idx = order[b * BS:(b + 1) * BS]
            prm = params(BS, rng, True)
            x, m, t = make_batch(data, idx, prm, arm == 'T', True, prng)
            ok = np.isfinite(t)
            x = x.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = net(x)
            if arm == 'M':
                mm = m.cuda(non_blocking=True)
                lg = y.float()
                bce = F.binary_cross_entropy_with_logits(lg, mm)
                p = torch.sigmoid(lg)
                dice = 1 - (2 * (p * mm).sum((1, 2)) + 1) / (p.sum((1, 2)) + mm.sum((1, 2)) + 1)
                loss = bce + dice.mean()
            else:
                tt = torch.from_numpy(np.where(ok, t, 0.0).astype(np.float32)).cuda()
                w = torch.from_numpy(ok.astype(np.float32)).cuda()
                loss = ((y.float() - tt).abs() * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        log('%s_s%d 第 %d 轮 loss %.4f  %.0fs' % (arm, seed, ep + 1, float(loss.detach()), time.time() - t0))
    net.eval()
    return net


@torch.no_grad()
def evaluate(net, arm, data, d_fixed, seed=1234, dump=None):
    rng = np.random.default_rng(seed)
    n = len(data['w'])
    prm = params(n, rng, False)
    errs, t_all = [], []
    shown = []
    for b in range(0, n, 256):
        idx = np.arange(b, min(n, b + 256))
        x, m, t = make_batch(data, idx, tuple(p[idx] for p in prm), arm == 'T', False, None, d_fixed=d_fixed)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = net(x.cuda())
        y = y.float().cpu().numpy()
        if arm == 'M':
            pm = 1 / (1 + np.exp(-y))
            pred = np.array([extent(q) for q in pm])
            tp = np.log(pred / OUT)
            if dump is not None and len(shown) < 12:
                shown += [(x[j], pm[j], None if m is None else m[j].numpy()) for j in range(min(12 - len(shown), len(idx)))]
        else:
            tp = y
        if t is not None:
            e = np.abs(tp - t) * 100.0
            e[~np.isfinite(e)] = 100.0                     # 量不出宽度记作误差 100（PREREG_silw）
            errs.append(e)
            t_all.append(t)
    if dump is not None and shown:
        rows = []
        for xx, pm, gm in shown:
            img = (np.clip(xx.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)[..., ::-1] * 255).astype(np.uint8)
            tiles = [img, cv2.cvtColor((pm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]
            if gm is not None:
                tiles.append(cv2.cvtColor((gm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            rows.append(np.concatenate(tiles, 1))
        cv2.imwrite(dump, np.concatenate([np.concatenate(rows[i:i + 2], 1) for i in range(0, len(rows) - 1, 2)], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return (np.concatenate(errs) if errs else None), (np.concatenate(t_all) if t_all else None)


def load(tag, sim=True):
    n = None
    rgb = np.load('%s/%s_rgb.npy' % (C, tag), mmap_mode='r')
    meta = pickle.load(open('%s/%s_meta.pkl' % (C, tag), 'rb'))
    n = LIMIT or len(meta)
    d = dict(rgb=np.ascontiguousarray(rgb[:n]), w=np.array([m['w'] for m in meta[:n]], np.float64), meta=meta[:n])
    if sim:
        d['ir'] = np.ascontiguousarray(np.load('%s/%s_ir.npy' % (C, tag), mmap_mode='r')[:n])
        d['soft'] = np.ascontiguousarray(np.load('%s/%s_soft.npy' % (C, tag), mmap_mode='r')[:n])
    else:
        d['ir'], d['soft'] = None, None
    return d


def main():
    out_dir = C + ('/smoke' if LIMIT else '')
    os.makedirs(out_dir, exist_ok=True)
    logf = open(out_dir + '/train_log.txt', 'a', encoding='utf-8')

    def log(s):
        line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), s)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    tr, te, rt = load('sim_train'), load('sim_test'), load('real_test', sim=False)
    log('训练 %d 个目标，仿真测试 %d，真实测试 %d' % (len(tr['w']), len(te['w']), len(rt['w'])))
    res = json.load(open(out_dir + '/results.json', encoding='utf-8')) if os.path.exists(out_dir + '/results.json') else {}
    for seed in SEEDS:
        for arm in ('T', 'R', 'M'):
            key = '%s_s%d' % (arm, seed)
            ck = '%s/%s.pt' % (out_dir, key)
            net = (MaskNet() if arm == 'M' else Scalar()).cuda()
            if os.path.exists(ck):
                net.load_state_dict(torch.load(ck, map_location='cuda'))
                net.eval()
            else:
                net = train(arm, seed, tr, log)
                torch.save(net.state_dict(), ck)
            if key in res:
                continue
            r = {}
            for dname, dv in (('sim_d1', 1.0), ('sim_d2.5', 2.5)):
                e, _ = evaluate(net, arm, te, dv, dump=('%s/masks_%s_sim.jpg' % (out_dir, key)) if (arm == 'M' and dname == 'sim_d1') else None)
                np.save('%s/err_%s_%s.npy' % (out_dir, key, dname), e)
                r[dname] = float(np.median(e))
                r[dname + '_fail'] = float(np.mean(e >= 100.0))
            if arm == 'M':
                evaluate(net, arm, rt, 1.0, dump='%s/masks_%s_real.jpg' % (out_dir, key))
            res[key] = r
            log('%s 评测 %s' % (key, json.dumps({k: round(v, 3) for k, v in r.items()}, ensure_ascii=False)))
            with open(out_dir + '/results.json.tmp', 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
            os.replace(out_dir + '/results.json.tmp', out_dir + '/results.json')
            del net
            torch.cuda.empty_cache()
    log('训练评测全部完成')


if __name__ == '__main__':
    main()
