# -*- coding: utf-8 -*-
"""2D 尺寸头探针：冻住 P1 骨干，只训一个预测「目标 2D 像素跨度」的头，量它跨域准不准。

为什么是这个数：单目测距 Z = f·L/s。s（2D 跨度）是图上直接可见的量，L（物体实际多大）不是。
现在的深度头把两者揉在一起自由回归，跨域时退回训练先验 -> 深度 x2.2。
若把深度改成 Z = f·L_3d/s_2d，那么【s_2d 的相对误差就是深度的相对误差】。
所以这个探针只回答一个问题：**冻住的仿真特征，能不能在真实图上把 2D 跨度量准。**

    python tools/annot_audit/probe_size2d.py [--epochs 3]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402  （它会 chdir 到 tools 并把仓库放进 sys.path）
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402

PP = 'cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml'
MAV = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
CKPT = U.M + '/sim_pp_realsize/P1/ckpt/best.pth'
STRIDE = 8


def box_2d(box9d, K):
    """3D 框 -> 投影 2D 包围盒 (u0,v0,w,h)，输入分辨率像素。"""
    c = (U.PROTO8 * box9d[3:6]) @ R.from_euler('xyz', box9d[6:9]).as_matrix().T + box9d[:3]
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv[:, 0].min(), uv[:, 1].min(), np.ptp(uv[:, 0]), np.ptp(uv[:, 1])


def extent_3d(box9d):
    """框在相机 x / y 方向上的物理跨度（正交近似），用来把 2D 跨度换回深度。"""
    c = (U.PROTO8 * box9d[3:6]) @ R.from_euler('xyz', box9d[6:9]).as_matrix().T
    return np.ptp(c[:, 0]), np.ptp(c[:, 1])


def make_target(batch, hm_h, hm_w):
    """按与 center_point_encoder 完全相同的落格方式，画 log(w2d) / log(h2d)。"""
    to_np = lambda t: (t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t))
    Ks_all = to_np(batch['intrinsic'])
    gt = to_np(batch['gt_box9d']).reshape(len(Ks_all), -1, 9)
    Ks = Ks_all.reshape(len(gt), -1, 3, 3)[:, 0]
    tgt = np.zeros((len(gt), 2, hm_h, hm_w), np.float32)
    msk = np.zeros((len(gt), 1, hm_h, hm_w), np.float32)
    for b in range(len(gt)):
        K = Ks[b]
        for g in gt[b]:
            if np.abs(g).sum() == 0 or g[2] <= 1e-6:
                continue
            uv = K @ g[:3]
            u, v = uv[0] / uv[2] / STRIDE, uv[1] / uv[2] / STRIDE
            wi, hi = int(u), int(v)
            if not (0 <= hi < hm_h and 0 <= wi < hm_w):
                continue
            _, _, w2, h2 = box_2d(g, K)
            if w2 < 1e-3 or h2 < 1e-3:
                continue
            tgt[b, 0, hi, wi] = np.log(w2)
            tgt[b, 1, hi, wi] = np.log(h2)
            msk[b, 0, hi, wi] = 1.0
    return torch.from_numpy(tgt), torch.from_numpy(msk)


def feats(model, batch, full=False):
    with torch.no_grad():
        if full:                       # 评测时走完整 forward，batch 里才有 pred_boxes9d
            batch = model(batch)
        else:
            for m in model.module_list:
                batch = m(batch)
    x = batch['features_2d']
    B, K, C, H, W = x.shape
    return x.reshape(B * K, C, H, W), batch


def build(cfg_file, training, interval, sets=None, bs=8, workers=4):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    if sets:
        cfg_from_list(sets, cfg)
    split = 'train' if training else 'test'
    cfg.DATA_CONFIG.SAMPLED_INTERVAL[split] = interval
    return build_dataloader(cfg.DATA_CONFIG, batch_size=bs, dist=False, workers=workers,
                            logger=common_utils.create_logger(), training=training), cfg


def evaluate(model, head, loader, tag):
    """在 GT 中心格上比预测 2D 跨度与真值；再用它 + 模型自己的 lwh/rot 解深度。"""
    model.eval(); head.eval()
    rw, rh, zr, pe, n = [], [], [], [], 0
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            f, batch = feats(model, batch, full=True)
            p = head(f).float().cpu().numpy()
            tn = lambda t: (t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t))
            gt = tn(batch['gt_box9d']).reshape(len(p), -1, 9)
            Ks = tn(batch['intrinsic']).reshape(len(p), -1, 3, 3)[:, 0]
            preds = batch['pred_boxes9d']
            for b in range(len(p)):
                K = Ks[b]
                fx, fy = K[0, 0], K[1, 1]
                for g in gt[b]:
                    if np.abs(g).sum() == 0 or g[2] <= 1e-6:
                        continue
                    uv = K @ g[:3]
                    u, v = uv[0] / uv[2] / STRIDE, uv[1] / uv[2] / STRIDE
                    wi, hi = int(u), int(v)
                    if not (0 <= hi < p.shape[2] and 0 <= wi < p.shape[3]):
                        continue
                    _, _, w2, h2 = box_2d(g, K)
                    if w2 < 1e-3 or h2 < 1e-3:
                        continue
                    pw, ph = float(np.exp(p[b, 0, hi, wi])), float(np.exp(p[b, 1, hi, wi]))
                    rw.append(pw / w2); rh.append(ph / h2); n += 1
                    # 用模型自己的 3D 尺寸/姿态 + 预测的 2D 跨度解深度
                    pb = preds[b]
                    if pb is None or len(pb) == 0:
                        continue
                    pb = np.asarray(pb.cpu() if torch.is_tensor(pb) else pb).reshape(-1, 9)
                    uvg = (K @ g[:3])[:2] / g[2]
                    j = int(np.argmin([np.linalg.norm((K @ q[:3])[:2] / q[2] - uvg) for q in pb]))
                    ex, ey = extent_3d(pb[j])
                    Z = (fx * ex * pw + fy * ey * ph) / max(pw * pw + ph * ph, 1e-9)
                    zr.append(Z / g[2])
                    ray = np.linalg.inv(K) @ np.array([uvg[0], uvg[1], 1.0])
                    pe.append(float(np.linalg.norm(ray / ray[2] * Z - g[:3])))
    q = lambda a: (np.median(a), np.percentile(a, 25), np.percentile(a, 75))
    mw, mh = np.median(rw), np.median(rh)
    rel = np.median(np.abs(np.array(rw) - 1))
    print('  %-30s n=%4d | 2D 宽比 %.3f 高比 %.3f | 宽相对误差中位 %.1f%% | 解出深度比 %.3f | 位置误差中位 %.3f m' % (
        tag, n, mw, mh, 100 * rel, np.median(zr) if zr else float('nan'),
        np.median(pe) if pe else float('nan')))
    return dict(n=n, w_ratio=mw, rel=rel, z_ratio=float(np.median(zr)) if zr else None,
                pos=float(np.median(pe)) if pe else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    (tr_ds, tr_loader, _), cfg = build(PP, True, 1)
    model = build_network(cfg.MODEL, tr_ds)
    a, b = model.load_params_from_file(CKPT, to_cpu=False)
    assert a == b, (a, b)
    model.cuda().eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    head = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                         nn.Conv2d(128, 2, 3, 1, 1)).cuda()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * len(tr_loader))
    print('冻结 P1 骨干，只训 2D 尺寸头：%d 轮 x %d 步' % (args.epochs, len(tr_loader)))
    t0 = time.time()
    for ep in range(args.epochs):
        head.train()
        acc, k = 0.0, 0
        for it, batch in enumerate(tr_loader):
            load_data_to_gpu(batch)
            f, batch = feats(model, batch)
            tgt, msk = make_target(batch, f.shape[2], f.shape[3])
            tgt, msk = tgt.cuda(), msk.cuda()
            if msk.sum() < 1:
                continue
            p = head(f)
            loss = (torch.abs(p - tgt) * msk).sum() / (2 * msk.sum())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            acc += float(loss.detach()); k += 1
            if it % 100 == 0:
                print('    轮 %d 步 %4d/%d  log-L1 %.4f  (%.0f s)' % (ep, it, len(tr_loader), acc / max(k, 1), time.time() - t0))
        print('  轮 %d 结束 log-L1 %.4f' % (ep, acc / max(k, 1)))

    print('\n评测（2D 宽比 = 预测跨度 / 真值跨度，1.00 最好；它的相对误差就是深度的相对误差）')
    (_, sl, _), _ = build(PP, False, 3, ['DATA_CONFIG.VAL_ZOOMS', '[2.5]'], workers=2)
    evaluate(model, head, sl, '仿真 test（域内）')
    (_, ml, _), _ = build(MAV, False, 5, None, workers=2)
    evaluate(model, head, ml, 'MAV6D 真实 test（零样本）')
    torch.save(head.state_dict(), U.TMP + '/probe_size2d_head.pth')
    print('头权重 ->', U.TMP + '/probe_size2d_head.pth')


if __name__ == '__main__':
    main()
