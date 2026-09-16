# -*- coding: utf-8 -*-
"""模型侧检查：BN 的 running statistics 是在仿真上统计的，真实图分布不同会让内部激活整体偏。

用【无标签】的 MAV6D 训练图重新统计 BN（AdaBN，只过前向、不反传、不碰标签），再在 test 上评。
若指标明显改善，说明零样本失败里有一部分是 BN 统计量不匹配 —— 这是纯模型侧问题，与采什么数据无关。
"""
import argparse
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

MAV = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
STRIDE = 8


def build(split, interval, bs=8):
    cfg = EasyDict(); cfg_from_yaml_file(MAV, cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL[split] = interval
    if split == 'train':
        cfg.DATA_CONFIG.DATA_SPLIT['test'] = 'train'      # 借 test 通道读 train 划分，且不开增广
        cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=bs, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    return cfg, ds, loader


def bns(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]


def recalibrate(model, loader, max_batches):
    for m in bns(model):
        m.reset_running_stats()
        m.momentum = None            # 累积平均
        m.train()
    n = 0
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            x = batch
            for mod in model.module_list:
                x = mod(x)
            n += 1
            if n >= max_batches:
                break
    for m in bns(model):
        m.eval()
    return n


def gt_cell_stats(model, loader, cfg):
    """在 GT 中心格上读深度/尺寸头，绕开检测，无选择偏倚。"""
    MAX_DIS = float(cfg.DATA_CONFIG.MAX_DIS); MAX_SIZE = float(cfg.DATA_CONFIG.MAX_SIZE)
    zr, sz, hm_pk = [], [], []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            out = model(batch)
            pd = out['pred_center_dict']
            cd = pd['center_dis'].float().cpu().numpy(); dm = pd['dim'].float().cpu().numpy()
            hm = torch.sigmoid(pd['hm']).float().cpu().numpy()
            tn = lambda t: (t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t))
            gt = tn(batch['gt_box9d']).reshape(len(cd), -1, 9)
            Ks = tn(batch['intrinsic']).reshape(len(cd), -1, 3, 3)[:, 0]
            for bi in range(len(cd)):
                K = Ks[bi]; f_in = float(np.sqrt(K[0, 0] * K[1, 1]))
                for g in gt[bi]:
                    if np.abs(g).sum() == 0 or g[2] <= 1e-6:
                        continue
                    uv = K @ g[:3]
                    wi, hi = int(uv[0] / uv[2] / STRIDE), int(uv[1] / uv[2] / STRIDE)
                    if not (0 <= hi < cd.shape[2] and 0 <= wi < cd.shape[3]):
                        continue
                    zr.append(cd[bi, 0, hi, wi] * MAX_DIS * f_in / 512.0 / g[2])
                    sz.append(float(np.linalg.norm(dm[bi, :, hi, wi] * MAX_SIZE)))
                    hm_pk.append(float(hm[bi, 0, hi, wi]))
    return np.median(zr), np.median(sz), np.median(hm_pk), len(zr)


def evaluate(model, loader):
    recs = pose_eval.run_inference(model, loader)
    s = pose_eval.summarize(pose_eval.match_records(recs), len(recs))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, default=400, help='用多少个 batch 的无标签真实图重统计 BN')
    a = ap.parse_args()
    cfg, _, test_loader = build('test', 10)
    _, _, cal_loader = build('train', 20)
    for tag, ck in (('P1（真机尺寸）', '/sim_pp_realsize/P1/ckpt/best.pth'),
                    ('S1（旧标签 8 场景）', '/sim_indoor8_mz/S1/ckpt/best.pth')):
        model = build_network(cfg.MODEL, test_loader.dataset)
        n, t = model.load_params_from_file(U.M + ck, to_cpu=False)
        assert n == t
        model.cuda().eval()
        s0 = evaluate(model, test_loader); g0 = gt_cell_stats(model, test_loader, cfg)
        nb = recalibrate(model, cal_loader, a.batches)
        s1 = evaluate(model, test_loader); g1 = gt_cell_stats(model, test_loader, cfg)
        print('\n%s（BN 用 %d 个 batch 的无标签 MAV6D 训练图重统计）' % (tag, nb))
        hdr = ('', '2D中心px', '位置中位m', '深度中位m', '角度中位°', '折叠角°', 'ACC0.2m', 'ACC20°', 'GT格深度比', 'GT格尺寸')
        print(('  %-8s' + ' %9s' * 9) % hdr)
        for nm, s, g in (('原始', s0, g0), ('AdaBN', s1, g1)):
            print(('  %-8s' + ' %9.3f' * 4 + ' %9.1f' + ' %9.3f' * 4) % (
                nm, s['uv_median'], s['pos_median'], s['z_median'], s['ang_median'], s['ang_fold_median'],
                s['ACC_pos_0.2'], s['ACC_rot_20'], g[0], g[1]))


if __name__ == '__main__':
    main()
