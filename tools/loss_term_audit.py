# -*- coding: utf-8 -*-
"""逐项拆开 CenterHead 的总损失，看五个头各自贡献多少。

get_loss() 把 hm / center_res / center_dis / dim / rot 五项【等权相加】，
但它们的量纲和量级完全不同。这个脚本在真实 batch 上把每一项单独算出来，
用来判断「旋转学不好」是不是因为它在总损失里根本没有分量。

用法：
    python tools/loss_term_audit.py --ckpt <权重>.pth --batches 20
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

from uavdet3d.config import cfg_from_yaml_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.model import build_network, load_data_to_gpu
from uavdet3d.utils import common_utils

CFG = 'cfgs/models/uavdet_3d/mav6d/centerdet.yaml'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default=CFG)
    ap.add_argument('--data-path', default='E:/MAV6D')
    ap.add_argument('--batches', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()

    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(args.cfg, cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.data_path

    # 要拿到 gt_center_dict 必须走训练分支，所以 training=True
    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=args.batch_size,
                                     dist=False, workers=0, logger=logger, training=True)
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    model.cuda().train()

    head = model.dense_head_2d
    acc = {k: [] for k in head.head_keys}
    npos = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            load_data_to_gpu(batch)
            model(batch)
            pd = head.forward_loss_dict['pred_center_dict']
            gd = head.forward_loss_dict['gt_center_dict']
            for k in head.head_keys:
                p, g = pd[k].reshape(-1), gd[k].reshape(-1)
                if k == 'hm':
                    l = F.binary_cross_entropy(torch.sigmoid(p), g, reduction='none')
                    m = g > 0
                    acc[k].append(float(l.sum() / (m.sum() + 1)))
                    npos.append(int(m.sum()))
                else:
                    m = torch.abs(g) > 0
                    if m.sum() == 0:
                        continue
                    acc[k].append(float(torch.abs(g[m] - p[m]).mean()))

    print('\n%d 个 batch (batch_size=%d)，正样本像素数中位 %d' %
          (args.batches, args.batch_size, int(np.median(npos)) if npos else -1))
    print('\n%-12s %10s %10s %10s' % ('损失项', '均值', '标准差', '占总损失'))
    print('-' * 46)
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    tot = sum(means.values())
    for k in head.head_keys:
        print('%-12s %10.4f %10.4f %9.1f%%'
              % (k, means[k], float(np.std(acc[k])) if acc[k] else 0.0,
                 100.0 * means[k] / max(tot, 1e-9)))
    print('-' * 46)
    print('%-12s %10.4f' % ('合计', tot))
    print('\n注意 get_loss() 是【等权相加】，没有任何 loss weight。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
