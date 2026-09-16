# -*- coding: utf-8 -*-
"""训练吞吐瓶颈诊断：数据集单样本耗时（含增广 + 编码）vs 网络前后向耗时（float32 / bf16 AMP）。"""
import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402


def main():
    cfg = EasyDict()
    cfg_from_yaml_file(sys.argv[1] if len(sys.argv) > 1 else 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml', cfg)
    logger = common_utils.create_logger()
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=logger, training=True)
    # 1) 单样本
    idx = np.random.RandomState(0).choice(len(ds), 64, replace=False)
    for i in idx[:4]:
        ds[int(i)]
    t0 = time.time()
    items = [ds[int(i)] for i in idx]
    t_item = (time.time() - t0) / len(idx)
    t0 = time.time()
    batches = [ds.collate_batch(items[k:k + 8]) for k in range(0, 64, 8)]
    t_coll = (time.time() - t0) / len(batches)
    print('数据：单样本 %.1f ms（2 个 worker 上限约 %.0f 样本/s = %.1f it/s）；collate %.1f ms/batch'
          % (1000 * t_item, 2 / t_item, 2 / t_item / 8, 1000 * t_coll), flush=True)
    # 2) 网络
    model = build_network(cfg.MODEL, ds).cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for amp in (False, True):
        for bench in (False, True):
            torch.backends.cudnn.benchmark = bench
            torch.backends.cudnn.deterministic = not bench
            for w in range(3):
                b = dict(batches[w % len(batches)])
                load_data_to_gpu(b)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    loss = model(b)['loss']
                loss.backward()
                opt.step()
                opt.zero_grad()
            torch.cuda.synchronize()
            n = 12
            t0 = time.time()
            for k in range(n):
                b = dict(batches[k % len(batches)])
                load_data_to_gpu(b)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    loss = model(b)['loss']
                loss.backward()
                opt.step()
                opt.zero_grad()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / n
            print('网络：amp=%s cudnn.benchmark=%s  %.1f ms/iter（%.1f it/s）loss %.3f'
                  % (amp, bench, 1000 * dt, 1 / dt, float(loss)), flush=True)


if __name__ == '__main__':
    main()
