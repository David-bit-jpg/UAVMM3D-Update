# -*- coding: utf-8 -*-
"""自标定稳不稳（2026-09-19）：标定值对 样本量 / 置信度门限 / split 的敏感性。

起因：三个权重里 MIX2 在 val 上标定出对角 0.387 m（真值 0.533），位置从 0.331 退到 0.699 m，
而 MIX1 / S2 标到 0.52 左右、追平「给真值尺寸」。先分清这是样本量不够、低质检测混进来，
还是这个权重本身在 val 那批图上就偏。

    python annot_audit/calib_stability.py --ckpt <权重> [--splits val,test]
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(TOOLS)
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402

CFG = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'
TRUE_DIAG = float(np.linalg.norm([0.34, 0.34, 0.23]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--splits', default='val,test')
    ap.add_argument('--intervals', default='2,8,40')
    ap.add_argument('--thrs', default='0.2,0.4,0.6')
    a = ap.parse_args()
    cfg = EasyDict()
    cfg_from_yaml_file(CFG, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred',
                   'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width',
                   'MODEL.POST_PROCESSING.MAX_OBJ', '1'], cfg)
    model = None
    for split in a.splits.split(','):
        for iv in [int(x) for x in a.intervals.split(',')]:
            dc = copy.deepcopy(cfg.DATA_CONFIG)
            dc.DATA_SPLIT['test'] = split
            dc.SAMPLED_INTERVAL['test'] = iv
            ds, loader, _ = build_dataloader(dc, batch_size=8, dist=False, workers=0,
                                             logger=common_utils.create_logger(), training=False)
            if model is None:
                model = build_network(cfg.MODEL, ds)
                n, t = model.load_params_from_file(a.ckpt, to_cpu=False)
                assert n == t
                model.cuda().eval()
            rows = []
            with torch.no_grad():
                for batch in loader:
                    load_data_to_gpu(batch)
                    res = model(batch)
                    for b in range(res['batch_size']):
                        p = np.asarray(res['pred_boxes9d'][b]).reshape(-1, 10)
                        c = np.asarray(res['confidence'][b]).reshape(-1)
                        if not len(c) or p[0, 2] <= 0:
                            continue
                        rows.append(np.concatenate([np.abs(p[0, 3:6]), [c[0]]]))
            if not rows:
                print('  %-5s 间隔 %-3d 无检测' % (split, iv))
                continue
            arr = np.asarray(rows)
            for thr in [float(x) for x in a.thrs.split(',')]:
                m = arr[arr[:, 3] >= thr]
                if len(m) < 5:
                    print('  %-5s 间隔 %-3d conf>=%.1f  样本不足（%d）' % (split, iv, thr, len(m)))
                    continue
                med = np.median(m[:, :3], axis=0)
                d = float(np.linalg.norm(med))
                print('  %-5s 间隔 %-3d conf>=%.1f n=%4d | lwh %s | 对角 %.3f（x%.2f）' % (
                    split, iv, thr, len(m), np.round(med, 3).tolist(), d, d / TRUE_DIAG), flush=True)


if __name__ == '__main__':
    main()
