# -*- coding: utf-8 -*-
"""把零样本深度偏差拆成「BN 统计量不匹配」和「细节/分辨率差」两部分，看能否合起来解释全部。"""
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402
import diag_adabn as A   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402


def degrader(real_pp, d):
    def f(dd):
        if d > 1.0:
            img = dd['image']; a = img[0].transpose(1, 2, 0); H, W = a.shape[:2]
            s = cv2.resize(a, (max(8, int(W / d)), max(8, int(H / d))), interpolation=cv2.INTER_AREA)
            dd['image'] = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[None].astype(img.dtype)
        return real_pp(dd)
    return f


def main():
    cfg, ds_t, test_loader = A.build('test', 10)
    _, ds_c, cal_loader = A.build('train', 20)
    base_t, base_c = test_loader.dataset.data_pre_processor, cal_loader.dataset.data_pre_processor
    for tag, ck in (('P1', '/sim_pp_realsize/P1/ckpt/best.pth'), ('S1', '/sim_indoor8_mz/S1/ckpt/best.pth')):
        print('\n%s · MAV6D（GT 中心格读深度头，n = 全部 GT）' % tag)
        print('  %-26s %10s %10s %10s' % ('', '深度比', '预测尺寸', 'hm响应'))
        for adabn in (False, True):
            for d in (1.0, 2.0):
                test_loader.dataset.data_pre_processor = degrader(base_t, d)
                cal_loader.dataset.data_pre_processor = degrader(base_c, d)
                model = build_network(cfg.MODEL, ds_t)
                n, t = model.load_params_from_file(U.M + ck, to_cpu=False)
                assert n == t
                model.cuda().eval()
                if adabn:
                    A.recalibrate(model, cal_loader, 400)
                zr, sz, hm, n_ = A.gt_cell_stats(model, test_loader, cfg)
                print('  %-26s %10.3f %10.3f %10.3f' % (
                    '%s + %s' % ('AdaBN' if adabn else '原始BN', '原图' if d == 1 else '降级x%.0f' % d), zr, sz, hm))
        test_loader.dataset.data_pre_processor = base_t
        cal_loader.dataset.data_pre_processor = base_c


if __name__ == '__main__':
    main()
