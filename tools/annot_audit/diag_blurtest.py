# -*- coding: utf-8 -*-
"""因果检验：把 MAV6D 输入降级到仿真的真实采样分辨率，看深度偏差是否消失。

仿真在 1280x720 渲染，裁 1280/z 再放大到 512 —— 目标的【真实采样像素】只有 输入跨度/z。
MAV6D 从 1920 一次降到 512，目标真实采样像素 = 输入跨度 x 1920/512 = 3.75 倍输入跨度。
若「认不出机型 -> 尺寸先验 -> 深度 x2.2」的根因是分辨率/细节差，
那么把 MAV6D 先降到 512/d 再升回 512（d = 模拟的降级倍数），深度比应当朝 1 靠。
"""
import os
import sys

import cv2
import numpy as np
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402


def degrader(real_pp, d):
    def f(dd):
        if d > 1.0:
            img = dd['image']
            a = img[0].transpose(1, 2, 0)
            H, W = a.shape[:2]
            small = cv2.resize(a, (max(8, int(W / d)), max(8, int(H / d))), interpolation=cv2.INTER_AREA)
            back = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
            dd['image'] = back.transpose(2, 0, 1)[None].astype(img.dtype)
        return real_pp(dd)
    return f


def run(ckpt, d, interval=10):
    cfg = EasyDict(); cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    ds.data_pre_processor = degrader(ds.data_pre_processor, d)
    model = build_network(cfg.MODEL, ds)
    a, b = model.load_params_from_file(ckpt, to_cpu=False)
    assert a == b
    model.cuda().eval()
    recs = pose_eval.run_inference(model, loader)
    rows = []
    for r in recs:
        K = r['K']
        for g in r['gt']:
            if not len(r['pred']):
                continue
            uvg = (K @ g[:3])[:2] / g[2]
            dd = [np.linalg.norm((K @ p[:3])[:2] / p[2] - uvg) for p in r['pred']]
            j = int(np.argmin(dd))
            s, _ = U.proj_extent(g, K)
            rows.append((dd[j], s, r['pred'][j][2] / g[2], np.linalg.norm(r['pred'][j][3:6]),
                         float(np.linalg.norm(r['pred'][j][:3] - g[:3]))))
    a_ = np.array(rows)
    ok = a_[:, 0] < 0.5 * a_[:, 1]
    return dict(found=ok.mean(), zr=np.median(a_[ok, 2]), size=np.median(a_[ok, 3]),
                pos=np.median(a_[ok, 4]), n=int(ok.sum()))


def main():
    for tag, ck in (('P1（真机尺寸，单场景）', '/sim_pp_realsize/P1/ckpt/best.pth'),
                    ('S1（旧标签，8 场景）', '/sim_indoor8_mz/S1/ckpt/best.pth')):
        print('\n%s · MAV6D，逐级降级真实图（d=1 原样）' % tag)
        print('  %-28s %8s %9s %9s %9s' % ('降级倍数', '找到率', '深度比', '预测尺寸', '位置误差m'))
        for d in (1.0, 2.0, 3.0, 4.0, 5.0):
            r = run(U.M + ck, d)
            print('  %-28s %8.3f %9.3f %9.3f %9.3f' % (
                'd=%.1f（目标真实像素 /%.1f）' % (d, d), r['found'], r['zr'], r['size'], r['pos']))


if __name__ == '__main__':
    main()
