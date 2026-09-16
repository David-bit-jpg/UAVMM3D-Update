# -*- coding: utf-8 -*-
"""同口径比较两域「网络真正看到的目标块」的细节量。

仿真缓存是 1280x720，裁窗 z 倍后缩到 512：z<2.5 是降采样，z>2.5 是【放大】（糊）。
MAV6D 是 1920x1080 一次性降到 512x288（锐）。
而 MAV6D 的虚拟深度 Zv≈3.5 对应仿真 z≈3.3 —— 恰好落在仿真被放大变糊的那一段。
所以要在【匹配的几何工作点】上比目标块的细节，而不是比整幅。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402
from easydict import EasyDict   # noqa: E402
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402


def grab(cfg_file, interval, sets, n=200, wh=None, post=None):
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    if sets:
        cfg_from_list(sets, cfg)
    if wh:
        cfg.DATA_CONFIG.IM_RESIZE = list(wh)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    nm = np.asarray(cfg.DATA_CONFIG.NORM_MEAN, np.float32).reshape(3, 1, 1)
    ns = np.asarray(cfg.DATA_CONFIG.NORM_STD, np.float32).reshape(3, 1, 1)
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=4, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    lap, spans, got = [], [], 0
    for batch in loader:
        img, K = batch['image'], batch['intrinsic']
        for b in range(len(img)):
            gt = np.asarray(batch['gt_box9d'][b]).reshape(-1, 9)
            gt = gt[np.abs(gt).sum(1) > 0]
            if not len(gt):
                continue
            g = gt[0]
            a = np.asarray(img[b]); a = a.reshape(-1, a.shape[-2], a.shape[-1])[:3]
            a = np.clip((a * ns + nm) * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            if post:
                h0, w0 = a.shape[:2]
                a = cv2.resize(a, (int(w0 * post), int(h0 * post)), interpolation=cv2.INTER_AREA)
            Km = np.asarray(K[b]).reshape(-1, 3, 3)[0].copy()
            if post:
                Km[:2] *= post
            s_px, c3 = U.proj_extent(g, Km)
            uv = (Km @ c3.T).T; uv = uv[:, :2] / uv[:, 2:3]
            u0, v0 = uv[:, 0].min(), uv[:, 1].min()
            w, h = np.ptp(uv[:, 0]), np.ptp(uv[:, 1])
            if w < 8 or h < 8 or u0 < 0 or v0 < 0 or u0 + w >= a.shape[1] or v0 + h >= a.shape[0]:
                continue
            patch = a[int(v0):int(v0 + h) + 1, int(u0):int(u0 + w) + 1]
            if patch.size == 0:
                continue
            # 统一放到 64x64 再算细节量：比较的是「同样大的目标块里有多少高频」
            p = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
            lap.append(cv2.Laplacian(p, cv2.CV_64F).var())
            spans.append(s_px); got += 1
            if got >= n:
                return np.array(lap), np.array(spans)
    return np.array(lap), np.array(spans)


def main():
    pp = 'cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml'
    mav = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
    print('目标块细节量（统一缩到 64x64 后的拉普拉斯方差）—— 数值越小越糊')
    l, s = grab(mav, 11, None)
    print('  %-34s n=%3d 目标像素跨度中位 %5.1f | 细节量中位 %6.1f [p25 %6.1f, p75 %6.1f]' % (
        'MAV6D 真实（1920->512 降采样）', len(l), np.median(s), np.median(l), np.percentile(l, 25), np.percentile(l, 75)))
    for z in ('1.0', '2.0', '2.5', '3.3', '4.0'):
        l, s = grab(pp, 3, ['DATA_CONFIG.VAL_ZOOMS', '[%s]' % z])
        mark = ' <= 原生，无重采样' if z == '2.5' else (' (放大)' if float(z) > 2.5 else '')
        print('  %-34s n=%3d 目标像素跨度中位 %5.1f | 细节量中位 %6.1f [p25 %6.1f, p75 %6.1f]%s' % (
            '仿真 裁窗 z=%s（1280/%s->512）' % (z, z), len(l), np.median(s), np.median(l),
            np.percentile(l, 25), np.percentile(l, 75), mark))
    print('')
    print('输入改成 384x216（不放大上限 z<=3.33，Zv 能覆盖到 MAV6D 的近端）：')
    l, s = grab(mav, 11, None, post=384.0 / 512.0)
    print('  %-34s n=%3d 目标像素跨度中位 %5.1f | 细节量中位 %6.1f [p25 %6.1f, p75 %6.1f]' % (
        'MAV6D 真实（等效降到 384 宽）', len(l), np.median(s), np.median(l), np.percentile(l, 25), np.percentile(l, 75)))
    l, s = grab(pp, 3, ['DATA_CONFIG.VAL_ZOOMS', '[3.33]'], wh=[384, 216])
    print('  %-34s n=%3d 目标像素跨度中位 %5.1f | 细节量中位 %6.1f [p25 %6.1f, p75 %6.1f]' % (
        '仿真 384 输入 z=3.33（原生）', len(l), np.median(s), np.median(l), np.percentile(l, 25), np.percentile(l, 75)))


if __name__ == '__main__':
    main()
