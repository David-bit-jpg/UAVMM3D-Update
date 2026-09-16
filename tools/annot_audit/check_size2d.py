# -*- coding: utf-8 -*-
"""size2d 头的自检：GT -> 监督图 -> 【用 size2d 几何解深度】的解码，必须还原回 GT。

若这条往返不成立，训练出来的东西一定是错的。同时对比一下不用 size2d（原来的 center_dis）的往返。
"""
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402
from uavdet3d.utils.object_encoder import all_object_encoders   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import encoder_geometry_kwargs   # noqa: E402
from scipy.spatial.transform import Rotation as R   # noqa: E402


def run(cfg_file, use_size2d, n=60):
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    cfg_from_list(['DATA_CONFIG.MAKE_SIZE2D', 'True'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = 7
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    dec = all_object_encoders[cfg.MODEL.POST_PROCESSING.DECONDER]
    kw = encoder_geometry_kwargs(cfg.DATA_CONFIG, dec, cfg.MODEL.POST_PROCESSING.DECONDER)
    MAX_DIS = float(cfg.DATA_CONFIG.MAX_DIS); MAX_SIZE = float(cfg.DATA_CONFIG.MAX_SIZE)
    seq = cfg.DATA_CONFIG.get('EULER_SEQ', 'xyz')
    dp, da, ds_, got, miss = [], [], [], 0, 0
    for item in range(len(ds)):
        d = ds[item]
        gt = np.asarray(d['gt_box9d']).reshape(-1, 9)
        gt = gt[np.abs(gt).sum(1) > 0]
        if not len(gt):
            continue
        t = lambda a: torch.from_numpy(np.asarray(a)).float()
        kws = dict(kw)
        if use_size2d:
            kws['size2d'] = t(d['size2d'])
        pred, conf = dec(t(d['hm']), t(d['center_res']), t(d['center_dis']) * MAX_DIS,
                         t(d['dim']) * MAX_SIZE, t(d['rot']),
                         np.asarray(d['intrinsic']), np.asarray(d['extrinsic']), np.asarray(d['distortion']),
                         d['new_im_size'][0], d['new_im_size'][1], d['raw_im_size'][0], d['raw_im_size'][1],
                         d['stride'], 1, max(5, len(gt)), **kws)
        for g in gt:
            if not len(pred):
                miss += 1; continue
            j = int(np.argmin(np.linalg.norm(pred[:, :3] - g[:3], axis=1)))
            p = pred[j]
            if np.linalg.norm(p[:3] - g[:3]) > 1.0:
                miss += 1; continue
            dp.append(np.linalg.norm(p[:3] - g[:3]))
            da.append(np.degrees((R.from_euler(seq, p[6:9]).inv() * R.from_euler(seq, g[6:9])).magnitude()))
            ds_.append(np.abs(p[3:6] - g[3:6]).max())
            got += 1
        if got >= n:
            break
    tag = '用 size2d 几何解深度' if use_size2d else '用 center_dis（原路径）'
    print('  %-26s n=%3d 漏 %d | 位置最大误差 %.2e m  角度 %.2e°  尺寸 %.2e m' % (
        tag, got, miss, max(dp) if dp else -1, max(da) if da else -1, max(ds_) if ds_ else -1))


def main():
    for cfg_file, nm in (('cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml', '仿真 pp_realsize'),
                         ('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', '真实 MAV6D')):
        print('\n%s：GT -> 监督图 -> 解码 往返' % nm)
        run(cfg_file, False)
        run(cfg_file, True)


if __name__ == '__main__':
    main()
