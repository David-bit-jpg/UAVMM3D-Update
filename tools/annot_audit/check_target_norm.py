# -*- coding: utf-8 -*-
"""自洽标准化的往返自检：GT -> 标准化监督图 -> 解码，必须逐位还原 GT。

同时查两件容易写岔的事：
  1. 前景掩码。标准化后 center_dis 真值会有负数，若还拿 `> 0` 当前景指示，近距离目标会被当背景丢掉。
     这里统计有多少目标的标准化真值是负的 —— 如果不为零，就证明旧掩码一定会漏。
  2. 解码常数。解码必须用【训练域】的 mu/sigma；在 MAV6D 上评测时若误用 MAV6D 自己的统计量，
     深度会整体错一个比例。这里用训练域常数解 MAV6D 的监督图，比对是否还原。

    python tools/annot_audit/check_target_norm.py
"""
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402
from scipy.spatial.transform import Rotation as R   # noqa: E402
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import encoder_geometry_kwargs, denormalize_regression   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402
from uavdet3d.utils.object_encoder import all_object_encoders   # noqa: E402


def run(cfg_file, tag, n=60, sets=None):
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    if sets:
        cfg_from_list(sets, cfg)
    dc = cfg.DATA_CONFIG
    mode = str(dc.get('TARGET_NORM', 'maxdis'))
    dc.SAMPLED_INTERVAL['test'] = 7
    ds, loader, _ = build_dataloader(dc, batch_size=1, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    dec = all_object_encoders[cfg.MODEL.POST_PROCESSING.DECONDER]
    kw = encoder_geometry_kwargs(dc, dec, cfg.MODEL.POST_PROCESSING.DECONDER)
    seq = dc.get('EULER_SEQ', 'xyz')
    neg, tot, dp, da, dsz = 0, 0, [], [], []
    for item in range(len(ds)):
        d = ds[item]
        gt = np.asarray(d['gt_box9d']).reshape(-1, 9)
        gt = gt[np.abs(gt).sum(1) > 0]
        if not len(gt):
            continue
        cd = np.asarray(d['center_dis']); dm = np.asarray(d['dim'])
        fg = np.asarray(d['fg_mask']) > 0 if 'fg_mask' in d else np.abs(cd) > 0
        neg += int((cd[fg[:, :1].repeat(cd.shape[1], 1) if fg.shape[1] == 1 else fg] < 0).sum())
        tot += int(fg.sum())
        t = lambda a: torch.from_numpy(np.asarray(a)).float()
        # 与 CenterDet.post_processing 共用同一个函数（DEPTH_TARGET / SIZE_SOURCE 也在里面）
        cd_dec, dm_dec = denormalize_regression(dc, cfg.MODEL.POST_PROCESSING, t(cd), t(dm))
        pred, conf = dec(t(d['hm']), t(d['center_res']), cd_dec, dm_dec, t(d['rot']),
                         np.asarray(d['intrinsic']), np.asarray(d['extrinsic']), np.asarray(d['distortion']),
                         d['new_im_size'][0], d['new_im_size'][1], d['raw_im_size'][0], d['raw_im_size'][1],
                         d['stride'], 1, max(5, len(gt)), **kw)
        for g in gt:
            if not len(pred):
                continue
            j = int(np.argmin(np.linalg.norm(pred[:, :3] - g[:3], axis=1)))
            p = pred[j]
            if np.linalg.norm(p[:3] - g[:3]) > 1.0:
                continue
            dp.append(np.linalg.norm(p[:3] - g[:3]))
            da.append(np.degrees((R.from_euler(seq, p[6:9]).inv() * R.from_euler(seq, g[6:9])).magnitude()))
            dsz.append(np.abs(p[3:6] - g[3:6]).max())
        if len(dp) >= n:
            break
    ok = dp and max(dp) < 1e-4 and max(da) < 1e-3 and max(dsz) < 1e-4
    print('  %-34s %-8s/%-9s n=%3d | 位置 %.2e m 角度 %.2e° 尺寸 %.2e m | %s' % (
        tag, mode, dc.get('DEPTH_TARGET', 'value'), len(dp), max(dp) if dp else -1, max(da) if da else -1, max(dsz) if dsz else -1,
        'PASS' if ok else '**FAIL**'))
    if mode == 'standard':
        print('        标准化后真值为负的前景格子: %d / %d (%.1f%%) —— 旧的 `center_dis > 0` 掩码会把这些漏掉' % (
            neg, tot, 100.0 * neg / max(tot, 1)))
    return ok


def main():
    allok = True
    K = ['MODEL.POST_PROCESSING.SIZE_SOURCE', 'known']
    for cfg, tag, sets in (('cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml', '仿真 pp_realsize（老路径）', None),
                           ('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', 'MAV6D（老路径）', None),
                           ('cfgs/models/uavdet_3d/camnorm/sim_pp_std.yaml', '仿真 pp_std（标准化）', None),
                           ('cfgs/models/uavdet_3d/camnorm/mav6d_std_pp.yaml', 'MAV6D + 训练域常数（评测口径）', None),
                           ('cfgs/models/uavdet_3d/camnorm/sim_pp_ratio.yaml', '仿真 pp_ratio（机身倍数）', None),
                           ('cfgs/models/uavdet_3d/camnorm/mav6d_ratio_pp.yaml', 'MAV6D 机身倍数 + 预测尺寸', None),
                           ('cfgs/models/uavdet_3d/camnorm/mav6d_ratio_pp.yaml', 'MAV6D 机身倍数 + 已知尺寸', K)):
        allok &= run(cfg, tag, sets=sets)
    print('\n==== %s ====' % ('全部通过' if allok else '有 FAIL，不许开训'))


if __name__ == '__main__':
    main()
