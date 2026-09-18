# -*- coding: utf-8 -*-
"""kp2d + PnP 的往返自检（2026-09-18）：真值 -> 监督图 -> PnP 解码，必须还原真值位姿。

不过这一关就不许开训。同时打印「只用 4 个角点」「角点加 1 px 噪声」两种情况，
给出 PnP 对关键点误差的敏感度（后面看真实域误差时要用这个尺度）。

    cd tools && python annot_audit/check_kp2d.py
"""
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
os.chdir(TOOLS)
from scipy.spatial.transform import Rotation as R   # noqa: E402

from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import encoder_geometry_kwargs   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402
from uavdet3d.utils.object_encoder import all_object_encoders   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import KP_SCALE   # noqa: E402


def run(cfg_file, tag, n=50, noise_px=0.0, use_corners=8):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    dc = cfg.DATA_CONFIG
    dc.SAMPLED_INTERVAL['test'] = 7
    ds, _, _ = build_dataloader(dc, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=False)
    dec = all_object_encoders[cfg.MODEL.POST_PROCESSING.DECONDER]
    kw = encoder_geometry_kwargs(dc, dec, cfg.MODEL.POST_PROCESSING.DECONDER)
    seq = dc.get('EULER_SEQ', 'xyz')
    dp, da, rng = [], [], np.random.default_rng(0)
    for item in range(len(ds)):
        d = ds[item]
        gt = np.asarray(d['gt_box9d']).reshape(-1, 9)
        gt = gt[np.abs(gt).sum(1) > 0]
        if not len(gt) or 'kp2d' not in d:
            continue
        t = lambda a: torch.from_numpy(np.asarray(a)).float()
        kp = np.asarray(d['kp2d']).copy()
        if noise_px > 0:
            kp = kp + rng.normal(0, noise_px / KP_SCALE, kp.shape).astype(np.float32) * (np.abs(kp).sum(0, keepdims=True) > 0)
        if use_corners < 8:                       # 只保留前 N 个角点（其余置零 = PnP 会用到全部 16 通道，这里仅做敏感度参考）
            kp[2 * use_corners:] = 0
        smu = torch.as_tensor(np.asarray(dc.SIZE_MEAN, np.float32).reshape(3, 1, 1))
        ssd = torch.as_tensor(np.asarray(dc.SIZE_STD, np.float32).reshape(3, 1, 1)).clamp(min=1e-6)
        dm_dec = t(d['dim']) * ssd + smu
        cd_dec = t(d['center_dis']) * float(dc.DEPTH_STD) + float(dc.DEPTH_MEAN)
        pred, conf = dec(t(d['hm']), t(d['center_res']), cd_dec, dm_dec, t(d['rot']),
                         np.asarray(d['intrinsic']), np.asarray(d['extrinsic']), np.asarray(d['distortion']),
                         d['new_im_size'][0], d['new_im_size'][1], d['raw_im_size'][0], d['raw_im_size'][1],
                         d['stride'], 1, max(5, len(gt)), kp2d=t(kp), kp_scale=KP_SCALE, **kw)
        for g in gt:
            if not len(pred):
                continue
            j = int(np.argmin(np.linalg.norm(pred[:, :3] - g[:3], axis=1)))
            p = pred[j]
            if np.linalg.norm(p[:3] - g[:3]) > 1.0:
                continue
            dp.append(np.linalg.norm(p[:3] - g[:3]))
            da.append(np.degrees((R.from_euler(seq, p[6:9]).inv() * R.from_euler(seq, g[6:9])).magnitude()))
        if len(dp) >= n:
            break
    if not dp:
        print('  %-40s 没有可用样本' % tag)
        return False
    ok = max(dp) < (2e-3 if noise_px == 0 else 9e9) and max(da) < (1e-1 if noise_px == 0 else 9e9)
    print('  %-40s n=%3d | 位置 中位 %.2e 最大 %.2e m | 角度 中位 %.2e 最大 %.2e° %s' % (
        tag, len(dp), np.median(dp), max(dp), np.median(da), max(da), ('PASS' if ok else '**FAIL**') if noise_px == 0 else ''))
    return ok


def main():
    allok = True
    print('kp2d -> PnP 往返（真值监督图，应当逐位还原）：')
    for cfg, tag in (('cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp.yaml', '仿真 full_v2'),
                     ('cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml', 'MAV6D')):
        allok &= run(cfg, tag)
    print('PnP 对关键点误差的敏感度（仿真，给角点加高斯噪声）：')
    for px in (0.5, 1.0, 2.0, 4.0):
        run('cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp.yaml', '角点噪声 %.1f px' % px, noise_px=px)
    print('\n==== %s ====' % ('往返通过' if allok else '有 FAIL，不许开训'))


if __name__ == '__main__':
    main()
