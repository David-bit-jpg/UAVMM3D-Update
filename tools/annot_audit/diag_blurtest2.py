# -*- coding: utf-8 -*-
"""严格版因果检验：直接在 GT 中心格上读深度头/尺寸头的输出，绕开检测，无选择偏倚（每帧都计入）。"""
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_units as U   # noqa: E402
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402

STRIDE = 8


def degrader(real_pp, d):
    def f(dd):
        if d > 1.0:
            img = dd['image']
            a = img[0].transpose(1, 2, 0)
            H, W = a.shape[:2]
            s = cv2.resize(a, (max(8, int(W / d)), max(8, int(H / d))), interpolation=cv2.INTER_AREA)
            dd['image'] = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[None].astype(img.dtype)
        return real_pp(dd)
    return f


def run(cfg_file, ckpt, d, interval, sets=None):
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    if sets:
        from uavdet3d.config import cfg_from_list
        cfg_from_list(sets, cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    ds.data_pre_processor = degrader(ds.data_pre_processor, d)
    model = build_network(cfg.MODEL, ds)
    a, b = model.load_params_from_file(ckpt, to_cpu=False)
    assert a == b
    model.cuda().eval()
    MAX_DIS = float(cfg.DATA_CONFIG.MAX_DIS); MAX_SIZE = float(cfg.DATA_CONFIG.MAX_SIZE)
    zr, sz, hm_pk = [], [], []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            out = model(batch)
            pd = out['pred_center_dict']
            cd = pd['center_dis'].float().cpu().numpy()
            dm = pd['dim'].float().cpu().numpy()
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
                    Zv = cd[bi, 0, hi, wi] * MAX_DIS
                    Z = Zv * f_in / 512.0
                    zr.append(Z / g[2])
                    sz.append(float(np.linalg.norm(dm[bi, :, hi, wi] * MAX_SIZE)))
                    hm_pk.append(float(hm[bi, 0, hi, wi]))
    return dict(n=len(zr), zr=float(np.median(zr)), sz=float(np.median(sz)), hm=float(np.median(hm_pk)))


def main():
    mav = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
    pp = 'cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml'
    for tag, ck, gts in (('P1（真机尺寸，GT 对角线 0.533）', '/sim_pp_realsize/P1/ckpt/best.pth', 0.533),
                         ('S1（旧标签，GT 对角线 0.533）', '/sim_indoor8_mz/S1/ckpt/best.pth', 0.533)):
        print('\n%s · MAV6D 真实 test，在 GT 中心格上读头的输出（n = 全部 GT，无筛选）' % tag)
        print('  %-30s %6s %9s %11s %9s' % ('', 'n', '深度比', '预测对角线', 'hm 响应'))
        for d in (1.0, 1.5, 2.0, 3.0, 4.0):
            r = run(mav, U.M + ck, d, 10)
            print('  %-30s %6d %9.3f %11.3f %9.3f' % ('降级 d=%.1f' % d, r['n'], r['zr'], r['sz'], r['hm']))
    r = run(pp, U.M + '/sim_pp_realsize/P1/ckpt/best.pth', 1.0, 3, ['DATA_CONFIG.VAL_ZOOMS', '[3.3]'])
    print('\n  参照：P1 在自己的仿真 test（z=3.3，域内）n=%d 深度比 %.3f 预测对角线 %.3f hm 响应 %.3f' % (
        r['n'], r['zr'], r['sz'], r['hm']))


if __name__ == '__main__':
    main()
