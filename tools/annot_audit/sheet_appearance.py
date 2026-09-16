# -*- coding: utf-8 -*-
"""同虚拟深度 Zv、同像素跨度下，把仿真与 MAV6D 的网络输入并排画出来（含 GT 框），看外观差在哪。"""
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
from scipy.spatial.transform import Rotation as R   # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
OUT = U.TMP + '/appearance_sim_vs_real.jpg'


def grab(cfg_file, interval, sets, n=6):
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    if sets:
        cfg_from_list(sets, cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=4, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    tiles = []
    for batch in loader:
        img, K = batch['image'], batch['intrinsic']
        for b in range(len(img)):
            gt = np.asarray(batch['gt_box9d'][b]).reshape(-1, 9)
            gt = gt[np.abs(gt).sum(1) > 0]
            if not len(gt):
                continue
            g = gt[0]
            _a = np.asarray(img[b]); _a = _a.reshape(-1, _a.shape[-2], _a.shape[-1])
            im = _a.transpose(1, 2, 0)
            im = np.clip((im - im.min()) / (im.ptp() + 1e-6) * 255, 0, 255).astype(np.uint8)
            im = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2BGR) if im.shape[2] >= 3 else cv2.cvtColor(im[..., 0], cv2.COLOR_GRAY2BGR)
            Km = np.asarray(K[b]).reshape(-1, 3, 3)[0]
            c = (U.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            uv = (Km @ c.T).T; uv = uv[:, :2] / uv[:, 2:3]
            f = float(np.sqrt(Km[0, 0] * Km[1, 1]))
            cu, cv_ = (Km @ g[:3])[:2] / g[2]
            half = 110
            pad = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT)
            crop = pad[int(cv_):int(cv_) + 2 * half, int(cu):int(cu) + 2 * half].copy()
            if crop.shape[0] != 2 * half or crop.shape[1] != 2 * half:
                continue
            q = lambda p: (int(p[0] - cu + half), int(p[1] - cv_ + half))
            for a_, b_ in EDGES:
                cv2.line(crop, q(uv[a_]), q(uv[b_]), (0, 220, 255), 1, cv2.LINE_AA)
            cv2.rectangle(crop, (0, 0), (2 * half, 30), (0, 0, 0), -1)
            cv2.putText(crop, 'Zv%.2f f%.0f span%.0fpx' % (g[2] * 512 / f, f, max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))),
                        (3, 20), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
            if len(tiles) >= n:
                return tiles
    return tiles


def main():
    sim = grab('cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml', 7, ['DATA_CONFIG.VAL_ZOOMS', '[3.5]'])
    real = grab('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', 97, None)
    n = min(len(sim), len(real))
    sheet = np.vstack([np.hstack(sim[:n]), np.hstack(real[:n])])
    cv2.putText(sheet, 'SIM pp_realsize (zoom3.5)', (5, 215), 0, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(sheet, 'REAL MAV6D', (5, 435), 0, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(OUT, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('->', OUT, sheet.shape)


if __name__ == '__main__':
    main()
