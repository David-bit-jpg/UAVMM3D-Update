# -*- coding: utf-8 -*-
"""虚拟深度覆盖假设检验（不涉及真实域差）：仿真验证帧在原图 1280x720 上裁 512x288 窗口（等效长焦，f_in=640，
Zv = 0.8 Z，落进 MAV6D 的虚拟深度范围），看仿真训练的模型会不会也把深度估大。

对照：同一批帧整幅缩到 512x288（f_in=256，Zv=2Z，与训练分布一致）。
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import camera_geometry as cg, common_utils   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=240)
    args = ap.parse_args()
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/sim_indoor8.yaml', cfg)
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    model.load_params_from_file(args.ckpt, to_cpu=False)
    model.cuda().eval()
    idx = pickle.load(open('E:/mmcache/indoor8cn/test/index.pkl', 'rb'))
    rng = np.random.RandomState(0)
    picks = rng.choice(ds.valid_idx, args.n, replace=False)
    mean = np.array(cfg.DATA_CONFIG.NORM_MEAN, np.float32)[:, None, None]
    std = np.array(cfg.DATA_CONFIG.NORM_STD, np.float32)[:, None, None]
    out = {'full': [], 'crop': []}
    for i in picks:
        m = idx['metas'][int(i)]
        if len(m['boxes9d']) != 1:
            continue
        img = cv2.imread(os.path.join('D:/data_collect', m['seq'], 'images_rgb', m['frame']))
        if img is None:
            continue
        K = np.asarray(m['K_raw'], np.float64)
        b = m['boxes9d'][0].astype(np.float64)
        u = K @ b[:3]
        cu, cv_ = u[0] / u[2], u[1] / u[2]
        views = {'full': (cv2.resize(img, (512, 288), interpolation=cv2.INTER_AREA), cg.scale_K(K, 512 / 1280, 288 / 720))}
        x0 = int(np.clip(round(cu - 256 + rng.uniform(-120, 120)), 0, 1280 - 512))
        y0 = int(np.clip(round(cv_ - 144 + rng.uniform(-60, 60)), 0, 720 - 288))
        if not (x0 + 8 <= cu < x0 + 504 and y0 + 8 <= cv_ < y0 + 280):
            continue
        views['crop'] = (np.ascontiguousarray(img[y0:y0 + 288, x0:x0 + 512]), cg.translate_K(K, -x0, -y0))
        for name, (im, Kv) in views.items():
            x = (im.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
            dd = {'image': x[None], 'gt_box9d': b[None].astype(np.float32), 'gt_name': np.array(['drone']),
                  'intrinsic': np.array([Kv]), 'extrinsic': np.array([np.eye(4)]), 'distortion': np.zeros((1, 5)),
                  'raw_im_size': np.array([512, 288]), 'new_im_size': np.array([512, 288]),
                  'obj_size': np.ones((1, 3)), 'stride': 8, 'scene_id': 's', 'seq_id': 's', 'frame_id': 'f'}
            dd = ds.data_pre_processor(dd)
            batch = ds.collate_batch([dd])
            load_data_to_gpu(batch)
            with torch.no_grad():
                o = model(batch)
            pred, conf = o['pred_boxes9d'][0], o['confidence'][0]
            if len(pred) == 0:
                continue
            p = np.asarray(pred)[int(np.argmax(conf))]
            f = cg.focal(Kv)
            up = Kv @ p[:3]
            out[name].append((p[2] / b[2], b[2] * 512 / f, p[2] * 512 / f, np.linalg.norm(up[:2] / up[2] - (Kv @ b[:3])[:2] / b[2]),
                              f * max(b[3:6]) / b[2]))
    for name, v in out.items():
        v = np.array(v)
        print('%-4s n=%d | f_in %s | 真值 Zv 中位 %.2f (p10 %.2f p90 %.2f) | 预测/真值深度 中位 %.3f (p10 %.3f p90 %.3f) | '
              '2D 中心误差 中位 %.1f px | 表观长边 中位 %.0f px'
              % (name, len(v), '256' if name == 'full' else '640', np.median(v[:, 1]), np.percentile(v[:, 1], 10),
                 np.percentile(v[:, 1], 90), np.median(v[:, 0]), np.percentile(v[:, 0], 10), np.percentile(v[:, 0], 90),
                 np.median(v[:, 3]), np.median(v[:, 4])))
        lo = v[:, 1] < 5
        if lo.any():
            print('      其中真值 Zv < 5（训练里罕见的近距离）%d 个：预测/真值深度 中位 %.3f' % (lo.sum(), np.median(v[lo, 0])))


if __name__ == '__main__':
    main()
