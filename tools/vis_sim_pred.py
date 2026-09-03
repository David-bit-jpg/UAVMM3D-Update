# -*- coding: utf-8 -*-
"""仿真域（LAAM6D / near 子集）上的预测可视化：GT(绿) vs 预测(红)。

仓库里 generate_prediction_dicts 内部有一大段被注释掉的可视化代码，
依赖一堆实例状态、不好单独用。这里做成独立脚本：跑一遍推理，
把 gt_boxes / pred_boxes（都是世界系的 9 点框）投影回 RGB 图上。

坐标链路：世界系 9 点 --xyz_to_uv(intrinsic, extrinsic)--> 像素
（xyz_to_uv 内部处理了 CARLA <-> OpenCV 的坐标轴置换）

用法：
    python tools/vis_sim_pred.py --ckpt <权重>.pth --num 6
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict

from uavdet3d.config import cfg_from_yaml_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.datasets.laam6d.dataset_utils import xyz_to_uv
from uavdet3d.model import build_network, load_data_to_gpu
from uavdet3d.utils import common_utils

# 9 点框的连线：0 是中心点，1..8 是角点
EDGES = [(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5),
         (1, 5), (2, 6), (3, 7), (4, 8)]


def draw_box9pts(img, pts_world, K, E, dist, color, thickness=2):
    uv = xyz_to_uv(pts_world, img.shape[1], img.shape[0],
                   intrinsic_mat=K, extrinsic_mat=E, distortion_matrix=dist,
                   return_average=False)
    uv = np.asarray(uv).reshape(-1, 2)
    if len(uv) < 9 or not np.all(np.isfinite(uv)):
        return img
    uv = uv.astype(int)
    h, w = img.shape[:2]
    if np.any(np.abs(uv) > 10 * max(h, w)):     # 投到极远处的就不画了
        return img
    for a, b in EDGES:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, thickness)
    cv2.circle(img, tuple(uv[0]), 6, color, -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='cfgs/models/uavdet_3d/laam6d/centerdet_rgb_near15.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num', type=int, default=6)
    ap.add_argument('--out', default='../output/vis_sim_pred')
    ap.add_argument('--score-thresh', type=float, default=0.3)
    ap.add_argument('--zoom', type=int, default=110,
                    help='围绕 GT 中心额外裁一张放大图的半径(像素)，0 表示不裁')
    args = ap.parse_args()

    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(args.cfg, cfg)

    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False,
                                     workers=0, logger=logger, training=False)
    print('测试帧数:', len(ds))

    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= args.num:
                break
            load_data_to_gpu(batch)
            batch = model(batch)
            annos = ds.generate_prediction_dicts(batch, None)
            for a in annos:
                if n >= args.num:
                    break
                seq_id, frame_id = a['seq_id'], a['frame_id']
                ip = os.path.join(str(ds.root_path), seq_id, 'images_rgb', frame_id)
                img = cv2.imread(ip)
                if img is None:
                    continue

                def to_np(x):
                    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

                K, E, D = to_np(a['intrinsic']), to_np(a['extrinsic']), to_np(a['distortion'])
                if K.ndim == 3:
                    K, E, D = K[0], E[0], D[0]

                gt = to_np(a['gt_boxes'])
                pr = to_np(a['pred_boxes'])
                conf = to_np(a['confidence']).reshape(-1)

                for g in gt:
                    img = draw_box9pts(img, g, K, E, D, (0, 255, 0), 2)
                kept = 0
                for i, p in enumerate(pr):
                    if i < len(conf) and conf[i] < args.score_thresh:
                        continue
                    img = draw_box9pts(img, p, K, E, D, (0, 0, 255), 2)
                    kept += 1

                cv2.putText(img, 'SIM  GT(green)=%d  Pred(red)=%d  %s'
                            % (len(gt), kept, seq_id.split('/')[0]),
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
                op = os.path.join(args.out, '%02d_%s_%s' % (n, seq_id.replace('/', '_')[:40], frame_id))
                cv2.imwrite(op, img)

                # 仿真里目标只有几十像素，全图看不清 —— 额外存一张围绕 GT 的放大裁剪
                if args.zoom and len(gt) > 0:
                    cs = xyz_to_uv(np.asarray([g[0] for g in gt]), img.shape[1], img.shape[0],
                                   intrinsic_mat=K, extrinsic_mat=E, distortion_matrix=D,
                                   return_average=False)
                    cs = np.asarray(cs).reshape(-1, 2)
                    cs = cs[np.all(np.isfinite(cs), axis=1)]
                    if len(cs):
                        cx, cy = int(np.mean(cs[:, 0])), int(np.mean(cs[:, 1]))
                        r = args.zoom
                        x0, y0 = max(0, cx - r), max(0, cy - r)
                        x1, y1 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)
                        crop = img[y0:y1, x0:x1]
                        if crop.size:
                            crop = cv2.resize(crop, None, fx=4, fy=4,
                                              interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(op.replace('.png', '_zoom.png').replace('.jpg', '_zoom.jpg'), crop)
                print('写出', op)
                n += 1
    print('共 %d 张 -> %s' % (n, args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
