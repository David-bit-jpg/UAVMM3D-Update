# -*- coding: utf-8 -*-
"""在【同一批 MAV6D 测试帧】上并排比较多个权重的预测：GT 绿框 + 每个模型一个颜色的预测框，
下方标注每个模型的位置/角度误差。用于第五版各臂的定性对比。

    D:/Miniconda3/envs/city/python.exe tools/vis_compare_mav6d.py \
        --ckpts "Cn_p05=<path>,S1MT_p05=<path>,ST_S1MT_p05=<path>" --num 6 --out output/vis_compare
帧的选择：测试集里均匀取 --num 帧（跳过 GT 深度极端的帧），所有模型用同一批帧。
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'E:/Open3DUAVDet')
from eval_on_mav6d import MAV6D_CFG, draw_box, infer_hm_channels  # noqa: E402
from uavdet3d.config import cfg_from_yaml_file  # noqa: E402
from uavdet3d.datasets import build_dataloader  # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu  # noqa: E402
from uavdet3d.utils import common_utils  # noqa: E402

COLORS = [(60, 200, 255), (255, 120, 60), (200, 80, 255), (80, 255, 160)]   # BGR，按 --ckpts 顺序


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', required=True, help='tag=path,tag=path,...')
    ap.add_argument('--num', type=int, default=6)
    ap.add_argument('--out', default='output/vis_compare')
    ap.add_argument('--data-path', default='E:/MAV6D')
    ap.add_argument('--crop', type=int, default=520, help='围绕目标裁一个方块（0 = 整帧）')
    args = ap.parse_args()
    models = [s.split('=', 1) for s in args.ckpts.split(',')]
    os.makedirs(args.out, exist_ok=True)
    logger = common_utils.create_logger()

    cfg = EasyDict()
    cfg_from_yaml_file(MAV6D_CFG, cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.data_path
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=1, dist=False,
                                workers=0, logger=logger, training=False)
    # 测试集按 CLASS_NAMES 顺序拼接（mavic2 在前、phantom4 在后），每类各均匀取一半，保证两类都有
    by_cls = {}
    for i, info in enumerate(ds.infos):
        by_cls.setdefault(info['cls_name'], []).append(i)
    picks = []
    per = max(1, args.num // max(len(by_cls), 1))
    for c in sorted(by_cls):
        ii = by_cls[c]
        picks += [int(ii[j]) for j in np.linspace(0, len(ii) - 1, per).astype(int)]
    picks = sorted(set(picks))[:args.num]
    print('选帧 %s' % picks)

    # 每个模型跑一遍这些帧
    preds = {}
    for tag, ck in models:
        cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.hm.out_channels = infer_hm_channels(ck)
        model = build_network(model_cfg=cfg.MODEL, dataset=ds)
        model.load_params_from_file(filename=ck, to_cpu=False, logger=logger)
        model.cuda().eval()
        out = {}
        with torch.no_grad():
            for i in picks:
                batch = ds.collate_batch([ds[i]])
                load_data_to_gpu(batch)
                batch = model(batch)
                p, c = batch['pred_boxes9d'][0], batch['confidence'][0]
                out[i] = None if (p is None or len(p) == 0) else np.asarray(p)[int(np.argmax(np.asarray(c)))]
        preds[tag] = out
        del model
        torch.cuda.empty_cache()
        print('%s 完成' % tag)

    K = ds.intrinsic
    dist = ds.distortion_matrix
    sheets = []
    for i in picks:
        info = ds.infos[i]
        img = cv2.imread(info['im_path'])
        if img is None:
            continue
        d = ds[i]
        gt = np.asarray(d['gt_box9d'])[0]
        img = draw_box(img, gt, K, dist, (60, 230, 60), 3)
        lines = ['%s  %s/%s  GT z=%.2fm' % (info['cls_name'], info['scene_id'], info['seq_id'], gt[2])]
        for k, (tag, _) in enumerate(models):
            p = preds[tag][i]
            if p is None:
                lines.append('%s: no detection' % tag)
                continue
            img = draw_box(img, p, K, dist, COLORS[k % len(COLORS)], 2)
            e = float(np.linalg.norm(p[:3] - gt[:3]))
            a = float(np.degrees((R.from_euler('xyz', p[6:9]).inv() * R.from_euler('xyz', gt[6:9])).magnitude()))
            lines.append('%s: %.3f m  %.0f deg' % (tag, e, a))
        # 围绕 GT 中心裁一块，目标才看得清
        if args.crop > 0:
            uv, _ = cv2.projectPoints(gt[:3].reshape(1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K, dist.reshape(1, -1))
            cx, cy = uv.reshape(2)
            h, w = img.shape[:2]
            s = args.crop // 2
            x0 = int(np.clip(cx - s, 0, w - 2 * s)); y0 = int(np.clip(cy - s, 0, h - 2 * s))
            img = img[y0:y0 + 2 * s, x0:x0 + 2 * s]
        img = cv2.resize(img, (520, 520))
        pad = np.zeros((150, 520, 3), np.uint8)
        for j, t in enumerate(lines):
            col = (255, 255, 255) if j == 0 else COLORS[(j - 1) % len(COLORS)]
            cv2.putText(pad, t, (8, 26 + j * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
        sheets.append(np.vstack([img, pad]))
    n = len(sheets)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    H, W = sheets[0].shape[:2]
    sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
    for k, s in enumerate(sheets):
        r, c = divmod(k, cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = s
    hdr = np.zeros((46, sheet.shape[1], 3), np.uint8)
    cv2.putText(hdr, 'GT = green;  ' + ';  '.join('%s' % t for t, _ in models), (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    sheet = np.vstack([hdr, sheet])
    p = os.path.join(args.out, 'compare_%s.jpg' % '_'.join(t for t, _ in models))
    cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('写出 %s  %s' % (p, sheet.shape))
    return 0


if __name__ == '__main__':
    sys.exit(main())
