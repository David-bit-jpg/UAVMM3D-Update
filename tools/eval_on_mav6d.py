# -*- coding: utf-8 -*-
"""在 MAV6D 测试集上统一评测任意 checkpoint，用于三臂对比。

三个臂的模型结构不完全一样（纯模拟那个 hm 是 7 类，MAV6D 上训的是 2 类），
所以不能直接用各自配置里的评测流程比。这里统一成【类别无关】的位姿误差：
每帧取置信度最高的那个检测，和 GT 比位置与旋转。

    A 纯模拟 : 仿真上训好的权重，零样本直接测 MAV6D
    B 迁移   : 仿真预训练 -> MAV6D 微调后的权重
    C 纯真实 : 只在 MAV6D 上从零训的权重

用法：
    python tools/eval_on_mav6d.py --ckpt <权重>.pth --tag A_sim_only --decode-max-dis 15
    python tools/eval_on_mav6d.py --ckpt <权重>.pth --tag C_real_only
    # 加 --vis 6 会顺便把预测框(红)和 GT 框(绿)画到图上

关于 --decode-max-dis：
center_dis 头输出的是归一化深度，米制深度 = 输出 x MAX_DIS。解码时必须用
【训练该权重时的 MAX_DIS】，否则深度会整体差一个比例。仿真那版是 15，
MAV6D 上训的是 8。不给就用配置里的值。
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

from uavdet3d.config import cfg_from_yaml_file
from uavdet3d.datasets import build_dataloader
from uavdet3d.model import build_network, load_data_to_gpu
from uavdet3d.utils import common_utils

MAV6D_CFG = 'cfgs/models/uavdet_3d/mav6d/centerdet.yaml'


def infer_hm_channels(ckpt_path):
    """从 ckpt 里读出 hm 头的输出通道数，据此建对应结构的模型。"""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = sd.get('model_state', sd)
    for k, v in sd.items():
        if k.startswith('dense_head_2d.hm.') and k.endswith('.weight') and v.ndim == 4:
            last = v
    return int(last.shape[0])


def draw_box(img, box9d, K, dist, color, thickness=2):
    """把 9D 框画到图上。box9d = [x,y,z,l,w,h,a1,a2,a3]，相机系，xyz 欧拉角。"""
    c = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
    pts = c * np.asarray(box9d[3:6])
    pts = pts @ R.from_euler('xyz', box9d[6:9]).as_matrix().T + np.asarray(box9d[0:3])
    if np.any(pts[:, 2] <= 1e-3):
        return img
    uv, _ = cv2.projectPoints(pts.astype(np.float64), np.zeros(3), np.zeros(3),
                              K, np.asarray(dist).reshape(1, -1))
    uv = uv.reshape(-1, 2).astype(int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, thickness)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='eval')
    ap.add_argument('--data-path', default='E:/MAV6D')
    ap.add_argument('--decode-max-dis', type=float, default=None,
                    help='训练该权重时的 MAX_DIS；不给则用 MAV6D 配置里的值')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--vis', type=int, default=0, help='额外画 N 张预测 vs GT 对比图')
    ap.add_argument('--vis-out', default=None)
    ap.add_argument('--rot-repr', default=None,
                    help="该权重训练时用的旋转表示（'euler6' / 'r6d'）。"
                         '不给就用 MAV6D 配置里的值。解码必须与编码一致')
    ap.add_argument('--no-norm', action='store_true',
                    help='去掉 mav6d.yaml 里的 NORM_MEAN/STD（评 2026-09-06 之前只除 255 训出来的旧权重时用）')
    ap.add_argument('--json', default=None,
                    help='把指标写成 JSON，便于把多次评测汇总成曲线')
    args = ap.parse_args()

    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(MAV6D_CFG, cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.data_path
    if args.decode_max_dis is not None:
        cfg.DATA_CONFIG.MAX_DIS = args.decode_max_dis
    if args.rot_repr is not None:
        cfg.DATA_CONFIG.ROT_REPR = args.rot_repr
    if args.no_norm:
        cfg.DATA_CONFIG.pop('NORM_MEAN', None)
        cfg.DATA_CONFIG.pop('NORM_STD', None)

    # 只按 ckpt 调模型的 hm 通道数；CLASS_NAMES 必须保持 MAV6D 的真实型号，
    # 因为 dataset 要用它去找 <型号>/split/ 目录。
    # 评测时不算 loss，所以 GT 热图(2 通道)和预测热图(可能 7 通道)通道数不同没关系；
    # 解码器对通道取 max，本来就是类别无关的。
    n_hm = infer_hm_channels(args.ckpt)
    cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.hm.out_channels = n_hm

    print('=' * 72)
    print('评测 : %s' % args.tag)
    print('权重 : %s' % args.ckpt)
    print('hm 通道 %d  解码 MAX_DIS %s  分辨率 %s'
          % (n_hm, cfg.DATA_CONFIG.MAX_DIS, list(cfg.DATA_CONFIG.IM_RESIZE)))
    print('=' * 72)

    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=args.batch_size,
                                     dist=False, workers=args.workers, logger=logger,
                                     training=False)
    print('测试帧数: %d' % len(ds))

    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()

    pos_err, ang_err, z_err, z_gt_all, conf_all = [], [], [], [], []
    vis_pool = []

    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            batch = model(batch)
            for b in range(batch['batch_size']):
                pred = batch['pred_boxes9d'][b]
                conf = batch['confidence'][b]
                gt = batch['gt_box9d'][b]
                gt = gt.cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
                gt = gt[0]
                if pred is None or len(pred) == 0:
                    continue
                pred = np.asarray(pred)
                conf = np.asarray(conf)
                k = int(np.argmax(conf))          # 类别无关：取置信度最高的那个
                p = pred[k]

                pos_err.append(float(np.linalg.norm(p[0:3] - gt[0:3])))
                z_err.append(float(abs(p[2] - gt[2])))
                z_gt_all.append(float(gt[2]))
                conf_all.append(float(conf[k]))
                rp = R.from_euler('xyz', p[6:9])
                rg = R.from_euler('xyz', gt[6:9])
                ang_err.append(float(np.degrees((rp.inv() * rg).magnitude())))

                if args.vis and len(vis_pool) < args.vis:
                    vis_pool.append((batch['scene_id'][b], batch['seq_id'][b],
                                     batch['frame_id'][b], p.copy(), gt.copy(),
                                     np.asarray(batch['intrinsic'][b])[0],
                                     np.asarray(batch['distortion'][b])[0]))

    pos_err = np.array(pos_err)
    ang_err = np.array(ang_err)
    z_err = np.array(z_err)
    print('\n有效样本 %d / %d' % (len(pos_err), len(ds)))
    if len(pos_err) == 0:
        print('!! 没有任何有效检测')
        return 1

    def stat(name, a, unit):
        print('  %-12s 中位 %8.3f  均值 %8.3f  25%% %8.3f  75%% %8.3f  90%% %8.3f  %s'
              % (name, np.median(a), a.mean(), np.percentile(a, 25),
                 np.percentile(a, 75), np.percentile(a, 90), unit))

    print('--- %s ---' % args.tag)
    stat('位置误差', pos_err, 'm')
    stat('深度误差', z_err, 'm')
    stat('角度误差', ang_err, 'deg')
    print('  GT 深度范围 %.2f ~ %.2f m (中位 %.2f)'
          % (min(z_gt_all), max(z_gt_all), np.median(z_gt_all)))
    for t in [0.1, 0.2, 0.5, 1.0]:
        print('  位置误差 < %.1f m 的比例: %5.1f%%' % (t, 100.0 * (pos_err < t).mean()))

    if args.json:
        import json
        res = {
            'tag': args.tag, 'ckpt': args.ckpt, 'n_valid': int(len(pos_err)),
            'n_total': int(len(ds)), 'decode_max_dis': float(cfg.DATA_CONFIG.MAX_DIS),
        }
        for nm, a in (('pos', pos_err), ('z', z_err), ('ang', ang_err)):
            res[nm + '_median'] = float(np.median(a))
            res[nm + '_mean'] = float(a.mean())
            res[nm + '_p90'] = float(np.percentile(a, 90))
        for t in (0.1, 0.2, 0.5, 1.0):
            res['acc_%g' % t] = float((pos_err < t).mean())
        for t in (5, 10, 20):
            res['acc_%ddeg' % t] = float((ang_err < t).mean())
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
        print('  指标写出 %s' % args.json)

    if vis_pool:
        out = args.vis_out or os.path.join('..', 'output', 'eval_vis_' + args.tag)
        os.makedirs(out, exist_ok=True)
        for sc, sq, fr, p, gt, K, dist in vis_pool:
            for cls in list(cfg.DATA_CONFIG.CLASS_NAMES) + ['mavic2', 'phantom4']:
                ip = os.path.join(args.data_path, cls, 'JPEGImages', sc, sq, fr)
                if os.path.exists(ip):
                    break
            img = cv2.imread(ip)
            if img is None:
                continue
            img = draw_box(img, gt, K, dist, (0, 255, 0), 2)      # GT 绿
            img = draw_box(img, p, K, dist, (0, 0, 255), 2)       # 预测 红
            e = float(np.linalg.norm(p[0:3] - gt[0:3]))
            cv2.putText(img, '%s  GT(green) Z=%.2fm  Pred(red) Z=%.2fm  err=%.3fm'
                        % (args.tag, gt[2], p[2], e), (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            op = os.path.join(out, '%s_%s_%s_%s' % (args.tag, sc, sq, fr))
            cv2.imwrite(op, img)
            print('  可视化写出 %s' % op)
    return 0


if __name__ == '__main__':
    sys.exit(main())
