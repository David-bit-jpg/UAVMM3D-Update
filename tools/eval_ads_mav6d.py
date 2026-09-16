# -*- coding: utf-8 -*-
"""用仓库自带的 LAA3D_ADS 指标（6dof error + accuracy、3D AP_R40、2D AP_R40、ADS）评测 MAV6D 上的权重，
输出格式与 laam6d 的评测打印一致，便于和源域结果对照。

    D:/Miniconda3/envs/city/python.exe tools/eval_ads_mav6d.py --ckpt <权重>.pth --tag S1MT_p05 [--decode-max-dis 40]

说明：
- 类别一律映射成 'drone'（各臂的 hm 通道数不同：源域 7 类、MAV6D 2 类），指标因此是类别无关的。
- ADS 指标里的 box9d_to_2d 假定框在【世界系】、按 CARLA 约定，会做 inv(extrinsic) 再 CARLA→OpenCV；
  我们的框已经在 OpenCV 相机系，所以传 extrinsic = CARLA→OpenCV 矩阵，让这两步互相抵消（自检见 --selftest）。
- 两套阈值都跑：
    laa    —— 与 cfgs/dataset_configs/uavdet_3d/laam6d.yaml 相同（AP3D 匹配阈值 1/2/4/6 m，位置归一化上限 8 m），
              和源域数字同口径，但 MAV6D 是 1.5–5.6 m 的室内场景，这套阈值偏松。
    indoor —— 按 MAV6D 的尺度收紧（AP3D 匹配阈值 0.05/0.1/0.2/0.5 m，位置归一化上限 1 m，朝向 30°，尺寸 0.2 m）。
"""
import argparse
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'E:/Open3DUAVDet')
from eval_on_mav6d import MAV6D_CFG, infer_hm_channels  # noqa: E402
from uavdet3d.config import cfg_from_yaml_file  # noqa: E402
from uavdet3d.datasets import build_dataloader  # noqa: E402
from uavdet3d.datasets.laam6d.ads_metric_laam6d import LAA3D_ADS_Metric  # noqa: E402
from uavdet3d.datasets.laam6d.dataset_utils import convert_9params_to_9points  # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu  # noqa: E402
from uavdet3d.utils import common_utils  # noqa: E402

CARLA_TO_OPENCV = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

PROFILES = {
    'laa': dict(DisMax=150, DisMin=0, MinPixel=16,
                AP2D=dict(IoUThresh=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], RecallNum=101),
                AP3D=dict(DisThresh=[1, 2, 4, 6], RecallNum=101),
                Dof6=dict(DisNormMax=8, OriNormMax=30, SizeNormMax=1)),
    'indoor': dict(DisMax=10, DisMin=0, MinPixel=16,
                   AP2D=dict(IoUThresh=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], RecallNum=101),
                   AP3D=dict(DisThresh=[0.05, 0.1, 0.2, 0.5], RecallNum=101),
                   Dof6=dict(DisNormMax=1, OriNormMax=30, SizeNormMax=0.2)),
}


def selftest():
    """box9d_to_2d 在传 extrinsic=CARLA→OpenCV 时应等价于直接用内参投影。"""
    m = LAA3D_ADS_Metric(eval_config=EasyDict(PROFILES['laa']), classes=['drone'], metric_save_path='.')
    K = np.array([[1979.4, 0.3984, 976.8189], [0.0, 1979.1, 533.9717], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    box = convert_9params_to_9points(np.array([[0.3, -0.2, 3.0, 0.34, 0.34, 0.23, 0.1, 0.2, 0.3]]))
    b2d, size = m.box9d_to_2d(box, intrinsic_mat=K, extrinsic_mat=CARLA_TO_OPENCV, distortion_matrix=dist)
    import cv2
    uv, _ = cv2.projectPoints(box[0, 1:].astype(np.float64), np.zeros(3), np.zeros(3), K, dist.reshape(1, -1))
    uv = uv.reshape(-1, 2)
    ref = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    err = float(np.abs(b2d[0] - ref).max())
    print('自检 box9d_to_2d：与直接投影的 2D 框最大差 %.2f px（应 <= 1，取整误差）' % err)
    assert err <= 1.5, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--data-path', default='E:/MAV6D')
    ap.add_argument('--decode-max-dis', type=float, default=None, help='零样本权重要用训练时的 40')
    ap.add_argument('--decode-max-size', type=float, default=None, help='零样本权重要用训练时的 4（mmcache MAX_SIZE）')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--no-norm', action='store_true', help='2026-09-06 之前训的权重没有按域归一化')
    ap.add_argument('--profiles', default='laa,indoor')
    ap.add_argument('--out', default='E:/Open3DUAVDet/output/ads_metrics')
    args = ap.parse_args()
    selftest()

    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(MAV6D_CFG, cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.data_path
    if args.decode_max_dis is not None:
        cfg.DATA_CONFIG.MAX_DIS = args.decode_max_dis
    if args.decode_max_size is not None:
        cfg.DATA_CONFIG.MAX_SIZE = args.decode_max_size
    if args.no_norm:
        cfg.DATA_CONFIG.pop('NORM_MEAN', None)
        cfg.DATA_CONFIG.pop('NORM_STD', None)
    cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.hm.out_channels = infer_hm_channels(args.ckpt)
    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=args.batch_size, dist=False,
                                     workers=args.workers, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()

    annos = []
    n_det = 0
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            batch = model(batch)
            for b in range(batch['batch_size']):
                def np_(x):
                    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
                gt = np_(batch['gt_box9d'][b])
                pred = batch['pred_boxes9d'][b]
                conf = batch['confidence'][b]
                pred = np_(pred) if pred is not None else np.zeros((0, 9))
                conf = np_(conf) if conf is not None else np.zeros(0)
                if len(pred):
                    n_det += 1
                annos.append({
                    'seq_id': str(batch['seq_id'][b]), 'frame_id': str(batch['frame_id'][b]),
                    'gt_names': np.array(['drone'] * len(gt)),
                    'gt_boxes': convert_9points_safe(gt),
                    'pred_names': np.array(['drone'] * len(pred)),
                    'pred_boxes': convert_9points_safe(pred[:, :9]),
                    'confidence': conf.astype(np.float32),
                    'intrinsic': np_(batch['intrinsic'][b])[0].astype(np.float64),
                    'extrinsic': CARLA_TO_OPENCV,
                    'distortion': np_(batch['distortion'][b])[0].astype(np.float64).reshape(-1),
                })
    print('\n测试帧 %d，有检出的帧 %d（%.1f%%）' % (len(annos), n_det, 100.0 * n_det / max(len(annos), 1)))

    for prof in args.profiles.split(','):
        out = os.path.join(args.out, args.tag, prof)
        os.makedirs(out, exist_ok=True)
        m = LAA3D_ADS_Metric(eval_config=EasyDict(PROFILES[prof]), classes=['drone'], metric_save_path=out)
        print('\n' + '=' * 78)
        print('%s   指标口径 = %s   %s' % (args.tag, prof,
              '(与源域 laam6d.yaml 同参数)' if prof == 'laa' else '(按 MAV6D 室内尺度收紧)'))
        print('=' * 78)
        s = m.eval(annos)
        print(s)
        open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8').write(s + '\n')
    return 0


def convert_9points_safe(params):
    p = np.asarray(params, dtype=np.float32)
    if p.size == 0:
        return np.empty((0, 9, 3), dtype=np.float32)
    return convert_9params_to_9points(p)


if __name__ == '__main__':
    sys.exit(main())
