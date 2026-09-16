# -*- coding: utf-8 -*-
"""camnorm 系列统一评测：类别无关位姿指标（uavdet3d/utils/pose_eval.py，分母 = 全部 GT）
+ 可选的仓库原版 LAA3D_ADS 打印（6dof error/accuracy、3D AP_R40、2D AP_R40、ADS）。

    python eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d.yaml --ckpt <权重>.pth --tag T \
        --split test --json ../output/camnorm/json/T.json --ads

与旧 eval_on_mav6d.py / eval_ads_mav6d.py 的区别（都对应审查条目）：
- 两域几何约定统一（虚拟深度、MAX_DIS / MAX_SIZE 相同），零样本不需要任何 --decode-max-* 换算（P2）
- ADS 前显式 set_default_euler_seq(EULER_SEQ)，九点生成 / 角度误差都按存储约定 'xyz'（P16）
- ADS 尺寸误差按 l,w,h 参数算（P15，见 eval6dof_laam6d.py）
- 准确率一律以全部 GT 为分母（P19）
- 权重必须逐张量全部载入，否则直接报错（迁移最怕以为载上了其实没有）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import camera_geometry as cg   # noqa: E402
from uavdet3d.utils import common_utils, frame_convention, pose_eval   # noqa: E402

CARLA_TO_OPENCV = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

ADS_PROFILES = {
    'laa': dict(DisMax=150, DisMin=0, MinPixel=16,
                AP2D=dict(IoUThresh=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], RecallNum=101),
                AP3D=dict(DisThresh=[1, 2, 4, 6], RecallNum=101),
                Dof6=dict(DisNormMax=8, OriNormMax=30, SizeNormMax=1)),
    'indoor': dict(DisMax=10, DisMin=0, MinPixel=16,
                   AP2D=dict(IoUThresh=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], RecallNum=101),
                   AP3D=dict(DisThresh=[0.05, 0.1, 0.2, 0.5], RecallNum=101),
                   Dof6=dict(DisNormMax=1, OriNormMax=30, SizeNormMax=0.2)),
}


def ads_eval(records, raw_whs, W, H, seq, profiles, out_root, tag):
    """仓库原版 LAA3D_ADS。框在 OpenCV 相机系，传 extrinsic=CARLA->OpenCV 让 box9d_to_2d 的两步变换抵消；
    内参换算回原图分辨率（MinPixel=16 是按原图像素定义的）。"""
    from uavdet3d.datasets.laam6d.ads_metric_laam6d import LAA3D_ADS_Metric
    from uavdet3d.datasets.laam6d.dataset_utils import convert_9params_to_9points
    frame_convention.set_default_euler_seq(seq)          # 审查 P16

    def pts(p):
        p = np.asarray(p, dtype=np.float32).reshape(-1, 9)
        return convert_9params_to_9points(p) if len(p) else np.empty((0, 9, 3), np.float32)

    annos = []
    for r, (rw, rh) in zip(records, raw_whs):
        annos.append({
            'seq_id': r['seq_id'], 'frame_id': r['frame_id'],
            'gt_names': np.array(['drone'] * len(r['gt'])), 'gt_boxes': pts(r['gt']),
            'pred_names': np.array(['drone'] * len(r['pred'])), 'pred_boxes': pts(r['pred']),
            'confidence': r['conf'].astype(np.float32),
            'intrinsic': cg.scale_K(r['K'], rw / float(W), rh / float(H)),
            'extrinsic': CARLA_TO_OPENCV, 'distortion': np.zeros(5),
        })
    reports = {}
    for prof in profiles:
        out = os.path.join(out_root, tag, prof)
        os.makedirs(out, exist_ok=True)
        m = LAA3D_ADS_Metric(eval_config=EasyDict(ADS_PROFILES[prof]), classes=['drone'], metric_save_path=out)
        s = m.eval(annos)
        open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8').write(s + '\n')
        reports[prof] = s
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--interval', type=int, default=1)
    ap.add_argument('--match', default='top1', choices=['top1', 'greedy'])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--json', default=None)
    ap.add_argument('--ads', action='store_true', help='同时跑仓库原版 LAA3D_ADS 并打印')
    ap.add_argument('--ads-profiles', default='laa,indoor')
    ap.add_argument('--ads-out', default='E:/Open3DUAVDet/output/camnorm/ads')
    ap.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    args = ap.parse_args()

    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(args.cfg, cfg)
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs, cfg)
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = args.split
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = args.interval
    seq = str(cfg.DATA_CONFIG.get('EULER_SEQ', 'xyz'))

    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=args.batch_size, dist=False,
                                     workers=args.workers, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    n_loaded, n_total = model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    if n_loaded != n_total:
        raise RuntimeError('权重没有全部载入（%d/%d），结构与配置不一致，拒绝评测' % (n_loaded, n_total))
    model.cuda().eval()

    summary, recs = pose_eval.evaluate(model, loader, seq=seq, match=args.match)
    raw_whs = [tuple(ds.metas[int(i)].get('raw_wh', (ds.W, ds.H))) for i in ds.valid_idx]
    assert len(raw_whs) == len(recs)

    print('=' * 100)
    print(pose_eval.format_summary('%s [%s %d 帧]' % (args.tag, args.split, len(recs)), summary))
    if 'pos_median' in summary:
        print('  准确率（分母 = 全部 %d 个 GT，没检出算失败）' % summary['n_gt'])
        print('    位置 <5cm %.1f%%  <10cm %.1f%%  <20cm %.1f%%  <50cm %.1f%%'
              % tuple(100 * summary['ACC_pos_%g' % t] for t in (0.05, 0.1, 0.2, 0.5)))
        print('    朝向 <5° %.1f%%  <10° %.1f%%  <20° %.1f%%  <30° %.1f%%   (180° 折算: %.1f / %.1f / %.1f / %.1f%%)'
              % (tuple(100 * summary['ACC_rot_%d' % t] for t in (5, 10, 20, 30)) +
                 tuple(100 * summary['ACC_rotfold_%d' % t] for t in (5, 10, 20, 30))))
        print('    位置且朝向 <5cm&5° %.1f%%  <10cm&10° %.1f%%  <20cm&20° %.1f%%   ADD<10%%直径 %.1f%%'
              % (100 * summary['ACC_0.05m5deg'], 100 * summary['ACC_0.1m10deg'], 100 * summary['ACC_0.2m20deg'],
                 100 * summary['ACC_add_10']))
    print('=' * 100)

    res = dict(summary)
    res.update({'tag': args.tag, 'ckpt': args.ckpt, 'split': args.split, 'interval': args.interval, 'cfg': args.cfg})
    best_json = os.path.join(os.path.dirname(args.ckpt), 'best.json')
    if os.path.basename(args.ckpt) == 'best.pth' and os.path.exists(best_json):
        res['best'] = json.load(open(best_json, encoding='utf-8'))
    if args.ads:
        reports = ads_eval(recs, raw_whs, ds.W, ds.H, seq, args.ads_profiles.split(','), args.ads_out, args.tag)
        for prof, s in reports.items():
            print('\n' + '#' * 30 + ' LAA3D_ADS 口径=%s ' % prof + '#' * 30)
            print(s)
        res['ads_reports'] = reports
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(res, open(args.json, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print('写出 %s' % args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
