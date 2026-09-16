# -*- coding: utf-8 -*-
"""纯仿真模型零样本误差拆解：问题出在 2D 定位 / 深度 / 尺寸口径 / 朝向 的哪一环？

对照：同一模型在仿真测试集（2.5 倍长焦裁窗，虚拟深度与 MAV6D 相当）上的同一套拆解 —— 域内能做到什么程度。

拆解项（每个 GT 取最近的检测）
  2D   中心像素误差 / 真值表观长边（<0.5 视为 2D 定位正确）
  深度 预测/真值；「尺寸口径校正」后 = 预测深度 * (真值尺寸 / 预测尺寸)（去掉「把机子认大」这一项）
  朝向 总误差；机体 z 轴夹角（倾斜是否对）；机头水平朝向差（绕世界竖直轴，只在 2D 定位正确的帧上统计）；
       误差 >150°（机头机尾对调）的比例

    python diag_zeroshot_decompose.py --ckpt ../output/models/uavdet_3d/camnorm/sim_indoor8_mz/S1/ckpt/best.pth
"""
import argparse
import os
import sys

import numpy as np
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
CAM2VICON_R = np.array([[0.6685859, -0.74342, 0.01787715], [0.01558769, -0.01002444, -0.99982825],
                        [0.74347153, 0.66874974, 0.004886]])


def infer(cfg_file, split, interval, ckpt, sets):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    if sets:
        cfg_from_list(sets, cfg)
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = split
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(ckpt, to_cpu=False)
    assert n == t
    model.cuda().eval()
    return pose_eval.run_inference(model, loader), ds


def up_vector(domain, K=None):
    """世界竖直向上方向在相机系中的表示。MAV6D 用 camera2vicon（VICON z 朝上）；仿真用每帧外参（这里只统计 MAV6D 的水平朝向）。"""
    return CAM2VICON_R @ np.array([0, 0, 1.0])


def decompose(name, recs, domain):
    rows = []
    up = up_vector(domain)
    for r in recs:
        if len(r['gt']) == 0 or len(r['pred']) == 0:
            continue
        K = r['K']
        for g in r['gt']:
            ug = K @ g[:3]
            ug = ug[:2] / ug[2]
            up_ = (K @ r['pred'][:, :3].T).T
            up_ = up_[:, :2] / up_[:, 2:3]
            j = int(np.argmin(np.linalg.norm(up_ - ug, axis=1)))
            p = r['pred'][j]
            gc = (PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            gu = (K @ gc.T).T
            gu = gu[:, :2] / gu[:, 2:3]
            app = max(gu[:, 0].max() - gu[:, 0].min(), gu[:, 1].max() - gu[:, 1].min())
            e2d = float(np.linalg.norm(up_[j] - ug))
            Rg = R.from_euler('xyz', g[6:9]).as_matrix()
            Rp = R.from_euler('xyz', p[6:9]).as_matrix()
            ang = float(np.degrees((R.from_matrix(Rg).inv() * R.from_matrix(Rp)).magnitude()))
            tilt = float(np.degrees(np.arccos(np.clip(Rg[:, 2] @ Rp[:, 2], -1, 1))))
            if domain == 'mav6d':
                hx = [v - (v @ up) * up for v in (Rg[:, 0], Rp[:, 0])]
                if min(np.linalg.norm(hx[0]), np.linalg.norm(hx[1])) > 1e-3:
                    a, b = hx[0] / np.linalg.norm(hx[0]), hx[1] / np.linalg.norm(hx[1])
                    yaw = float(np.degrees(np.arctan2(np.cross(a, b) @ up, a @ b)))
                else:
                    yaw = np.nan
            else:
                yaw = np.nan
            size_ratio = float(np.max(p[3:6]) / np.max(g[3:6]))
            rows.append((e2d / max(app, 1.0), p[2] / g[2], p[2] / g[2] / size_ratio, ang, tilt, abs(yaw), size_ratio, app))
    a = np.array(rows, dtype=np.float64)
    ok2d = a[:, 0] < 0.5
    print('== %s（%d 个目标）' % (name, len(a)))
    print('   2D：中心误差/表观长边 中位 %.2f；2D 定位正确（<0.5 个机身）%.1f%%；表观长边中位 %.0f px'
          % (np.median(a[:, 0]), 100 * ok2d.mean(), np.median(a[:, 7])))
    for lab, sel in (('全部', np.ones(len(a), bool)), ('2D 正确的帧', ok2d)):
        if not sel.any():
            continue
        b = a[sel]
        print('   [%s] 深度 预测/真值 中位 %.3f (p10 %.2f p90 %.2f) | 尺寸口径校正后 %.3f (p10 %.2f p90 %.2f) | 预测/真值尺寸 %.2f'
              % (lab, np.median(b[:, 1]), np.percentile(b[:, 1], 10), np.percentile(b[:, 1], 90), np.median(b[:, 2]),
                 np.percentile(b[:, 2], 10), np.percentile(b[:, 2], 90), np.median(b[:, 6])))
        yaw = b[:, 5][np.isfinite(b[:, 5])]
        print('   [%s] 朝向 总误差中位 %.1f° | 机体 z 夹角（倾斜）中位 %.1f° | 机头水平朝向差 中位 %s | 误差>150° %.1f%% | <30° %.1f%%'
              % (lab, np.median(b[:, 3]), np.median(b[:, 4]), ('%.1f°' % np.median(yaw)) if len(yaw) else '-',
                 100 * (b[:, 3] > 150).mean(), 100 * (b[:, 3] < 30).mean()))
        if len(yaw):
            h = np.histogram(yaw, bins=[0, 30, 60, 90, 120, 150, 180])[0]
            print('   [%s] 机头水平朝向差分布（0-30/30-60/60-90/90-120/120-150/150-180°）%s' % (lab, (h / len(yaw) * 100).round(1).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--mav-split', default='test')
    ap.add_argument('--sim-interval', type=int, default=5)
    ap.add_argument('--sim-cfg', default='cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz.yaml',
                    help='仿真域内参照用哪版标签（index_nose / index_nosescale 对应的配置）')
    args = ap.parse_args()
    recs, _ = infer(args.sim_cfg, 'test', args.sim_interval, args.ckpt,
                    ['DATA_CONFIG.VAL_ZOOMS', '[2.5]'])
    decompose('仿真测试集（2.5 倍长焦，域内参照）', recs, 'sim')
    recs, _ = infer('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', args.mav_split, 1, args.ckpt, None)
    decompose('MAV6D %s（零样本）' % args.mav_split, recs, 'mav6d')
    for cls in ('mavic2', 'phantom4'):
        sub = [r for r in recs if r['seq_id'].startswith(cls)]
        decompose('MAV6D %s · %s' % (args.mav_split, cls), sub, 'mav6d')


if __name__ == '__main__':
    main()
