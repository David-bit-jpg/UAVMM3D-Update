# -*- coding: utf-8 -*-
"""关键点 + PnP 臂（K30）到底给了什么（2026-09-22）：仿真 test 与真实 test，只算检出目标。

    航向（不折）     机头误差 0~180°：<30° 占比、>150° 占比（= 前后对调）—— kp2d 有序角点是否真的分出了前后
    航向折180        与 orient_compare 同定义
    倾斜
    位置             PnP 位置（K 臂原样 both）vs 只取 PnP 旋转、位置走 center_dis（rot）——
                     K 臂没有 size2d 头，位置只能比 center_dis 路；说明 PnP 位置对尺寸多敏感
    角点             预测角点 vs 真值角点投影的像素误差（输入 512 分辨率，8 点平均）

    python annot_audit/kp_arm_diag.py --run <run_dir> --real-cfg cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.datasets.pre_processor.pre_processor import KP_SCALE   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def last_ckpt(run):
    cks = glob.glob(os.path.join(run, 'ckpt', 'checkpoint_epoch_*.pth'))
    return max(cks, key=lambda c: int(re.search(r'epoch_(\d+)', c).group(1)))


def heading_err_deg(Rg, Rp):
    E = Rg.T @ Rp
    return abs(np.degrees(np.arctan2(E[1, 0], E[0, 0])))


def run(cf, ckpt, parts, interval):
    c = EasyDict()
    cfg_from_yaml_file(cf, c)
    c.MODEL.POST_PROCESSING.KP2D_POSE_PARTS = parts        # K 臂配置里没有这个键，cfg_from_list 会拒绝新键，直接赋值
    c.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(c.DATA_CONFIG, batch_size=8, dist=False, workers=0, logger=common_utils.create_logger(), training=False)
    m = build_network(c.MODEL, ds)
    n, t = m.load_params_from_file(ckpt, to_cpu=False)
    assert n == t
    m.cuda().eval()
    stride = int(c.DATA_CONFIG.STRIDE)
    out = {'he': [], 'tilt': [], 'pos': [], 'kp': [], 'cls': []}
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            ret = m(batch)
            for b in range(ret['batch_size']):
                g = batch['gt_box9d'][b].cpu().numpy().reshape(-1, 9)
                g = g[np.abs(g).sum(1) > 0]
                pred = ret['pred_boxes9d'][b]
                conf = ret['confidence'][b]
                if pred is None or len(conf) == 0:
                    continue
                pred = np.asarray(pred).reshape(len(pred), -1)[:, :9]
                conf = np.asarray(conf).reshape(-1)
                K = batch['intrinsic'][b].reshape(-1, 3, 3)[0]
                K = np.asarray(K.cpu() if hasattr(K, 'cpu') else K, np.float64)
                kp = batch['pred_center_dict']['kp2d'][b].float().cpu().numpy()         # (B*K, 16, h, w) 里第 b 张（K=1）
                for gb in g:
                    d = np.linalg.norm(pred[:, :3] - gb[:3], axis=1)
                    cand = [j for j in np.argsort(-conf) if d[j] < 1.0]
                    if not cand:
                        continue
                    p = pred[cand[0]]
                    Rg, Rp = R.from_euler('xyz', gb[6:9]).as_matrix(), R.from_euler('xyz', p[6:9]).as_matrix()
                    out['he'].append(heading_err_deg(Rg, Rp))
                    out['cls'].append(str(batch['seq_id'][b]).replace(chr(92), '/').split('/')[-1].split('_')[0] if 'carla_data' in str(batch['seq_id'][b]) else str(batch['seq_id'][b]).split('/')[0])
                    out['tilt'].append(np.degrees(np.arccos(np.clip(Rg[:, 2] @ Rp[:, 2], -1, 1))))
                    out['pos'].append(d[cand[0]])
                    # 角点像素误差：真值中心格上的预测偏移 vs 真值角点相对中心的偏移
                    q = K @ gb[:3]
                    u, v = q[0] / q[2], q[1] / q[2]
                    iu, iv = int(u / stride), int(v / stride)
                    if 0 <= iv < kp.shape[1] and 0 <= iu < kp.shape[2]:
                        C = (PROTO8 * gb[3:6]) @ Rg.T + gb[:3]
                        uv = (C @ K.T)
                        uv = uv[:, :2] / uv[:, 2:]
                        gt_off = uv - np.array([u, v])
                        pr_off = kp[:, iv, iu].reshape(8, 2) * KP_SCALE
                        out['kp'].append(np.linalg.norm(pr_off - gt_off, axis=1).mean())
    del m
    torch.cuda.empty_cache()
    return {k: np.asarray(v) for k, v in out.items()}


def per_class(o):
    cls = o['cls']
    lines = []
    for c in sorted(set(cls)):
        s = cls == c
        he = o['he'][s]
        lines.append('      %-18s n=%4d 航向不折中位 %5.1f° <30° %4.1f%% >150° %4.1f%% | 折180 %5.1f° | 倾斜 %5.1f°' % (
            c, s.sum(), np.median(he), 100 * (he < 30).mean(), 100 * (he > 150).mean(), np.median(np.minimum(he, 180 - he)), np.median(o['tilt'][s])))
    return chr(10).join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--sim-cfg', default=None)
    ap.add_argument('--real-cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml')
    ap.add_argument('--interval', type=int, default=4)
    ap.add_argument('--ckpts', nargs='*', default=None, help='指定若干权重（缺省取 run 里最后一轮）')
    ap.add_argument('--parts', nargs='+', default=['both', 'rot'])
    ap.add_argument('--per-class', action='store_true', help='按机型拆（仿真按序列名里的机型，真实按 phantom4/mavic2）')
    ap.add_argument('--domains', nargs='+', default=['仿真', '真实'])
    a = ap.parse_args()
    for ck in (a.ckpts or [last_ckpt(a.run)]):
        run_one(a, ck)


def run_one(a, ck):
    sim_cfg = a.sim_cfg or os.path.join('cfgs', 'models', 'uavdet_3d', 'camnorm', os.path.basename(os.path.dirname(os.path.normpath(a.run))) + '.yaml')
    print('权重', ck)
    print('%-6s %-5s %6s | %10s %9s %9s | %6s | %8s | %8s' % ('域', '位姿', '目标', '航向不折中位', '<30°', '>150°', '折180', '倾斜', 'PnP位置中位'))
    for dom, cf in (('仿真', sim_cfg), ('真实', a.real_cfg)):
        if dom not in a.domains:
            continue
        for parts in a.parts:
            o = run(cf, ck, parts, a.interval)
            he = o['he']
            print('%-6s %-5s %6d | %9.1f° %8.1f%% %8.1f%% | %5.1f° | %7.1f° | %7.3f m   角点误差中位 %.1f px' % (
                dom, parts, len(he), np.median(he), 100 * (he < 30).mean(), 100 * (he > 150).mean(),
                np.median(np.minimum(he, 180 - he)), np.median(o['tilt']), np.median(o['pos']), np.median(o['kp'])))
            if a.per_class:
                print(per_class(o))


if __name__ == '__main__':
    main()
