# -*- coding: utf-8 -*-
"""机头（机体 x 轴）约定核对：用「在真实 MAV6D 上微调过的模型」去看仿真测试集。

微调模型学到的是 MAV6D 的机头约定（x 指向云台）；它在仿真图上预测的 x 轴指向仿真无人机的「视觉机头」。
把它与仿真标签的 x 轴比，绕机体 z 的带符号偏航差分布就说明仿真标签的 x 轴相对视觉机头转了多少：
    峰在 0 / ±180 -> 两域约定一致（180 = 机头机尾分不清）；峰在 ±90 -> 仿真网格相对标签转了 90°。
对照：纯仿真模型 S1 在同一批帧上应当接近 0。逐机型统计。

    python diag_heading_convention.py --ckpt ../output/models/uavdet_3d/camnorm/mav6d/T_p025_s0/ckpt/best.pth --tag T_p025_s0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--interval', type=int, default=3)
    ap.add_argument('--zoom', default='2.5')
    args = ap.parse_args()
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz.yaml', cfg)
    cfg_from_list(['DATA_CONFIG.VAL_ZOOMS', '[%s]' % args.zoom], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = args.interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(args.ckpt, to_cpu=False)
    assert n == t
    model.cuda().eval()
    recs = pose_eval.run_inference(model, loader)
    names = {}
    for i in ds.valid_idx:
        m = ds.metas[int(i)]
        names[(m['seq'], m['frame'])] = list(m['names'])
    per = {}
    n_gt = n_det = 0
    for r in recs:
        nm = names[(r['seq_id'], r['frame_id'])]
        K = r['K']
        for g, model_name in zip(r['gt'], nm):
            n_gt += 1
            if len(r['pred']) == 0:
                continue
            ug = K @ g[:3]
            ug = ug[:2] / ug[2]
            up_ = (K @ r['pred'][:, :3].T).T
            up_ = up_[:, :2] / up_[:, 2:3]
            j = int(np.argmin(np.linalg.norm(up_ - ug, axis=1)))
            gc = (K @ g[:3])
            app = 512.0 * max(g[3:6]) / g[2]          # 表观大小（缓存像素，粗略）
            if np.linalg.norm(up_[j] - ug) > 0.5 * max(app, 1.0):
                continue                               # 2D 都没对上的不算
            n_det += 1
            Rg = R.from_euler('xyz', g[6:9]).as_matrix()
            Rp = R.from_euler('xyz', r['pred'][j][6:9]).as_matrix()
            zb = Rg[:, 2]                              # 机体 z（≈ 竖直）
            a = Rg[:, 0]
            b = Rp[:, 0] - (Rp[:, 0] @ zb) * zb
            if np.linalg.norm(b) < 1e-3:
                continue
            b /= np.linalg.norm(b)
            yaw = float(np.degrees(np.arctan2(np.cross(a, b) @ zb, a @ b)))
            per.setdefault(model_name, []).append(yaw)
    print('%s：仿真测试集（裁窗 %s 倍），GT %d，2D 对上 %d' % (args.tag or args.ckpt, args.zoom, n_gt, n_det))
    bins = np.arange(-180, 181, 30)
    labels = ['%d~%d' % (a, a + 30) for a in range(-180, 180, 30)]
    print('%-18s %5s  %s' % ('机型', 'n', '  '.join('%8s' % l for l in labels)))
    allv = []
    for k in sorted(per):
        v = np.array(per[k])
        allv.append(v)
        h = np.histogram(v, bins=bins)[0] / len(v) * 100
        print('%-18s %5d  %s   | 中位 %+.1f°  |偏差|<30° %.0f%%  ±(60~120)° %.0f%%  >150° %.0f%%'
              % (k, len(v), '  '.join('%8.1f' % x for x in h), np.median(v), 100 * (np.abs(v) < 30).mean(),
                 100 * ((np.abs(v) > 60) & (np.abs(v) < 120)).mean(), 100 * (np.abs(v) > 150).mean()))
    v = np.concatenate(allv)
    h = np.histogram(v, bins=bins)[0] / len(v) * 100
    print('%-18s %5d  %s' % ('全部', len(v), '  '.join('%8.1f' % x for x in h)))


if __name__ == '__main__':
    main()
