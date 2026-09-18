# -*- coding: utf-8 -*-
"""双权重融合（2026-09-18）：深度取早期权重、朝向取后期权重。

为什么：这两个头对训练时长的需求相反 ——
    深度（几何解，靠 2D 跨度）：第 4~5 轮最好，之后跨域退化（0.50 -> 0.77 m）
    朝向（直接回归，靠细节）：一路变好（折算 61 -> 41）
所以先看融合的上界：同一份检测下，位置用早期权重、旋转用后期权重。
只做推理，不训练；两条权重必须结构相同。

    python annot_audit/fuse_two.py --early <ckpt> --late <ckpt> [--interval 2]
"""
import argparse
import os
import sys

import numpy as np
import torch
from easydict import EasyDict

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(TOOLS)
from scipy.spatial.transform import Rotation as R   # noqa: E402

import eval_2d as E   # noqa: E402
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402


def np_(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml')
    ap.add_argument('--early', required=True)
    ap.add_argument('--late', required=True)
    ap.add_argument('--interval', type=int, default=2)
    ap.add_argument('--size-source', default='known')
    a = ap.parse_args()
    cfg = EasyDict()
    cfg_from_yaml_file(a.cfg, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.SIZE_SOURCE', a.size_source,
                   'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width',
                   'MODEL.POST_PROCESSING.MAX_OBJ', '1'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = a.interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    seq = str(cfg.DATA_CONFIG.get('EULER_SEQ', 'xyz'))
    models = []
    for ck in (a.early, a.late):
        m = build_network(cfg.MODEL, ds)
        n, t = m.load_params_from_file(ck, to_cpu=False)
        assert n == t, (ck, n, t)
        models.append(m.cuda().eval())
    res = {k: {'pos': [], 'ang': []} for k in ('early', 'late', 'fused')}
    with torch.no_grad():
        for batch in loader:
            gts = [np_(x).reshape(-1, 9) for x in batch['gt_box9d']]
            load_data_to_gpu(batch)
            outs = [m(dict(batch)) for m in models]
            for b in range(outs[0]['batch_size']):
                gt = gts[b][np.abs(gts[b]).sum(1) > 0]
                if not len(gt):
                    continue
                g = gt[0]
                pe = np_(outs[0]['pred_boxes9d'][b])
                pl = np_(outs[1]['pred_boxes9d'][b])
                if not len(pe) or not len(pl):
                    continue
                pe, pl = pe[0], pl[0]
                ang = lambda p: float(np.degrees((R.from_euler(seq, p[6:9]).inv() * R.from_euler(seq, g[6:9])).magnitude()))
                res['early']['pos'].append(float(np.linalg.norm(pe[:3] - g[:3])))
                res['early']['ang'].append(ang(pe))
                res['late']['pos'].append(float(np.linalg.norm(pl[:3] - g[:3])))
                res['late']['ang'].append(ang(pl))
                res['fused']['pos'].append(float(np.linalg.norm(pe[:3] - g[:3])))   # 位置来自早期
                res['fused']['ang'].append(ang(pl))                                  # 朝向来自后期
    for k, lab in (('early', '早期权重'), ('late', '后期权重'), ('fused', '融合（位置早/朝向晚）')):
        p = np.array(res[k]['pos'])
        an = np.array(res[k]['ang'])
        if not len(p):
            continue
        fold = np.minimum(an, 180 - an)
        print('  %-22s n=%4d | 位置中位 %.3f m | <0.2m %.1f%% | <0.5m %.1f%% | 角度中位 %.1f°（折算 %.1f）' % (
            lab, len(p), np.median(p), 100 * (p < 0.2).mean(), 100 * (p < 0.5).mean(), np.median(an), np.median(fold)))


if __name__ == '__main__':
    main()
