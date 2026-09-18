# -*- coding: utf-8 -*-
"""纯 2D 评测：纯仿真训练的模型在 MAV6D test 上「图像里找没找到、框得准不准」，不看深度和姿态（2026-09-17）。

MAV6D 每帧恰好 1 架。所有量都在网络输入分辨率 512x288 上（原图 1920 宽，约 x3.75）。
  GT 2D 框     = GT 3D 框 8 角点投影的外接框（裁到画面内）；GT 中心 = 3D 中心投影
  预测中心     = 预测 3D 中心投影 —— 解码器把中心放在「热力图峰值 + 偏移」的视线上，所以它与深度无关，纯 2D
  预测 2D 框   = 预测 3D 框投影外接框（依赖 尺寸/深度 之比和姿态，不依赖深度绝对值）
指标
  top-1：中心误差（px）、命中率（误差 <= GT 框长边一半，即峰值落在机身上）、2D IoU、框大小比（预测长边 / GT 长边）
  AP  ：每帧最多取 10 个峰值（分数 >= 0.01），全数据集按分数排序，
        AP_ctr = 中心落在机身上（<= 半个长边）算 TP；AP_10px = 中心误差 <= 10 px；AP_IoU50 / AP_IoU30 = 2D 框 IoU
        VOC 全点插值；每个 GT 只配一次，多余的算 FP

    cd tools && python annot_audit/eval_2d.py [--interval 1]
"""
import argparse
import json
import os
import sys

import numpy as np
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TOOLS))
os.chdir(TOOLS)
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
M = '../output/models/uavdet_3d/camnorm'
C = 'cfgs/models/uavdet_3d/camnorm/'
# 名字, 评测配置, 权重, 说明
MODELS = [
    ('S1', C + 'mav6d.yaml', M + '/sim_indoor8_mz/S1/ckpt/best.pth', '仿真 · 旧 indoor8 八场景 · ResNet8x'),
    ('S2', C + 'mav6d.yaml', M + '/sim_indoor8_mz_nose/S2/ckpt/best.pth', '仿真 · S1 + 机头修正'),
    ('S3', C + 'mav6d.yaml', M + '/sim_indoor8_mz_nosescale/S3/ckpt/best.pth', '仿真 · S2 + 尺度修正'),
    ('P1', C + 'mav6d.yaml', M + '/sim_pp_realsize/P1/ckpt/best.pth', '仿真 · PowerPlant 真机尺寸 · ResNet8x（归一化常数用错）'),
    ('P1b', C + 'mav6d.yaml', M + '/sim_pp_realsize/P1b/ckpt/best.pth', '仿真 · P1 修正归一化'),
    ('R1', C + 'mav6d_r34.yaml', M + '/sim_pp_r34/R1/ckpt/best.pth', '仿真 · P1b 换 ImageNet ResNet-34'),
    ('N1*', C + 'mav6d_std_pp.yaml', M + '/sim_pp_std/N1/ckpt/best.pth', '仿真 · bias + 标准化（只训了 2 轮）'),
    ('G1*', C + 'mav6d_g1.yaml', M + '/sim_pp_g1/G1/ckpt/best.pth', '仿真 · ImageNet + size2d（只训了 3 轮）'),
    ('Real', C + 'mav6d.yaml', M + '/mav6d/C_p050_s0/ckpt/best.pth', '参照：MAV6D 50% 训练集从零训（真实上限）'),
]


def box2d(corners, K, W, H):
    uv = (K @ corners.T).T
    if (uv[:, 2] <= 1e-6).any():
        return None
    uv = uv[:, :2] / uv[:, 2:3]
    x0, y0 = np.clip(uv.min(0), 0, [W, H])
    x1, y1 = np.clip(uv.max(0), 0, [W, H])
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return None
    return np.array([x0, y0, x1, y1])


def corners_of(b):
    return (PROTO8 * b[3:6]) @ R.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def proj(K, p):
    q = K @ p
    return q[:2] / q[2]


def voc_ap(scores, tp, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp, float)[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def infer(cfg_file, ckpt, interval):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    cfg_from_list(['MODEL.POST_PROCESSING.MAX_OBJ', '10', 'MODEL.POST_PROCESSING.SCORE_THRESH', '0.01'], cfg)
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    n, t = model.load_params_from_file(ckpt, to_cpu=False)
    assert n == t, '权重没有全部载入 %d/%d' % (n, t)
    model.cuda().eval()
    return pose_eval.run_inference(model, loader), ds.new_im_width, ds.new_im_hight


def evaluate(records, W, H):
    top = {'err': [], 'hit': [], 'iou': [], 'ratio': [], 'score': [], 'gt_long': []}
    det = {'ctr': [], 'px10': [], 'iou50': [], 'iou30': []}
    scores = []
    n_gt = 0
    for r in records:
        K = r['K']
        for g in r['gt']:
            n_gt += 1
            gb = box2d(corners_of(g), K, W, H)
            gc = proj(K, g[:3])
            glong = max(gb[2] - gb[0], gb[3] - gb[1]) if gb is not None else 1.0
            used = {'ctr': False, 'px10': False, 'iou50': False, 'iou30': False}
            order = np.argsort(-r['conf'])
            for rank, j in enumerate(order):
                p = r['pred'][j]
                pc = proj(K, p[:3]) if p[2] > 1e-6 else np.array([1e9, 1e9])
                e = float(np.linalg.norm(pc - gc))
                pb = box2d(corners_of(p), K, W, H) if p[2] > 1e-6 else None
                u = iou(pb, gb)
                if rank == 0:
                    top['err'].append(e)
                    top['hit'].append(e <= 0.5 * glong)
                    top['iou'].append(u)
                    top['score'].append(float(r['conf'][j]))
                    top['gt_long'].append(glong)
                    if pb is not None and e <= 0.5 * glong:
                        top['ratio'].append(max(pb[2] - pb[0], pb[3] - pb[1]) / glong)
                scores.append(float(r['conf'][j]))
                for k, ok in (('ctr', e <= 0.5 * glong), ('px10', e <= 10.0), ('iou50', u >= 0.5), ('iou30', u >= 0.3)):
                    if ok and not used[k]:
                        det[k].append(1)
                        used[k] = True
                    else:
                        det[k].append(0)
            if len(order) == 0:
                top['err'].append(np.inf)
                top['hit'].append(False)
                top['iou'].append(0.0)
                top['score'].append(0.0)
    err = np.array(top['err'])
    fin = err[np.isfinite(err)]
    out = {
        'n_gt': n_gt,
        'top1_score_med': float(np.median(top['score'])),
        'ctr_err_med': float(np.median(fin)), 'ctr_err_p90': float(np.percentile(fin, 90)),
        'acc_5px': float((err <= 5).mean()), 'acc_10px': float((err <= 10).mean()), 'acc_20px': float((err <= 20).mean()),
        'hit_rate': float(np.mean(top['hit'])),
        'iou_med': float(np.median(top['iou'])), 'iou_ge50': float((np.array(top['iou']) >= 0.5).mean()),
        'size_ratio_med': float(np.median(top['ratio'])) if top['ratio'] else float('nan'),
        'size_ratio_p10': float(np.percentile(top['ratio'], 10)) if top['ratio'] else float('nan'),
        'size_ratio_p90': float(np.percentile(top['ratio'], 90)) if top['ratio'] else float('nan'),
        'AP_ctr': voc_ap(scores, det['ctr'], n_gt), 'AP_10px': voc_ap(scores, det['px10'], n_gt),
        'AP_IoU30': voc_ap(scores, det['iou30'], n_gt), 'AP_IoU50': voc_ap(scores, det['iou50'], n_gt),
        'gt_long_med': float(np.median(top['gt_long'])) if top['gt_long'] else float('nan'),
    }
    return out, err, np.array(top['iou'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=int, default=1)
    ap.add_argument('--only', default=None)
    ap.add_argument('--out', default='../output/camnorm/eval_2d')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results, curves = {}, {}
    for name, cfgf, ckpt, desc in MODELS:
        if a.only and name not in a.only.split(','):
            continue
        if not os.path.exists(ckpt):
            print('跳过 %s：没有 %s' % (name, ckpt), flush=True)
            continue
        recs, W, H = infer(cfgf, ckpt, a.interval)
        res, err, ious = evaluate(recs, W, H)
        res['desc'] = desc
        results[name] = res
        curves[name] = (err, ious)
        print('%-5s 中心误差中位 %.1f px | 命中 %.1f%% | IoU 中位 %.2f | AP_ctr %.1f | AP_IoU50 %.1f' % (
            name, res['ctr_err_med'], 100 * res['hit_rate'], res['iou_med'], 100 * res['AP_ctr'], 100 * res['AP_IoU50']), flush=True)
        json.dump(results, open(os.path.join(a.out, 'eval_2d.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        np.savez(os.path.join(a.out, 'curves.npz'), **{k + '_err': v[0] for k, v in curves.items()},
                 **{k + '_iou': v[1] for k, v in curves.items()})

    print()
    print('MAV6D test，每帧 1 架；像素在 512x288 输入上（GT 框长边中位 %.0f px）' % next(iter(results.values()))['gt_long_med'])
    print('%-5s %6s | %7s %7s %6s %6s %6s %6s | %6s %6s | %6s %6s %6s %6s | %-6s' % (
        '模型', '峰值分', '中心中位', 'p90', '<=5px', '<=10px', '<=20px', '命中', 'IoU中位', 'IoU>=.5',
        'AP_ctr', 'AP10px', 'AP.3', 'AP.5', '框大小比 p10/中位/p90'))
    for name, r in results.items():
        print('%-5s %6.2f | %7.1f %7.1f %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %6.2f %5.1f%% | %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %.2f / %.2f / %.2f' % (
            name, r['top1_score_med'], r['ctr_err_med'], r['ctr_err_p90'], 100 * r['acc_5px'], 100 * r['acc_10px'],
            100 * r['acc_20px'], 100 * r['hit_rate'], r['iou_med'], 100 * r['iou_ge50'], 100 * r['AP_ctr'],
            100 * r['AP_10px'], 100 * r['AP_IoU30'], 100 * r['AP_IoU50'],
            r['size_ratio_p10'], r['size_ratio_med'], r['size_ratio_p90']))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for name, (err, ious) in curves.items():
        ls = '-' if not name.endswith('*') else '--'
        lw = 2.6 if name in ('R1', 'Real') else 1.5
        xs = np.linspace(0, 100, 201)
        axs[0].plot(xs, [(err <= x).mean() for x in xs], ls, lw=lw, label=name)
        xs = np.linspace(0, 1, 101)
        axs[1].plot(xs, [(ious >= x).mean() for x in xs], ls, lw=lw, label=name)
    axs[0].set_title('top-1 中心误差累计分布（512 宽输入像素）')
    axs[0].set_xlabel('误差 <= x px')
    axs[0].set_ylabel('帧占比')
    axs[1].set_title('top-1 2D 框 IoU（IoU >= x 的帧占比）')
    axs[1].set_xlabel('IoU')
    for ax in axs:
        ax.grid(alpha=.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'eval_2d.png'), dpi=90)
    print('图 ->', os.path.join(a.out, 'eval_2d.png'))


if __name__ == '__main__':
    main()
