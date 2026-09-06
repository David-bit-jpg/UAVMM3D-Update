# -*- coding: utf-8 -*-
"""半监督迁移（自训练）第一步：用一个已在 MAV6D 上微调过的权重，给训练集里【预算之外、没有位姿标签的帧】打伪标签，
写进 MAV6D 根目录下的一对新目录（E 盘不是 NTFS，做不了目录联接，所以不建平行根目录，图片原地复用）：
    <root>/<cls>/labels_<suffix>/...      -> 预算帧 = 原真标签；其余帧 = 伪标签（与原标签同格式，前 9 个数原样复制，后 7 个数换成预测位姿）；
                                             测试帧的真标签也复制进来（训练末评测用）
    <root>/<cls>/split_<suffix>/train.txt -> 预算帧 + 置信度达标的伪标签帧；test.txt 原样复制
训练时：--set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.LABEL_DIR labels_<suffix> DATA_CONFIG.SPLIT_DIR split_<suffix> DATA_CONFIG.SAMPLED_INTERVAL.train 1
（数据集的 LABEL_DIR / SPLIT_DIR 两个键缺省仍是 labels / split，不影响别的实验）。
预算帧的定义与 MAV6D_Det_Dataset.include_MAV6D_data 完全一致：两类 split 列表拼接后每 --interval 取 1。
伪标签质量用真标签评估（只写进 stats.json 供报告，训练看不到）。
    D:/Miniconda3/envs/city/python.exe tools/mav6d_pseudo_label.py --ckpt <p05 权重> --interval 20 --tag MT_p05 --suffix pseudo_MT_p05
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_on_mav6d import MAV6D_CFG, infer_hm_channels  # noqa: E402
from uavdet3d.config import cfg_from_yaml_file  # noqa: E402
from uavdet3d.datasets import build_dataloader  # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu  # noqa: E402
from uavdet3d.utils import common_utils  # noqa: E402

# 与 uavdet3d/datasets/mav6d/mav6d_utils.read_truth_Rt 里的矩阵一致
CAM2VICON = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                      [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                      [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                      [0, 0, 0, 1]])
CLASSES = ['mavic2', 'phantom4']


def label_to_box(line):
    """原标签行 -> 相机系 (R, t)（与 read_truth_Rt 一致）。"""
    pose = [float(x) for x in line.strip().split(' ')]
    u = pose[9:]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([u[-4], u[-3], u[-2], u[-1]]).as_matrix()
    T[:3, 3] = u[0:3]
    M = CAM2VICON @ T
    return M[:3, :3], M[:3, 3]


def box_to_label_line(orig_line, box9d):
    """预测的相机系 9D 框 -> 原格式标签行（前 9 个数照抄，后 7 个数 = VICON 系下的 xyz + 四元数 xyzw）。"""
    nums = orig_line.strip().split(' ')
    assert len(nums) == 16, '标签格式不是 16 个数: %d' % len(nums)
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', box9d[6:9]).as_matrix()
    T[:3, 3] = box9d[:3]
    V = np.linalg.inv(CAM2VICON) @ T
    q = R.from_matrix(V[:3, :3]).as_quat()
    vals = list(V[:3, 3]) + list(q)
    return ' '.join(nums[:9] + ['%.10g' % v for v in vals]) + '\n'


def selftest():
    line = '1 0 0 0 0 0 0 1 1 -2.0770854045 -2.32592433316 1.68957559722 -0.00979701357452 0.0352060809993 0.336171945944 0.941091373431'
    Rm, t = label_to_box(line)
    box = np.r_[t, [0.34, 0.34, 0.23], R.from_matrix(Rm).as_euler('xyz')]
    Rm2, t2 = label_to_box(box_to_label_line(line, box))
    err_t = float(np.abs(t2 - t).max())
    err_r = float(np.degrees((R.from_matrix(Rm2).inv() * R.from_matrix(Rm)).magnitude()))
    assert err_t < 1e-6 and err_r < 1e-4, (err_t, err_r)
    print('标签往返自检通过：位置 %.1e m，旋转 %.1e°' % (err_t, err_r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--root', default='E:/MAV6D')
    ap.add_argument('--suffix', required=True, help='写到 <root>/<cls>/labels_<suffix> 与 split_<suffix>')
    ap.add_argument('--interval', type=int, default=20, help='预算帧采样间隔（与微调时 SAMPLED_INTERVAL.train 相同）')
    ap.add_argument('--score', type=float, default=0.3, help='伪标签置信度下限')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=0, help='冒烟：只推理前 N 帧')
    args = ap.parse_args()
    selftest()
    t0 = time.time()
    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(MAV6D_CFG, cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.root
    cfg.DATA_CONFIG.DATA_SPLIT.test = 'train'          # 用评测流程遍历训练集全部帧
    cfg.DATA_CONFIG.SAMPLED_INTERVAL.test = 1
    cfg.MODEL.DENSE_HEAD_2D.SEPARATE_HEAD_CFG.HEAD_DICT.hm.out_channels = infer_hm_channels(args.ckpt)
    ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, batch_size=args.batch_size, dist=False,
                                     workers=args.workers, logger=logger, training=False)
    entries = [tuple(x) for x in ds.sample_scene_list]          # (cls, scene, seq, frame)，split 顺序
    labeled = set(entries[i] for i in range(0, len(entries), args.interval))
    print('训练集 %d 帧，预算帧 %d（间隔 %d）' % (len(entries), len(labeled), args.interval))

    model = build_network(model_cfg=cfg.MODEL, dataset=ds)
    model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()
    preds = {}
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            batch = model(batch)
            for b in range(batch['batch_size']):
                key = (str(batch['scene_id'][b]), str(batch['seq_id'][b]), str(batch['frame_id'][b]))
                pred, conf = batch['pred_boxes9d'][b], batch['confidence'][b]
                if pred is None or len(pred) == 0:
                    preds[key] = None
                else:
                    k = int(np.argmax(np.asarray(conf)))
                    preds[key] = (np.asarray(pred)[k].astype(np.float64), float(np.asarray(conf)[k]))
                n_done += 1
            if args.max_frames and n_done >= args.max_frames:
                break
    print('推理 %d 帧，%.0fs' % (n_done, time.time() - t0))

    # ---- 组装新的标签 / split 目录（图片原地复用）----
    ldir, sdir = 'labels_' + args.suffix, 'split_' + args.suffix
    kept = {c: [] for c in CLASSES}
    stat = {'tag': args.tag, 'ckpt': args.ckpt, 'interval': args.interval, 'score': args.score,
            'label_dir': ldir, 'split_dir': sdir,
            'n_total': len(entries), 'n_labeled': len(labeled), 'n_pseudo': 0, 'n_no_det': 0, 'n_low_conf': 0}
    pos_err, ang_err, confs = [], [], []
    for cls in CLASSES:
        for d in (ldir, sdir):
            if os.path.exists(os.path.join(args.root, cls, d)):
                shutil.rmtree(os.path.join(args.root, cls, d))
        os.makedirs(os.path.join(args.root, cls, sdir), exist_ok=True)
        # 测试 split 与其标签原样复制
        shutil.copy(os.path.join(args.root, cls, 'split', 'test.txt'), os.path.join(args.root, cls, sdir, 'test.txt'))
        for line in open(os.path.join(args.root, cls, 'split', 'test.txt')):
            sc, sq, fr = line.strip().split('/')[-3:]
            src = os.path.join(args.root, cls, 'labels', sc, sq, os.path.splitext(fr)[0] + '.txt')
            dst = os.path.join(args.root, cls, ldir, sc, sq, os.path.splitext(fr)[0] + '.txt')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
    for e in entries:
        cls, sc, sq, fr = e
        src = os.path.join(args.root, cls, 'labels', sc, sq, os.path.splitext(fr)[0] + '.txt')
        dst = os.path.join(args.root, cls, ldir, sc, sq, os.path.splitext(fr)[0] + '.txt')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if e in labeled:
            shutil.copy(src, dst)
            kept[cls].append('%s/%s/%s' % (sc, sq, fr))
            continue
        key = (sc, sq, fr)
        if key not in preds:
            continue                                   # 冒烟时没推理到的帧
        p = preds[key]
        if p is None:
            stat['n_no_det'] += 1
            continue
        box, conf = p
        if conf < args.score:
            stat['n_low_conf'] += 1
            continue
        orig = open(src).readline()
        open(dst, 'w').write(box_to_label_line(orig, box))
        kept[cls].append('%s/%s/%s' % (sc, sq, fr))
        stat['n_pseudo'] += 1
        Rg, tg = label_to_box(orig)
        pos_err.append(float(np.linalg.norm(box[:3] - tg)))
        ang_err.append(float(np.degrees((R.from_euler('xyz', box[6:9]).inv() * R.from_matrix(Rg)).magnitude())))
        confs.append(conf)
    for cls in CLASSES:
        open(os.path.join(args.root, cls, sdir, 'train.txt'), 'w').write(''.join(l + '\n' for l in kept[cls]))
        stat['n_train_%s' % cls] = len(kept[cls])
    if pos_err:
        stat.update({'pseudo_pos_median': float(np.median(pos_err)), 'pseudo_pos_mean': float(np.mean(pos_err)),
                     'pseudo_ang_median': float(np.median(ang_err)), 'pseudo_ang_mean': float(np.mean(ang_err)),
                     'pseudo_conf_mean': float(np.mean(confs)),
                     'pseudo_acc_0.2': float(np.mean(np.array(pos_err) < 0.2))})
    for cls in CLASSES:
        json.dump(stat, open(os.path.join(args.root, cls, sdir, 'stats.json'), 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(stat, indent=1, ensure_ascii=False))
    print('写到 %s/<cls>/{%s,%s}，%.0fs' % (args.root, ldir, sdir, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
