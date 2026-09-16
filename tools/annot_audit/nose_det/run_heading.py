# -*- coding: utf-8 -*-
"""nose_det：机头（机体 x 轴）约定核对 —— tools/diag_heading_convention.py 的逐样本落盘版。

口径与原脚本相同：真实微调模型在仿真测试集上预测的机体 x 轴（= 它学到的 MAV6D「视觉机头」），
投到 GT 机体 z 的水平面上，与 GT 标签 x 轴的带符号夹角 yaw = atan2((a x b)·z_gt, a·b)，
    yaw > 0 : 视觉机头相对标签 x 绕机体 z（朝上）逆时针（俯视）转了 yaw；
    等价地，标签 x 相对视觉机头顺时针转了 |yaw|。
与原脚本的两点不同：
  1) 机型归属：原脚本把 batch 里的 GT 框与 meta['names'] 按位置 zip。裁窗（zoom>1）会丢掉中心出窗的框，
     多机帧会错位。这里把 batch 的 GT 框与 meta['boxes9d'] 逐行精确匹配回去取名字，同时记下原 zip 归属以量化错位。
  2) 逐 GT 记录到 CSV（yaw、2D 误差、置信度、倾斜差、总角差等），供多次运行按 n 加权合并。

    python annot_audit/nose_det/run_heading.py --ckpt <best.pth> --tag T_p025_s0 --zoom 2.5 --interval 1
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, HERE)
from uavdet3d.config import cfg_from_list, cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network, load_data_to_gpu   # noqa: E402
from uavdet3d.utils import common_utils   # noqa: E402
from nose_common import OUT_ROOT, table   # noqa: E402

COLS = ['tag', 'zoom', 'seq', 'frame', 'gi', 'n_gt_batch', 'n_gt_meta', 'name', 'name_zip', 'meta_box_idx',
        'n_pred', 'matched', 'e2d_px', 'app_px', 'conf', 'gt_z', 'gt_maxdim', 'pred_z', 'pred_maxdim',
        'yaw', 'tilt', 'ang']


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--interval', type=int, default=1)
    ap.add_argument('--zoom', default='2.5')
    ap.add_argument('--out', default=OUT_ROOT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
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

    meta_by_key = {}
    for i in ds.valid_idx:
        m = ds.metas[int(i)]
        key = (m['seq'], m['frame'])
        assert key not in meta_by_key, '帧键重复 %s' % (key,)
        meta_by_key[key] = m

    rows = []
    n_frames = 0
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            out = model(batch)
            for b in range(out['batch_size']):
                n_frames += 1
                gt = _np(batch['gt_box9d'][b]).astype(np.float64).reshape(-1, 9)
                gt = gt[np.abs(gt).sum(1) > 0]
                pred = out['pred_boxes9d'][b]
                conf = out['confidence'][b]
                pred = _np(pred).astype(np.float64) if pred is not None else np.zeros((0, 10))
                pred = pred.reshape(len(pred), -1)[:, :9] if len(pred) else np.zeros((0, 9))
                conf = _np(conf).astype(np.float64).reshape(-1) if conf is not None and len(conf) else np.zeros(0)
                K = _np(batch['intrinsic'][b]).astype(np.float64).reshape(-1, 3, 3)[0]
                seq, frame = str(batch['seq_id'][b]), str(batch['frame_id'][b])
                meta = meta_by_key[(seq, frame)]
                mb = np.asarray(meta['boxes9d'], dtype=np.float32).reshape(-1, 9)
                mnames = list(meta['names'])
                if len(pred):
                    up_ = (K @ pred[:, :3].T).T
                    up_ = up_[:, :2] / up_[:, 2:3]
                for gi, g in enumerate(gt):
                    # 机型：精确匹配回 meta（测试模式不动 3D 框，只有 float32 舍入）
                    d_meta = np.linalg.norm(mb - g.astype(np.float32)[None], axis=1)
                    kk = int(np.argmin(d_meta))
                    assert d_meta[kk] < 1e-3, '框匹配不回 meta：%s %s %.4f' % (seq, frame, d_meta[kk])
                    name = mnames[kk]
                    name_zip = mnames[gi] if gi < len(mnames) else ''
                    row = {'tag': args.tag, 'zoom': args.zoom, 'seq': seq, 'frame': frame, 'gi': gi,
                           'n_gt_batch': len(gt), 'n_gt_meta': len(mnames), 'name': name, 'name_zip': name_zip,
                           'meta_box_idx': kk, 'n_pred': len(pred), 'matched': 0, 'e2d_px': np.nan, 'app_px': np.nan,
                           'conf': np.nan, 'gt_z': g[2], 'gt_maxdim': max(g[3:6]), 'pred_z': np.nan,
                           'pred_maxdim': np.nan, 'yaw': np.nan, 'tilt': np.nan, 'ang': np.nan}
                    app = 512.0 * max(g[3:6]) / g[2]          # 与原脚本一致的粗略表观大小
                    row['app_px'] = app
                    if len(pred):
                        ug = K @ g[:3]
                        ug = ug[:2] / ug[2]
                        dd = np.linalg.norm(up_ - ug, axis=1)
                        j = int(np.argmin(dd))
                        p = pred[j]
                        row['e2d_px'] = float(dd[j])
                        row['conf'] = float(conf[j]) if len(conf) > j else np.nan
                        row['pred_z'] = p[2]
                        row['pred_maxdim'] = max(p[3:6])
                        Rg = R.from_euler('xyz', g[6:9]).as_matrix()
                        Rp = R.from_euler('xyz', p[6:9]).as_matrix()
                        zb = Rg[:, 2]
                        a = Rg[:, 0]
                        bb = Rp[:, 0] - (Rp[:, 0] @ zb) * zb
                        row['tilt'] = float(np.degrees(np.arccos(np.clip(zb @ Rp[:, 2], -1, 1))))
                        row['ang'] = float(np.degrees((R.from_matrix(Rg).inv() * R.from_matrix(Rp)).magnitude()))
                        if dd[j] <= 0.5 * max(app, 1.0) and np.linalg.norm(bb) >= 1e-3:
                            bb /= np.linalg.norm(bb)
                            row['yaw'] = float(np.degrees(np.arctan2(np.cross(a, bb) @ zb, a @ bb)))
                            row['matched'] = 1
                    rows.append(row)

    ztag = str(args.zoom).replace('.', 'p')
    csv_path = os.path.join(args.out, 'heading_%s_z%s.csv' % (args.tag, ztag))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_gt = len(rows)
    n_det = sum(r['matched'] for r in rows)
    n_mis = sum(1 for r in rows if r['name'] != r['name_zip'])
    per, per_zip = {}, {}
    for r in rows:
        if r['matched']:
            per.setdefault(r['name'], []).append(r['yaw'])
            per_zip.setdefault(r['name_zip'], []).append(r['yaw'])
    head = ('%s：仿真测试集（裁窗 %s 倍，interval %d），帧 %d，GT %d，2D 对上 %d（%.1f%%），'
            '原脚本 zip 归属错位的 GT %d（%.1f%%），耗时 %.0f s'
            % (args.tag, args.zoom, args.interval, n_frames, n_gt, n_det, 100.0 * n_det / max(n_gt, 1),
               n_mis, 100.0 * n_mis / max(n_gt, 1), time.time() - t0))
    txt = head + '\n' + table(per, '[机型按 meta 框精确匹配]') + '\n' + table(per_zip, '[机型按原脚本 zip 归属（仅供对照）]')
    print(txt)
    with open(os.path.join(args.out, 'heading_%s_z%s.txt' % (args.tag, ztag)), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
