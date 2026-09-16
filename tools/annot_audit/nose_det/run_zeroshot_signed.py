# -*- coding: utf-8 -*-
"""nose_det：纯仿真模型零样本看 MAV6D —— tools/diag_zeroshot_decompose.py 的「带符号机头水平朝向差」补充版。

原脚本只给 |yaw| 的 0~180 分布，分不清「±90 双峰」和「0/180 双峰」。这里复用它的推理与匹配口径
（每 GT 取 2D 最近检测；2D 定位正确 = 中心误差 < 0.5 个表观长边），把绕世界竖直轴（VICON z，经 camera2vicon
转到相机系 ≈ -y_cam）的带符号水平朝向差逐帧落 CSV，并按机型给 30° 直方图。
    yaw > 0 : 预测机头相对标签 x 绕竖直向上轴逆时针（俯视）。

    python annot_audit/nose_det/run_zeroshot_signed.py --ckpt <S1 best.pth> --tag S1
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import diag_zeroshot_decompose as dz   # noqa: E402  只读复用：infer / PROTO8 / CAM2VICON_R
from nose_common import OUT_ROOT, table, stats   # noqa: E402

COLS = ['tag', 'cls', 'seq', 'frame', 'n_pred', 'j_near', 'j_top1', 'conf', 'e2d_ratio', 'app_px', 'depth_ratio',
        'depth_ratio_sizecorr', 'size_ratio', 'ang', 'tilt', 'yaw']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='S1')
    ap.add_argument('--mav-split', default='test')
    ap.add_argument('--out', default=OUT_ROOT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    recs, _ = dz.infer('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', args.mav_split, 1, args.ckpt, None)
    up = dz.CAM2VICON_R @ np.array([0, 0, 1.0])
    rows = []
    for r in recs:
        if len(r['gt']) == 0:
            continue
        K = r['K']
        g = r['gt'][0]
        cls = r['seq_id'].split('/')[0]
        row = {'tag': args.tag, 'cls': cls, 'seq': r['seq_id'], 'frame': r['frame_id'], 'n_pred': len(r['pred']),
               'j_near': -1, 'j_top1': -1, 'conf': np.nan, 'e2d_ratio': np.nan, 'app_px': np.nan, 'depth_ratio': np.nan,
               'depth_ratio_sizecorr': np.nan, 'size_ratio': np.nan, 'ang': np.nan, 'tilt': np.nan, 'yaw': np.nan}
        gc = (dz.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
        gu = (K @ gc.T).T
        gu = gu[:, :2] / gu[:, 2:3]
        app = max(gu[:, 0].max() - gu[:, 0].min(), gu[:, 1].max() - gu[:, 1].min())
        row['app_px'] = float(app)
        if len(r['pred']):
            ug = K @ g[:3]
            ug = ug[:2] / ug[2]
            up_ = (K @ r['pred'][:, :3].T).T
            up_ = up_[:, :2] / up_[:, 2:3]
            j = int(np.argmin(np.linalg.norm(up_ - ug, axis=1)))
            p = r['pred'][j]
            row['j_near'] = j
            row['j_top1'] = int(np.argmax(r['conf'])) if len(r['conf']) else -1
            row['conf'] = float(r['conf'][j]) if len(r['conf']) > j else np.nan
            row['e2d_ratio'] = float(np.linalg.norm(up_[j] - ug)) / max(app, 1.0)
            Rg = R.from_euler('xyz', g[6:9]).as_matrix()
            Rp = R.from_euler('xyz', p[6:9]).as_matrix()
            row['ang'] = float(np.degrees((R.from_matrix(Rg).inv() * R.from_matrix(Rp)).magnitude()))
            row['tilt'] = float(np.degrees(np.arccos(np.clip(Rg[:, 2] @ Rp[:, 2], -1, 1))))
            hx = [v - (v @ up) * up for v in (Rg[:, 0], Rp[:, 0])]
            if min(np.linalg.norm(hx[0]), np.linalg.norm(hx[1])) > 1e-3:
                a, b = hx[0] / np.linalg.norm(hx[0]), hx[1] / np.linalg.norm(hx[1])
                row['yaw'] = float(np.degrees(np.arctan2(np.cross(a, b) @ up, a @ b)))
            row['size_ratio'] = float(np.max(p[3:6]) / np.max(g[3:6]))
            row['depth_ratio'] = float(p[2] / g[2])
            row['depth_ratio_sizecorr'] = float(p[2] / g[2] / row['size_ratio'])
        rows.append(row)

    csv_path = os.path.join(args.out, 'zeroshot_signed_%s_mav6d_%s.csv' % (args.tag, args.mav_split))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = ['%s 零样本 MAV6D %s：帧 %d，有检测 %d，耗时 %.0f s'
             % (args.tag, args.mav_split, len(rows), sum(r['n_pred'] > 0 for r in rows), time.time() - t0)]
    for lab, cond in (('全部有检测的帧', lambda r: r['n_pred'] > 0),
                      ('2D 定位正确的帧（中心误差 < 0.5 表观长边）', lambda r: r['n_pred'] > 0 and r['e2d_ratio'] < 0.5)):
        per = {}
        for r in rows:
            if cond(r) and np.isfinite(r['yaw']):
                per.setdefault(r['cls'], []).append(r['yaw'])
        lines.append(table(per, '[%s] 带符号机头水平朝向差（绕 VICON 竖直轴）' % lab))
        for cls in sorted(per):
            v = np.abs(np.asarray(per[cls]))
            h = np.histogram(v, bins=[0, 30, 60, 90, 120, 150, 180])[0] / len(v) * 100
            sub = [r for r in rows if cond(r) and r['cls'] == cls]
            ang = np.array([r['ang'] for r in sub])
            lines.append('   %-9s |yaw| 0-30/30-60/60-90/90-120/120-150/150-180: %s | |yaw|>150 %.1f%% | 总角差>150 %.1f%% '
                         '| 总角差中位 %.1f° | 倾斜中位 %.1f° | 深度比中位 %.2f | 尺寸比中位 %.2f | 2D 正确 %.1f%%'
                         % (cls, h.round(1).tolist(), 100 * (v > 150).mean(), 100 * (ang > 150).mean(), np.median(ang),
                            np.median([r['tilt'] for r in sub]), np.median([r['depth_ratio'] for r in sub]),
                            np.median([r['size_ratio'] for r in sub]),
                            100 * np.mean([r['e2d_ratio'] < 0.5 for r in sub])))
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out, 'zeroshot_signed_%s_mav6d_%s.txt' % (args.tag, args.mav_split)), 'w',
              encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
