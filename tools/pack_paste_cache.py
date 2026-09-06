# -*- coding: utf-8 -*-
"""把 mm_paste_aug.py 的 .npz 样本打包成训练缓存（与 MMCache_Det_Dataset 同布局），外加点云与说明。

    D:/Miniconda3/envs/city/python.exe tools/pack_paste_cache.py --samples E:/data_collect/aug_paste_v1/samples \
        --out E:/data_collect/aug_paste_v1/cache

输出：
  <out>/train/{rgb,ir,depth,tag,dvs}.npy, radar_hm.npy(float16), index.pkl, vis_score.npy, translated.npy
  <out>_teacher/train/rgb.npy        —— 仿真背景版 RGB（教师输入；数据集配置 TEACHER_RGB_DIR 指到 <out>_teacher）
  <out>/pointclouds/<name>_lidar.npy / _radar.npy   —— 增广后的点云（B 帧 LiDAR / 雷达传感器系）
  <out>/README.md                                    —— 生成参数与统计
"""
import argparse
import collections
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mm_cache import CLASSES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--note', default='', help='写进 README 的一句话（生成命令 / 参数）')
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.samples, '*.npz')))
    assert files, '没有样本'
    z0 = np.load(files[0], allow_pickle=True)
    H, W = z0['rgb'].shape[:2]
    N = len(files)
    od = os.path.join(args.out, args.split)
    ot = os.path.join(args.out + '_teacher', args.split)
    pc = os.path.join(args.out, 'pointclouds')
    for p in (od, ot, pc):
        os.makedirs(p, exist_ok=True)
    mm = {
        'rgb': np.lib.format.open_memmap(os.path.join(od, 'rgb.npy'), 'w+', np.uint8, (N, H, W, 3)),
        'ir': np.lib.format.open_memmap(os.path.join(od, 'ir.npy'), 'w+', np.uint8, (N, H, W)),
        'depth': np.lib.format.open_memmap(os.path.join(od, 'depth.npy'), 'w+', np.uint16, (N, H, W)),
        'tag': np.lib.format.open_memmap(os.path.join(od, 'tag.npy'), 'w+', np.uint8, (N, H, W)),
        'dvs': np.lib.format.open_memmap(os.path.join(od, 'dvs.npy'), 'w+', np.uint8, (N, H, W, 3)),
        'radar_hm': np.lib.format.open_memmap(os.path.join(od, 'radar_hm.npy'), 'w+', np.float16, (N, H, W)),
        'rgb_t': np.lib.format.open_memmap(os.path.join(ot, 'rgb.npy'), 'w+', np.uint8, (N, H, W, 3)),
    }
    metas = []
    stats = collections.defaultdict(list)
    t0 = time.time()
    for k, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        name = os.path.splitext(os.path.basename(f))[0]
        mm['rgb'][k] = z['rgb']
        mm['rgb_t'][k] = z['rgb_sim']
        mm['ir'][k] = z['ir']
        mm['depth'][k] = z['depth']
        mm['tag'][k] = z['tag']
        mm['dvs'][k] = z['dvs']
        mm['radar_hm'][k] = z['radar_hm'].astype(np.float16)
        np.save(os.path.join(pc, name + '_lidar.npy'), z['lidar_pts'].astype(np.float32))
        np.save(os.path.join(pc, name + '_radar.npy'), z['radar_pts'].astype(np.float32))
        ck = z['checks'].item() if 'checks' in z.files else {}
        b = z['box9d'].astype(np.float32)
        cls = str(z['name'])
        metas.append({
            'seq': 'aug_paste/' + name.split('_')[1] + '_' + name.split('_')[2], 'frame': name + '.png',
            'K_raw': np.asarray(z['K_raw'], np.float32), 'raw_wh': tuple(int(v) for v in z['raw_wh']),
            'boxes9d': b[None], 'names': [cls], 'qualified': np.array([True]),
            'lidar_noise': float(ck.get('sigma_B', -1)), 'ir_shift': np.zeros(2, np.float32), 'lidar_frame': '',
            'crop_xy': (0, 0), 'crop_wh': tuple(int(v) for v in z['raw_wh']),
            'src': str(z['A']), 'bg': str(z['B']), 's': float(z['s']),
            'range_old': float(ck.get('range_old', -1)), 'range_new': float(np.linalg.norm(b[:3])),
            'px': float(ck.get('px_new', -1)), 'checks': {kk: v for kk, v in ck.items() if not isinstance(v, str)},
        })
        stats['cls'].append(cls)
        stats['weather'].append(metas[-1]['seq'].split('/')[1])
        stats['range'].append(metas[-1]['range_new'])
        stats['px'].append(metas[-1]['px'])
        stats['s'].append(float(z['s']))
        if (k + 1) % 200 == 0:
            print('  %d/%d  %.1fs' % (k + 1, N, time.time() - t0), flush=True)
    for v in mm.values():
        v.flush()
    index = {'W': W, 'H': H, 'classes': CLASSES, 'split': args.split, 'source': 'tools/mm_paste_aug.py', 'metas': metas,
             'valid_idx': np.arange(N)}
    pickle.dump(index, open(os.path.join(od, 'index.pkl'), 'wb'))
    np.save(os.path.join(od, 'vis_score.npy'), np.full(N, 100.0, np.float32))
    np.save(os.path.join(od, 'translated.npy'), np.ones(N, bool))
    open(os.path.join(od, 'READY'), 'w').write('ok\n')
    # ---- README ----
    cc = collections.Counter(stats['cls'])
    wc = collections.Counter(stats['weather'])
    rg, px, ss = np.array(stats['range']), np.array(stats['px']), np.array(stats['s'])
    lines = ['# aug_paste 交叉贴增广数据（%s）' % time.strftime('%Y-%m-%d %H:%M'), '',
             '样本 %d 个，分辨率 %dx%d（与训练缓存一致）。生成：`tools/mm_paste_aug.py`，打包：`tools/pack_paste_cache.py`。' % (N, W, H), '']
    if args.note:
        lines += ['生成命令 / 参数：', '', '    ' + args.note, '']
    lines += ['## 规则', '',
              '- 源帧 A：白天、RGB 可见、最近的合格无人机；用 GrabCut 紧致 matte 抠出机身（边缘去色染），其余无人机全部抹掉。',
              '- 背景帧 B：整帧全部无人机用 inpaint 抹掉，RGB 再用本地 SDXL img2img（强度 0.45，中性提示词）翻成照片质感；IR/DVS 在各自相机里抹除、按新目标深度对齐。',
              '- 位置：随机像素；距离：先抽 MAV6D 真实距离分布（2.0/3.4/4.9 m 为 5/50/95 分位），再按机型框边长等比例放远 r_new = r × max(dim)/0.34；缩放倍数由距离物理推出（≤2.0）；表观 40–140 px。',
              '- 几何：绕相机中心的旋转（像素为精确单应）+ 沿视线平移；3D 框中心/旋转同变换；LiDAR 与雷达的无人机点做同一刚体变换后写回 B 的传感器系点云，'
              '深度/tag/雷达热图从新点云重投影；LiDAR 先沿 A 的射线去噪收回框内，再沿 B 的 LiDAR 射线按 B 序列的 σ 加噪。',
              '- 机型轮流选，每帧一个目标；色调匹配 0.6。', '',
              '## 统计', '',
              '| 项目 | 值 |', '|---|---|',
              '| 机型 | ' + ', '.join('%s %d' % (k, v) for k, v in sorted(cc.items())) + ' |',
              '| 天气 | ' + ', '.join('%s %d' % (k, v) for k, v in sorted(wc.items())) + ' |',
              '| 距离 m（p05/p50/p95） | %.1f / %.1f / %.1f |' % tuple(np.percentile(rg, [5, 50, 95])),
              '| 表观 px（p05/p50/p95） | %.0f / %.0f / %.0f |' % tuple(np.percentile(px, [5, 50, 95])),
              '| 缩放倍数（p05/p50/p95） | %.2f / %.2f / %.2f |' % tuple(np.percentile(ss, [5, 50, 95])), '',
              '## 目录', '',
              '- `%s/train/`：rgb（翻译背景，学生输入）、ir、depth（cm，uint16）、tag、dvs、radar_hm（float16）、index.pkl、vis_score.npy、translated.npy' % os.path.basename(args.out),
              '- `%s_teacher/train/rgb.npy`：仿真背景版 RGB（教师输入），数据集配置里 `TEACHER_RGB_DIR` 指到该目录' % os.path.basename(args.out),
              '- `pointclouds/`：每个样本增广后的 LiDAR（x,y,z,intensity,tag，B 帧 LiDAR 系）与雷达（vel,az,alt,depth,tag，B 帧雷达系）点云',
              '- `../samples/`：原始 .npz（含变换 R_cam/t/s、源帧、背景帧、校验值）；`../viewer/index.html`：前 200 个样本的查看器', '',
              '## 用法', '',
              '数据集配置：`DATA_PATH: <本目录>`，`REQUIRE_TRANSLATED: true`，`TEACHER_RGB_DIR: <本目录>_teacher`（见 `tools/cfgs/dataset_configs/uavdet_3d/mmcache_crop_bgx.yaml`）。', '']
    open(os.path.join(args.out, 'README.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    json.dump({'n': N, 'cls': dict(cc), 'weather': dict(wc)}, open(os.path.join(args.out, 'stats.json'), 'w'), ensure_ascii=False, indent=1)
    print('打包完成：%d 样本 -> %s（%.0fs）' % (N, args.out, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
