# -*- coding: utf-8 -*-
"""算一个 mmcache 缓存自己的 RGB 归一化统计量，打印成可直接粘进配置的 YAML 行。

为什么要有这个工具（2026-09-16）：`build_mm_cache.py` 不算归一化统计量，只有 `build_mav6d_cache.py` 算。
新建仿真缓存时，配置往往只改 DATA_PATH 而沿用上一个缓存的 NORM_MEAN/NORM_STD ——
`camnorm_sim_pp_realsize.yaml` 就是这样，用了旧 indoor8 八场景的常数去归一化新采的单场景 PowerPlant，
结果网络实际看到的输入是 均值 [+0.25,+0.11,-0.13]、标准差 [0.78,1.01,0.88]（应为 0 / 1），
凭空制造了一个色偏 + 对比度差。**建完新缓存必须跑一次这个脚本并把结果写进配置。**

约定与 build_mav6d_cache.py 一致：缓存原始帧（不裁窗、不增广）、BGR 顺序、x/255 后、float64 累加。

    python tools/calc_cache_norm.py E:/mmcache/pp_realsize [--split train] [--frames 600]
"""
import argparse
import os
import pickle

import cv2
import numpy as np


def iter_frames(split_dir, sel):
    jb = os.path.join(split_dir, 'rgb_jpg.bin')
    if os.path.exists(jb):
        ji = np.load(os.path.join(split_dir, 'rgb_jpg_index.npy'), mmap_mode='r')
        with open(jb, 'rb') as f:
            for i in sel:
                o, ln = ji[int(i)]
                f.seek(int(o))
                img = cv2.imdecode(np.frombuffer(f.read(int(ln)), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img
    else:
        mm = np.load(os.path.join(split_dir, 'rgb.npy'), mmap_mode='r')
        for i in sel:
            yield np.asarray(mm[int(i)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--split', default='train')
    ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--write', default=None, help='把 NORM_MEAN/NORM_STD 两行就地写进这个 yaml（有则替换）')
    a = ap.parse_args()

    split_dir = os.path.join(a.root, a.split)
    idx = pickle.load(open(os.path.join(split_dir, 'index.pkl'), 'rb'))
    vi = list(idx['valid_idx'])
    sel = vi[:: max(1, len(vi) // a.frames)][:a.frames]

    n = 0
    s1 = np.zeros(3, np.float64)
    s2 = np.zeros(3, np.float64)
    for img in iter_frames(split_dir, sel):
        x = img.reshape(-1, 3).astype(np.float64) / 255.0
        n += len(x)
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 1e-12))
    ml = 'NORM_MEAN: [%.4f, %.4f, %.4f]' % tuple(mean)
    sl = 'NORM_STD: [%.4f, %.4f, %.4f]' % tuple(std)
    print('%s/%s：%d 帧 %d 像素（BGR，x/255）' % (a.root, a.split, len(sel), n))
    print(ml)
    print(sl)

    if a.write:
        txt = open(a.write, encoding='utf-8').read()
        lines, hit = [], False
        for ln in txt.split('\n'):
            if ln.startswith('NORM_MEAN:'):
                lines.append(ml); hit = True
            elif ln.startswith('NORM_STD:'):
                lines.append(sl)
            else:
                lines.append(ln)
        if not hit:
            lines += ['# 本域归一化统计量（tools/calc_cache_norm.py 实测，%s/%s）' % (a.root, a.split), ml, sl]
        open(a.write, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
        print('-> 已写入', a.write)


if __name__ == '__main__':
    main()
