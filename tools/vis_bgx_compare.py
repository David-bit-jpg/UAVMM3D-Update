# -*- coding: utf-8 -*-
"""把同一批帧在几个翻译变体缓存里的结果并排：原图 | 变体1 | 变体2 | ...，外加目标周围 4 倍放大的一行。

    D:/SD/venv/Scripts/python.exe tools/vis_bgx_compare.py --src E:/mmcache/demo_crop --split train \
        --variants "A 0.6 skyline=E:/mmcache/demo_A" "B 0.45 neutral=E:/mmcache/demo_B" "C 0.6 neutral 30步=E:/mmcache/demo_C" \
        --out ../output/bgx_compare --rows 6

只依赖 numpy / PIL（在 D:\\SD 的 venv 里能跑）。绿框只画在原图列（定位用），其它列不画，便于看贴回处的接缝。
"""
import argparse
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2real_diffuse import box_uv, draw_box  # noqa: E402
from sim2real_bg_translate import cache_K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--variants', nargs='+', required=True, help='"标题=缓存目录" 列表')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', type=int, default=6, help='每张大图几帧')
    ap.add_argument('--scale', type=float, default=1.5, help='整图显示倍率')
    ap.add_argument('--zoom', type=int, default=4, help='目标区域放大倍率')
    ap.add_argument('--zoom-win', default='128x72', help='目标区域窗口（缓存像素）')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    src_split = os.path.join(args.src, args.split)
    idx = pickle.load(open(src_split + '/index.pkl', 'rb'))
    metas, W, H = idx['metas'], int(idx['W']), int(idx['H'])
    src_rgb = np.load(src_split + '/rgb.npy', mmap_mode='r')
    names, rgbs, trs, strs = [], [], [], []
    for v in args.variants:
        name, path = v.split('=', 1)
        d = os.path.join(path, args.split)
        names.append(name)
        rgbs.append(np.load(d + '/rgb.npy', mmap_mode='r'))
        trs.append(np.load(d + '/translated.npy'))
        sp = d + '/strength.npy'
        strs.append(np.load(sp) if os.path.exists(sp) else None)
    common = np.where(np.logical_and.reduce(trs))[0]
    # 顺序：按第一个变体的翻译顺序不可知，这里按天气分组再按索引，便于对照
    common = sorted(common, key=lambda i: (metas[i]['seq'].split('/')[3], i))
    if args.limit:
        common = common[:args.limit]
    print('共同已翻译帧', len(common))

    zw, zh = [int(v) for v in args.zoom_win.split('x')]
    pw, ph = int(W * args.scale), int(H * args.scale)
    ncol = 1 + len(names)
    zcol_w = zw * args.zoom
    row_h = ph + zh * args.zoom + 8 + 20
    sheet_w = ncol * (pw + 8)
    k = 0
    for s0 in range(0, len(common), args.rows):
        ids = common[s0:s0 + args.rows]
        sheet = Image.new('RGB', (sheet_w, row_h * len(ids)), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for r, i in enumerate(ids):
            m = metas[i]
            K = cache_K(m, W, H)
            boxes = [np.asarray(b, np.float64) for b in m['boxes9d']]
            # 主目标 = 角尺度最大的合格目标
            cands = [(max(b[3:6]) / max(b[2], 1e-3), b) for b, q in zip(boxes, m['qualified']) if q] or [(0, boxes[0])]
            b = max(cands, key=lambda t: t[0])[1]
            uv = box_uv(b, K)
            cx, cy = float(uv[:, 0].mean()), float(uv[:, 1].mean())
            x0 = int(np.clip(cx - zw / 2, 0, W - zw))
            y0 = int(np.clip(cy - zh / 2, 0, H - zh))
            y = r * row_h
            imgs = [Image.fromarray(np.ascontiguousarray(src_rgb[i]))] + [Image.fromarray(np.ascontiguousarray(g[i])) for g in rgbs]
            labels = ['original  %s | %s | %d target(s)' % (m['seq'].split('/')[3], os.path.splitext(m['frame'])[0], len(boxes))]
            for n, st in zip(names, strs):
                labels.append('%s%s' % (n, ('  s=%.2f' % st[i]) if st is not None and st[i] > 0 else ''))
            for c, (im, lab) in enumerate(zip(imgs, labels)):
                x = c * (pw + 8)
                big = im.copy()
                if c == 0:
                    for bb in boxes:
                        draw_box(big, bb, K, width=1, label='%.1fm' % bb[2])
                big = big.resize((pw, ph), Image.BILINEAR)
                sheet.paste(big, (x, y + 20))
                d.text((x + 4, y + 4), lab, fill=(255, 255, 255))
                z = im.crop((x0, y0, x0 + zw, y0 + zh)).resize((zcol_w, zh * args.zoom), Image.NEAREST)
                sheet.paste(z, (x, y + 20 + ph + 8))
        op = os.path.join(args.out, 'compare_%02d.jpg' % k)
        sheet.save(op, quality=88)
        print('写出', op, sheet.size)
        k += 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
