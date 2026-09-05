# -*- coding: utf-8 -*-
"""背景专用的「仿真 RGB -> 照片质感」翻译：对整个多模态缓存跑，可断点续跑。

用户定的规则（2026-09-06）：
  1. 抠出无人机：帧内【全部有标签目标】（index 里的 boxes9d，40 m 内）的投影凸包，外扩 --dilate 再羽化 --feather，
     这是「不许动」的区域；
  2. 还原背景：把这些区域用周围像素抹成模糊背景（归一化卷积），得到一张没有无人机的底图；
  3. 每帧最多 --max-targets（默认 2）架：目标更多的帧不翻译（translated=False，训练时按此过滤）；
  4. 只翻译背景：SDXL img2img，提示词只写场景 + 天气、不写 drone（写了会把路灯画成无人机），
     负面词含 drone/aircraft；底图从缓存分辨率 512x288 放大到 --gen-wh 再翻译再缩回；
  5. 贴回：凸包内像素 = 缓存原像素，一个不动（逐帧 assert）；只有凸包外 dilate 环里的羽化过渡是
     「像素填充到模糊背景」。

输出：与源缓存同布局的新目录 --dst（rgb.npy 被逐帧替换；ir/depth/tag/index.pkl/vis_score.npy 原样复制），
外加 translated.npy（bool/帧，断点续跑用）、bgx_meta.json、跑完写 READY；--vis-n > 0 时另存对比图。

用法（D:\\SD 的 venv；权重在 D:\\SD\\hf）：
    HF_HOME=D:/SD/hf HF_HUB_OFFLINE=1 D:/SD/venv/Scripts/python.exe tools/sim2real_bg_translate.py \\
        --src E:/mmcache/smoke_crop --split train --dst E:/mmcache/smoke_crop_bgx --vis-n 24 --limit 24
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2real_diffuse import WEATHER_WORDS, box_mask, box_uv, draw_box, fill_background, paste_back  # noqa: E402

os.environ.setdefault('HF_HOME', 'D:/SD/hf')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

BG_PROMPT = ('a real photograph of a city skyline and sky, %s, DSLR photo, sharp focus, '
             'realistic lighting, photorealistic, film grain, high detail')
BG_NEGATIVE = ('cartoon, cgi, render, video game, painting, blurry, low quality, deformed, extra objects, '
               'text, watermark, drone, aircraft, bird, helicopter')


def cache_K(meta, W, H):
    """index 里的 K_raw 是【原始（或裁剪后）分辨率】的内参，换算到缓存分辨率。"""
    raw_w, raw_h = meta['raw_wh']
    K = np.array(meta['K_raw'], dtype=np.float64).copy()
    K[0, :] *= W / float(raw_w)
    K[1, :] *= H / float(raw_h)
    return K


def prepare_dst(src_split, dst_split, n_frames):
    """目标目录：复制不变的模态与索引；rgb.npy 首次复制一份（未翻译的帧保持原像素）。"""
    os.makedirs(dst_split, exist_ok=True)
    for name in ('ir.npy', 'depth.npy', 'tag.npy', 'index.pkl', 'vis_score.npy'):
        s, d = os.path.join(src_split, name), os.path.join(dst_split, name)
        if os.path.exists(s) and not os.path.exists(d):
            print('复制', name, '...', flush=True)
            shutil.copyfile(s, d)
    s, d = os.path.join(src_split, 'rgb.npy'), os.path.join(dst_split, 'rgb.npy')
    if not os.path.exists(d):
        print('复制 rgb.npy ...', flush=True)
        shutil.copyfile(s, d)
    tp = os.path.join(dst_split, 'translated.npy')
    translated = np.load(tp) if os.path.exists(tp) else np.zeros(n_frames, dtype=bool)
    sp = os.path.join(dst_split, 'strength.npy')
    strength_used = np.load(sp) if os.path.exists(sp) else np.zeros(n_frames, dtype=np.float32)
    return translated, strength_used


def bg_luminance(rgb_bgr, mask):
    """目标掩码之外的亮度均值（缓存是 BGR）。"""
    lum = np.asarray(rgb_bgr, np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)
    bg = lum[mask < 0.01]
    return float(bg.mean()) if bg.size else float(lum.mean())


def make_sheet(orig, plate, bg, final, boxes, K, title, path, up=2):
    """原图 | 抹掉目标的底图 | 翻译后的背景（贴回前）| 最终合成，各放大 up 倍，绿框 = 有标签目标。"""
    W, H = orig.size
    panels = [orig.copy(), plate.copy(), bg.copy(), final.copy()]
    for p in (panels[0], panels[3]):
        for b in boxes:
            draw_box(p, b, K, width=1, label='%.1fm' % b[2])
    panels = [p.resize((W * up, H * up), Image.NEAREST) for p in panels]
    titles = ['original (cache px)', 'targets erased -> blurred background plate', 'background translated (before paste)',
              'final: original target pixels pasted back']
    sheet = Image.new('RGB', (W * up * 4 + 8 * 3, H * up + 24), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, (p, t) in enumerate(zip(panels, titles)):
        x = i * (W * up + 8)
        sheet.paste(p, (x, 24))
        d.text((x + 6, 6), t, fill=(255, 255, 255))
    d.text((W * up * 4 - 420, 6), title, fill=(200, 200, 200))
    sheet.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='源缓存根目录，如 E:/mmcache/smoke_crop')
    ap.add_argument('--split', required=True, choices=['train', 'test'])
    ap.add_argument('--dst', required=True, help='输出缓存根目录（同布局）')
    ap.add_argument('--max-targets', type=int, default=2, help='有标签目标多于此数的帧不翻译')
    ap.add_argument('--min-vis', type=float, default=5.0, help='RGB 可见度低于此的帧不翻译（vis_score.npy）')
    ap.add_argument('--min-angsize', type=float, default=0.0, help='最大合格目标的角尺度（max(dim)/z）低于此的帧不翻译')
    ap.add_argument('--strength', type=float, default=0.6)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--guidance', type=float, default=6.0)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--gen-wh', default='1024x576', help='送进 SDXL 的分辨率（底图先放大到这里）')
    ap.add_argument('--dilate', type=int, default=5, help='凸包外扩像素（缓存分辨率）')
    ap.add_argument('--feather', type=int, default=3, help='羽化像素（缓存分辨率）')
    ap.add_argument('--sigma', type=float, default=12.0, help='抹除目标时的模糊半径（缓存分辨率）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='调试：只翻译前 N 个候选帧')
    ap.add_argument('--vis-n', type=int, default=0, help='前 N 个翻译帧另存 4 栏对比图')
    ap.add_argument('--vis-dir', default='')
    ap.add_argument('--flush-every', type=int, default=64)
    ap.add_argument('--no-shuffle', action='store_true', help='按索引顺序翻译（默认按 --seed 打乱候选顺序）')
    # 夜景帧背景几乎全黑（实测 8 种天气的 *_night 帧背景亮度均值 0-24，白天 >= 78）：强度 0.6 会凭空编出车、灯塔；
    # 对这种帧只做弱翻译（加传感器质感、不编内容），阈值按 BGR 亮度均值（目标掩码之外）
    ap.add_argument('--dark-mean', type=float, default=32.0, help='背景亮度均值低于此视为暗帧')
    ap.add_argument('--dark-strength', type=float, default=0.35, help='暗帧用的 img2img 强度')
    ap.add_argument('--only', choices=['', 'dark', 'bright'], default='', help='调试：只翻译暗帧 / 亮帧')
    args = ap.parse_args()

    src_split = os.path.join(args.src, args.split)
    dst_split = os.path.join(args.dst, args.split)
    idx = pickle.load(open(os.path.join(src_split, 'index.pkl'), 'rb'))
    metas, valid = idx['metas'], np.asarray(idx['valid_idx'])
    W, H = int(idx['W']), int(idx['H'])
    gw, gh = [int(v) for v in args.gen_wh.split('x')]
    src_rgb = np.load(os.path.join(src_split, 'rgb.npy'), mmap_mode='r')
    n_frames = src_rgb.shape[0]
    translated, strength_used = prepare_dst(src_split, dst_split, n_frames)
    dst_rgb = np.load(os.path.join(dst_split, 'rgb.npy'), mmap_mode='r+')
    vis_path = os.path.join(src_split, 'vis_score.npy')
    vis = np.load(vis_path) if os.path.exists(vis_path) else None

    # ---- 候选帧 ----
    skipped = {'targets': 0, 'vis': 0, 'angsize': 0, 'done': 0}
    cand = []
    for i in valid:
        i = int(i)
        m = metas[i]
        if translated[i]:
            skipped['done'] += 1
            continue
        if len(m['boxes9d']) > args.max_targets:
            skipped['targets'] += 1
            continue
        if vis is not None and args.min_vis > 0 and vis[i] < args.min_vis:
            skipped['vis'] += 1
            continue
        if args.min_angsize > 0:
            angs = [max(b[3:6]) / max(b[2], 1e-3) for b, q in zip(m['boxes9d'], m['qualified']) if q]
            if not angs or max(angs) < args.min_angsize:
                skipped['angsize'] += 1
                continue
        cand.append(i)
    if not args.no_shuffle:
        # 打乱候选顺序：随时中断都得到一个跨序列/跨天气均匀的已翻译子集；--vis-n 的样例也因此是多样的
        cand = [int(c) for c in np.random.RandomState(args.seed).permutation(np.asarray(cand, dtype=np.int64))]
    if args.limit > 0:
        cand = cand[:args.limit]
    print('缓存 %s/%s：%d 帧有效，已翻译 %d，跳过（目标>%d：%d，可见度<%.1f：%d，角尺度：%d），本次候选 %d'
          % (args.src, args.split, len(valid), skipped['done'], args.max_targets, skipped['targets'],
             args.min_vis, skipped['vis'], skipped['angsize'], len(cand)), flush=True)
    if not cand:
        open(os.path.join(dst_split, 'READY'), 'w').write('ok\n')
        return 0

    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, variant='fp16',
        use_safetensors=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    if args.vis_n > 0:
        vis_dir = args.vis_dir or os.path.join(args.dst, 'vis_' + args.split)
        os.makedirs(vis_dir, exist_ok=True)

    t0 = time.time()
    n_done = 0
    n_dark = 0
    pending = []          # 未凑满 batch 的暗帧 / 亮帧分别攒着（同一次 pipe 调用只能一个强度）
    queues = {'bright': [], 'dark': []}

    def prep(i):
        m = metas[i]
        rgb = np.ascontiguousarray(src_rgb[i])
        K = cache_K(m, W, H)
        boxes = [np.asarray(b, dtype=np.float64) for b in m['boxes9d']]
        mask = box_mask((W, H), boxes, K, dilate=args.dilate, feather=args.feather)
        dark = bg_luminance(rgb, mask) < args.dark_mean
        return dict(i=i, orig=Image.fromarray(rgb), mask=mask, K=K, boxes=boxes, dark=dark,
                    weather=m['seq'].split('/')[3] if len(m['seq'].split('/')) > 3 else '')

    def run_batch(items, strength):
        nonlocal n_done, n_dark
        plates = [fill_background(it['orig'], it['mask'], sigma=args.sigma) for it in items]
        ins = [p.resize((gw, gh), Image.BICUBIC) for p in plates]
        prompts = [BG_PROMPT % WEATHER_WORDS.get(it['weather'], '') for it in items]
        gens = [torch.Generator('cuda').manual_seed(args.seed + it['i']) for it in items]
        outs = pipe(prompt=prompts, negative_prompt=[BG_NEGATIVE] * len(items), image=ins, strength=strength,
                    num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gens).images
        for j, it in enumerate(items):
            i = it['i']
            bg = outs[j].resize((W, H), Image.LANCZOS)
            final = paste_back(bg, it['orig'], it['mask'])
            fa, oa = np.asarray(final), np.asarray(it['orig'])
            hard = it['mask'] >= 0.999
            assert np.array_equal(fa[hard], oa[hard]), '凸包内像素被改了：帧 %d' % i
            dst_rgb[i] = fa
            translated[i] = True
            strength_used[i] = strength
            n_dark += int(it['dark'])
            if args.vis_n > 0 and n_done < args.vis_n:
                m = metas[i]
                title = '%s | %s | %d target(s) | %s s=%.2f steps=%d' % (
                    it['weather'], os.path.splitext(m['frame'])[0], len(it['boxes']), 'DARK' if it['dark'] else 'bright',
                    strength, args.steps)
                make_sheet(it['orig'], plates[j], bg, final, it['boxes'], it['K'], title,
                           os.path.join(vis_dir, '%03d_%s_%s.jpg' % (n_done, it['weather'], os.path.splitext(m['frame'])[0])))
            n_done += 1

    def checkpoint(force=False):
        dst_rgb.flush()
        np.save(os.path.join(dst_split, 'translated.npy'), translated)
        np.save(os.path.join(dst_split, 'strength.npy'), strength_used)
        el = time.time() - t0
        print('  %d/%d（暗帧 %d）  %.2f s/帧  剩余约 %.1f min' % (n_done, len(cand), n_dark, el / max(n_done, 1),
                                                             el / max(n_done, 1) * (len(cand) - n_done) / 60.0), flush=True)

    last_ck = 0
    for i in cand:
        it = prep(i)
        if args.only == 'dark' and not it['dark']:
            continue
        if args.only == 'bright' and it['dark']:
            continue
        q = queues['dark' if it['dark'] else 'bright']
        q.append(it)
        if len(q) >= args.batch:
            run_batch(q, args.dark_strength if it['dark'] else args.strength)
            q.clear()
        if n_done - last_ck >= args.flush_every:
            checkpoint()
            last_ck = n_done
    for name, q in queues.items():
        if q:
            run_batch(q, args.dark_strength if name == 'dark' else args.strength)
            q.clear()
    checkpoint()
    meta = {'args': vars(args), 'n_valid': int(len(valid)), 'n_translated_total': int(translated.sum()),
            'n_this_run': n_done, 'n_dark_this_run': n_dark, 'sec_per_frame': (time.time() - t0) / max(n_done, 1),
            'skipped': skipped, 'prompt': BG_PROMPT, 'negative': BG_NEGATIVE}
    json.dump(meta, open(os.path.join(dst_split, 'bgx_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if args.limit == 0:
        open(os.path.join(dst_split, 'READY'), 'w').write('ok\n')
    print('完成：本次 %d 帧，累计 %d/%d，%.2f s/帧' % (n_done, int(translated.sum()), len(valid), meta['sec_per_frame']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
