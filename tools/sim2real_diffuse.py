# -*- coding: utf-8 -*-
"""用本地 SDXL 做「仿真 RGB -> 照片质感」的图像翻译小样，看几何保不保得住、风格像不像真实。

只翻译 RGB；IR、LiDAR、3D 标签一律不动（教师看真实几何，学生看风格化 RGB）。
每帧在【原图 1:1】裁一个包含最近合格目标的窗口（不缩放，不丢原生像素），
用 img2img 以几档强度生成，把 GT 框用同一个 K 画到原图和生成图上 —— 框里还有没有无人机、
无人机有没有被画歪，一眼能看出来。

在 D:\\SD 的 venv 里跑（diffusers 0.40；SDXL base 权重缓存在 D:\\SD\\hf，必须设 HF_HOME，
否则 huggingface 会去 C 盘找一份不完整的缓存然后联网失败）：
    HF_HOME=D:/SD/hf HF_HUB_OFFLINE=1 D:\\SD\\venv\\Scripts\\python.exe tools/sim2real_diffuse.py --n 8 --both --out ../output/sim2real_demo
不依赖本仓库的其它模块（那个 venv 里没有 cv2/scipy）。

姿态保持（用户 2026-09-06 指出：强度 0.6 时无人机姿态本身会被改掉，旋转标签就错了）：
    --preserve  生成后把 GT 框凸包（外扩 10 px、羽化）内的【原始像素】贴回去 —— 无人机逐像素不变，
                只翻译背景 / 天气 / 光照；这是结构上保证姿态不变，而不是靠提示词。
    --both      每档强度同时出「不回贴」「回贴」两张对比。
    标题里的 box-edge IoU = 框内边缘图（原图 vs 生成图）的 IoU，是机身有没有被重画/转向的粗判据；
    不回贴时低于 ~0.3 基本就是姿态被改了。提示词只写风格，不写机型名（会被画成文字）、不写姿态词。
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

PROTO = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
WEATHER_WORDS = {
    'clear_day': 'clear daylight', 'clear_night': 'night, city lights, low light photo',
    'fog_day': 'dense fog, overcast', 'fog_night': 'foggy night, street lamps',
    'rain_day': 'rainy overcast day, wet surfaces', 'rain_night': 'rainy night, wet reflections',
    'snow_day': 'snowfall, overcast winter day', 'snow_night': 'snowy night, falling snow',
}


def euler_xyz_to_R(a):
    """与 scipy Rotation.from_euler('xyz', a) 一致：R = Rz(a3) Ry(a2) Rx(a1)（外旋）。"""
    x, y, z = a
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def box_uv(b, K):
    pts = (PROTO * b[3:6]) @ euler_xyz_to_R(b[6:9]).T + b[:3]
    uv = (K @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def draw_box(img, b, K, color=(60, 220, 60), width=3, label=None):
    d = ImageDraw.Draw(img)
    uv = box_uv(b, K)
    for i, j in EDGES:
        d.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=width)
    if label:
        d.text((max(2, uv[:, 0].min()), max(2, uv[:, 1].min() - 14)), label, fill=color)
    return img


def convex_hull(uv):
    """Andrew 单调链凸包，输入 (N,2)，返回顶点列表。"""
    pts = sorted(set(tuple(p) for p in uv))
    if len(pts) < 3:
        return pts
    def cross(o, a, b2):
        return (a[0] - o[0]) * (b2[1] - o[1]) - (a[1] - o[1]) * (b2[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def box_mask(size, boxes, K, dilate=10, feather=6):
    """一个或多个 GT 框的 8 角点投影凸包的并集 -> 软掩码 (H,W) float in [0,1]，向外扩 dilate 像素再羽化。
    必须把帧里【所有有标签的目标】都放进来：只保护一个目标，其它有标签的目标就会被模型重画（fog_night 那帧的教训）。"""
    from PIL import ImageFilter
    W, H = size
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes[None]
    m = Image.new('L', (W, H), 0)
    for b in boxes:
        hull = convex_hull(box_uv(b, K))
        if len(hull) >= 3:
            ImageDraw.Draw(m).polygon(hull, fill=255)
    if dilate > 0:
        m = m.filter(ImageFilter.MaxFilter(2 * dilate + 1))
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(m, dtype=np.float32) / 255.0


def paste_back(gen, orig, mask):
    """生成图在掩码区域回贴原图像素（无人机姿态逐像素保持）。"""
    g = np.asarray(gen, dtype=np.float32)
    o = np.asarray(orig, dtype=np.float32)
    out = g * (1 - mask[..., None]) + o * mask[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def edge_iou_in_box(a, b, mask_hard):
    """框内边缘一致性：两图灰度 FIND_EDGES 二值化后在框内的 IoU（姿态是否被改的粗判据）。"""
    from PIL import ImageFilter
    def edges(im):
        e = np.asarray(im.convert('L').filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        return e > 40
    ea, eb = edges(a), edges(b)
    m = mask_hard > 0.5
    inter = np.logical_and(np.logical_and(ea, eb), m).sum()
    union = np.logical_and(np.logical_or(ea, eb), m).sum()
    return float(inter) / max(float(union), 1.0)


def fill_background(img, mask, sigma=25):
    """把掩码区域用周围背景「抹掉」（归一化卷积：模糊(图*(1-m)) / 模糊(1-m)），得到一张没有无人机的底图。
    抹得不精细没关系——后面 img2img 会把它重画成合理的背景；关键是底图里没有无人机可供模型「重新发挥」。"""
    from PIL import ImageFilter
    im = np.asarray(img, dtype=np.float32)
    inv = (1.0 - mask)[..., None]
    num = Image.fromarray(np.clip(im * inv, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    den = Image.fromarray((inv[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)[..., None] / 255.0
    fill = num / np.maximum(den, 1e-3)
    out = im * inv + fill * (1 - inv)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def zoom_sheet(panels, b, K, path, titles=None, bands=None, win=(384, 288), up=2):
    """每个面板在目标框周围裁同一个 1:1 窗口，最近邻放大 up 倍横向拼接——姿态/边缘有没有变一眼看清。
    上方放标题：文字（titles）或从原大表上裁下来的标题条（bands，--rezoom 模式）。"""
    uv = box_uv(b, K)
    W, H = panels[0].size
    bw, bh = uv[:, 0].max() - uv[:, 0].min(), uv[:, 1].max() - uv[:, 1].min()
    ww, wh = int(max(win[0], bw * 1.3)), int(max(win[1], bh * 1.3))
    ww, wh = min(ww, W), min(wh, H)
    cx, cy = float(uv[:, 0].mean()), float(uv[:, 1].mean())
    x0 = int(np.clip(cx - ww / 2, 0, W - ww))
    y0 = int(np.clip(cy - wh / 2, 0, H - wh))
    crops = [p.crop((x0, y0, x0 + ww, y0 + wh)).resize((ww * up, wh * up), Image.NEAREST) for p in panels]
    cw, ch = crops[0].size
    band_h = 24 * up
    sheet = Image.new('RGB', (cw * len(crops) + 8 * (len(crops) - 1), ch + band_h), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(crops):
        x = i * (cw + 8)
        sheet.paste(c, (x, band_h))
        if bands is not None:
            bd = bands[i]
            bd = bd.resize((min(bd.size[0] * up, cw), bd.size[1] * up), Image.NEAREST)
            sheet.paste(bd, (x, 0))
        elif titles is not None:
            d.text((x + 6, 6), titles[i], fill=(255, 255, 255))
    sheet.save(path, quality=92)


def pick_frames(index_path, vis_path, n, min_vis=8.0):
    idx = pickle.load(open(index_path, 'rb'))
    vis = np.load(vis_path)
    best = {}
    for i in idx['valid_idx']:
        m = idx['metas'][i]
        if vis[i] < min_vis:
            continue
        cands = [(max(b[3:6]) / b[2], b) for b, q in zip(m['boxes9d'], m['qualified']) if q]
        if not cands:
            continue
        ang, b = max(cands, key=lambda t: t[0])
        w = m['seq'].split('/')[3]
        score = ang * (1.5 if m['lidar_noise'] == 0 else 1.0)
        if w not in best or score > best[w][0]:
            best[w] = (score, m, b)
    picks = [v[1:] for _, v in sorted(best.items())][:n]
    return picks


def native_crop(root, m, b, win=(1024, 576), seed=0):
    """原图 1:1 裁窗口（含目标，位置带确定性随机偏移），返回 PIL 图与裁剪后的 K。"""
    img = Image.open(os.path.join(root, m['seq'], 'images_rgb', m['frame'])).convert('RGB')
    W, H = img.size
    K = np.array(m['K_raw'], dtype=np.float64).copy()
    cu = (K @ b[:3])[:2] / b[2]
    rng = np.random.RandomState(seed)
    cw, ch = win
    lo_x, hi_x = int(max(0, cu[0] - cw + 40)), int(min(W - cw, cu[0] - 40))
    lo_y, hi_y = int(max(0, cu[1] - ch + 40)), int(min(H - ch, cu[1] - 40))
    x0 = int(rng.randint(lo_x, hi_x + 1)) if hi_x >= lo_x else int(np.clip(cu[0] - cw / 2, 0, W - cw))
    y0 = int(rng.randint(lo_y, hi_y + 1)) if hi_y >= lo_y else int(np.clip(cu[1] - ch / 2, 0, H - ch))
    crop = img.crop((x0, y0, x0 + cw, y0 + ch))
    K[0, 2] -= x0
    K[1, 2] -= y0
    return crop, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='E:/mmcache/mm20')
    ap.add_argument('--root', default='E:/data_collect')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--strengths', default='0.3,0.45,0.6')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='../output/sim2real_demo')
    ap.add_argument('--no-diffusion', action='store_true', help='只出原图裁剪+框，不跑模型')
    ap.add_argument('--preserve', action='store_true', help='生成后把 GT 框区域的原始像素贴回去（姿态逐像素保持）')
    ap.add_argument('--both', action='store_true', help='每档强度出「不回贴」与「回贴」两张，便于对比')
    ap.add_argument('--bg-only', action='store_true',
                    help='再出一张「先把无人机从底图抹掉 -> 只翻译背景 -> 把原始无人机贴回」：'
                         '模型根本看不到无人机，不会在框外画出鬼影，姿态逐像素不变')
    ap.add_argument('--rezoom', action='store_true', help='不跑模型：把 --out 里已有的大表按目标框裁放大版再看一遍')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    strengths = [float(s) for s in args.strengths.split(',')]

    picks = pick_frames(os.path.join(args.cache, 'train', 'index.pkl'),
                        os.path.join(args.cache, 'train', 'vis_score.npy'), args.n)
    print('选了 %d 帧' % len(picks))

    if args.rezoom:
        for k, (m, b) in enumerate(picks):
            weather = m['seq'].split('/')[3]
            crop, K = native_crop(args.root, m, b, seed=args.seed + k)
            sp = os.path.join(args.out, '%02d_%s.jpg' % (k, weather))
            if not os.path.exists(sp):
                continue
            sheet = Image.open(sp).convert('RGB')
            W, H = crop.size
            n = (sheet.size[0] + 8) // (W + 8)
            panels = [sheet.crop((i * (W + 8), 24, i * (W + 8) + W, 24 + H)) for i in range(n)]
            bands = [sheet.crop((i * (W + 8), 0, i * (W + 8) + min(W, 420), 24)) for i in range(n)]
            zp = sp[:-4] + '_zoom.jpg'
            zoom_sheet(panels, b, K, zp, bands=bands)
            print('写出', zp)
        return 0

    pipe = None
    if not args.no_diffusion:
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

    report = []
    for k, (m, b) in enumerate(picks):
        weather = m['seq'].split('/')[3]
        crop, K = native_crop(args.root, m, b, seed=args.seed + k)
        name = m['names'][int(np.argmax([max(bb[3:6]) / bb[2] if q else -1 for bb, q in zip(m['boxes9d'], m['qualified'])]))]
        label = '%s %.1fm' % (name, b[2])
        # 帧里所有有标签的目标（boxes9d = 40 m 内全部标签，不只是 qualified 的近目标、更不只是挑出来的那个）：
        # 掩码要保护全部，图上其它目标画黄框
        targets = [np.asarray(bb, dtype=np.float64) for bb in m['boxes9d']]
        others = [bb for bb in targets if not np.allclose(bb, np.asarray(b, dtype=np.float64))]

        def annotate(im):
            for bb in others:
                draw_box(im, bb, K, color=(235, 200, 40), width=2, label='GT %.1fm' % bb[2])
            return draw_box(im, b, K, label='GT ' + label)

        panels = [annotate(crop.copy())]
        titles = ['original (native 1:1 crop)  %d labelled target(s)' % len(targets)]
        if pipe is not None:
            import torch
            # 提示词只写风格：不写机型名（会被画成文字）、不写姿态词
            prompt = ('a real photograph of a small quadcopter drone in the sky over a city, %s, '
                      'DSLR photo, sharp focus, realistic lighting, photorealistic, film grain, high detail'
                      % WEATHER_WORDS.get(weather, ''))
            negative = 'cartoon, cgi, render, video game, painting, blurry, low quality, deformed, extra objects, text, watermark'
            mask = box_mask(crop.size, targets, K)
            bg_prompt = ('a real photograph of a city skyline and sky, %s, DSLR photo, sharp focus, '
                         'realistic lighting, photorealistic, film grain, high detail' % WEATHER_WORDS.get(weather, ''))
            bg_negative = negative + ', drone, aircraft, bird, helicopter'
            bg_base = fill_background(crop, mask) if args.bg_only else None
            for s in strengths:
                g = torch.Generator('cuda').manual_seed(args.seed + k)
                out = pipe(prompt=prompt, negative_prompt=negative, image=crop, strength=s,
                           num_inference_steps=args.steps, guidance_scale=args.guidance, generator=g).images[0]
                out = out.resize(crop.size)
                variants = []
                if args.both or not args.preserve:
                    variants.append(('img2img %.2f  box-edge IoU %.2f' % (s, edge_iou_in_box(crop, out, mask)), out))
                if args.both or args.preserve:
                    kept = paste_back(out, crop, mask)
                    variants.append(('img2img %.2f + paste-back target  (IoU %.2f)' % (s, edge_iou_in_box(crop, kept, mask)), kept))
                if args.bg_only:
                    g = torch.Generator('cuda').manual_seed(args.seed + k)
                    bg = pipe(prompt=bg_prompt, negative_prompt=bg_negative, image=bg_base, strength=s,
                              num_inference_steps=args.steps, guidance_scale=args.guidance, generator=g).images[0]
                    bg = paste_back(bg.resize(crop.size), crop, mask)
                    variants.append(('bg-only %.2f (target erased, translated, original pasted)  (IoU %.2f)'
                                     % (s, edge_iou_in_box(crop, bg, mask)), bg))
                for t, im in variants:
                    panels.append(annotate(im))
                    titles.append(t)
                    print('  [%02d %s] %s' % (k, weather, t))
        # 拼图：一行
        W, H = panels[0].size
        sheet = Image.new('RGB', (W * len(panels) + 8 * (len(panels) - 1), H + 24), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for i, (p, t) in enumerate(zip(panels, titles)):
            x = i * (W + 8)
            sheet.paste(p, (x, 24))
            d.text((x + 6, 6), t, fill=(255, 255, 255))
        d.text((W - 260, 6), '%s | %s' % (weather, os.path.splitext(m['frame'])[0]), fill=(200, 200, 200))
        op = os.path.join(args.out, '%02d_%s.jpg' % (k, weather))
        sheet.save(op, quality=90)
        zoom_sheet(panels, b, K, op[:-4] + '_zoom.jpg', titles=titles)
        report.append({'k': k, 'weather': weather, 'seq': m['seq'], 'frame': m['frame'],
                       'range_m': float(b[2]), 'titles': titles})
        print('写出', op)
    import json
    json.dump(report, open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
