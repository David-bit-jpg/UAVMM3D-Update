# -*- coding: utf-8 -*-
"""用本地 SDXL 做「仿真 RGB -> 照片质感」的图像翻译小样，看几何保不保得住、风格像不像真实。

只翻译 RGB；IR、LiDAR、3D 标签一律不动（教师看真实几何，学生看风格化 RGB）。
每帧在【原图 1:1】裁一个包含最近合格目标的窗口（不缩放，不丢原生像素），
用 img2img 以几档强度生成，把 GT 框用同一个 K 画到原图和生成图上 —— 框里还有没有无人机、
无人机有没有被画歪，一眼能看出来。

在 D:\\SD 的 venv 里跑（diffusers 0.40，SDXL base 已缓存）：
    D:\\SD\\venv\\Scripts\\python.exe tools/sim2real_diffuse.py --n 8 --out ../output/sim2real_demo
不依赖本仓库的其它模块（那个 venv 里没有 cv2/scipy）。
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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    strengths = [float(s) for s in args.strengths.split(',')]

    picks = pick_frames(os.path.join(args.cache, 'train', 'index.pkl'),
                        os.path.join(args.cache, 'train', 'vis_score.npy'), args.n)
    print('选了 %d 帧' % len(picks))

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

    for k, (m, b) in enumerate(picks):
        weather = m['seq'].split('/')[3]
        crop, K = native_crop(args.root, m, b, seed=args.seed + k)
        name = m['names'][int(np.argmax([max(bb[3:6]) / bb[2] if q else -1 for bb, q in zip(m['boxes9d'], m['qualified'])]))]
        label = '%s %.1fm' % (name, b[2])
        panels = [draw_box(crop.copy(), b, K, label='GT ' + label)]
        titles = ['original (native 1:1 crop)']
        if pipe is not None:
            import torch
            prompt = ('a real photograph of a %s quadcopter drone flying over a city, %s, '
                      'DSLR photo, sharp focus, realistic lighting, photorealistic, high detail' % (name.replace('-', ' '), WEATHER_WORDS.get(weather, '')))
            negative = 'cartoon, cgi, render, video game, painting, blurry, low quality, deformed, extra objects'
            for s in strengths:
                g = torch.Generator('cuda').manual_seed(args.seed + k)
                out = pipe(prompt=prompt, negative_prompt=negative, image=crop, strength=s,
                           num_inference_steps=args.steps, guidance_scale=args.guidance, generator=g).images[0]
                out = out.resize(crop.size)
                panels.append(draw_box(out, b, K, label='GT ' + label))
                titles.append('img2img strength %.2f' % s)
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
        print('写出', op)
    return 0


if __name__ == '__main__':
    sys.exit(main())
