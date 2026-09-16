# -*- coding: utf-8 -*-
"""camnorm 结果可视化：纯真实 / 纯仿真 / 迁移 同帧对比 + 随预算的曲线 + 原版 LAA3D_ADS 打印。

    python vis_camnorm_compare.py --arms C_p005_s0 Z_S1 T_p005_s0 --n 8 --out ../output/camnorm/vis/compare_p005
    python vis_camnorm_compare.py --curves-only

同帧对比：从 MAV6D 测试集按序列均匀抽 N 帧（固定种子，各臂同一批帧），每行一帧、每列一个臂，
以真值为中心放大裁块：绿 = 真值 3D 框、红 = 预测 3D 框（红色箭头 = 预测机头 x 轴，绿色箭头 = 真值机头），标位置/角度误差。
推理用 workers=0，不额外开 dataloader 进程（这台机器同时 >=3 个带 worker 的进程会锁死）。
"""
import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'output', 'models', 'uavdet_3d', 'camnorm')
JS = os.path.join(ROOT, 'output', 'camnorm', 'json')
MAV_CFG = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
BUDGETS = [1, 5, 10, 25, 50, 100]
PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def arm_label(tag):
    if tag.startswith('Z_'):
        return '纯仿真(零样本)'
    m = re.match(r'([CT])_p(\d+)_s(\d)', tag)
    if m:
        return '%s %d%% 种子%s' % ('纯真实' if m.group(1) == 'C' else '迁移', int(m.group(2)), m.group(3))
    return tag


def arm_ckpt(tag):
    if tag.startswith('Z_'):
        return os.path.join(MODELS, 'sim_indoor8_mz', tag[2:], 'ckpt', 'best.pth')
    return os.path.join(MODELS, 'mav6d', tag, 'ckpt', 'best.pth')


def draw_box(img, K, box, color, arrow_color, scale):
    Rm = R.from_euler('xyz', box[6:9]).as_matrix()
    c = (PROTO8 * box[3:6]) @ Rm.T + box[:3]
    if c[:, 2].min() <= 0.05:
        return
    Ks = K.copy()
    Ks[:2] *= scale
    Ks[0, 2] = (K[0, 2] + 0.5) * scale - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * scale - 0.5

    def p(x):
        u = Ks @ x
        return int(round(u[0] / u[2])), int(round(u[1] / u[2]))
    for a, b in EDGES:
        cv2.line(img, p(c[a]), p(c[b]), color, 2, cv2.LINE_AA)
    cv2.arrowedLine(img, p(box[:3]), p(box[:3] + Rm[:, 0] * 0.3), arrow_color, 2, cv2.LINE_AA, tipLength=0.25)


def put_cn(img, text, org, size=18, color=(255, 255, 255)):
    """cv2 不能写中文：用 PIL 画。"""
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for f in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simsun.ttc'):
        if os.path.exists(f):
            font = ImageFont.truetype(f, size)
            break
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((org[0] + 1, org[1] + 1), text, font=font, fill=(0, 0, 0))
    d.text(org, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def run_arm(tag, frame_idx):
    cfg = EasyDict()
    cfg_from_yaml_file(MAV_CFG, cfg)
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    ds.valid_idx = np.asarray(frame_idx)
    model = build_network(cfg.MODEL, ds)
    n_loaded, n_total = model.load_params_from_file(arm_ckpt(tag), to_cpu=False)
    assert n_loaded == n_total, tag
    model.cuda().eval()
    recs = pose_eval.run_inference(model, loader)
    del model
    torch.cuda.empty_cache()
    return recs


def pick_frames(n, seed):
    import pickle
    idx = pickle.load(open('E:/mmcache/mav6d_cn/test/index.pkl', 'rb'))
    by_seq = {}
    for i in idx['valid_idx']:
        by_seq.setdefault(idx['metas'][int(i)]['seq'], []).append(int(i))
    rng = np.random.RandomState(seed)
    seqs = sorted(by_seq)
    out = []
    k = 0
    while len(out) < n:
        s = seqs[k % len(seqs)]
        out.append(int(rng.choice(by_seq[s])))
        k += 1
    return sorted(out), idx


def compare_sheet(tags, n, seed, out):
    frames, idx = pick_frames(n, seed)
    rgb = np.load('E:/mmcache/mav6d_cn/test/rgb.npy', mmap_mode='r')
    results = {t: run_arm(t, frames) for t in tags}
    scale, half = 3, 110
    rows = []
    for fi, i in enumerate(frames):
        tiles = []
        for t in tags:
            r = results[t][fi]
            img = cv2.resize(np.ascontiguousarray(rgb[i]), (512 * scale, 288 * scale), interpolation=cv2.INTER_CUBIC)
            g = r['gt'][0]
            K = r['K']
            draw_box(img, K, g, (0, 220, 0), (0, 255, 0), scale)
            txt = '无检测'
            if len(r['pred']):
                p = r['pred'][int(np.argmax(r['conf']))]
                draw_box(img, K, p, (0, 0, 255), (0, 0, 255), scale)
                pe = float(np.linalg.norm(p[:3] - g[:3]))
                ae = float(np.degrees((R.from_euler('xyz', p[6:9]).inv() * R.from_euler('xyz', g[6:9])).magnitude()))
                txt = '位置 %.2f m  角度 %.0f°' % (pe, ae)
            u = K @ g[:3]
            cu, cv_ = int(u[0] / u[2] * scale), int(u[1] / u[2] * scale)
            # 裁块大小随真值框投影大小自适应（近处的机子框很大）
            gc = (PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            gu = (K @ gc.T).T
            gu = gu[:, :2] / gu[:, 2:3] * scale
            ext = max(np.abs(gu[:, 0] - cu).max(), np.abs(gu[:, 1] - cv_).max())
            h2 = int(max(half, 1.6 * ext))
            pad = cv2.copyMakeBorder(img, h2, h2, h2, h2, cv2.BORDER_CONSTANT)
            tile = cv2.resize(pad[cv_:cv_ + 2 * h2, cu:cu + 2 * h2], (300, 300), interpolation=cv2.INTER_AREA)
            tile = put_cn(tile, txt, (6, 272), 17)
            if fi == 0:
                tile = put_cn(tile, arm_label(t), (6, 4), 19, (0, 255, 255))
            tiles.append(tile)
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out + '.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('写出', out + '.jpg', sheet.shape)


def load_json(tag):
    p = os.path.join(JS, '%s_test.json' % tag)
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding='utf-8'))
    for prof, txt in (d.get('ads_reports') or {}).items():
        m = re.search(r'LAA3D_ADS_drone \(%\) : ([0-9.]+)', txt)
        if m:
            d['ads_' + prof] = float(m.group(1))
    return d


def curves(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False
    metrics = [('pos_median', '位置误差中位 (m)', 1, True), ('ang_median', '角度误差中位 (°)', 1, True),
               ('ACC_0.1m10deg', '位置<10cm 且 角度<10° (%)', 100, False), ('ads_indoor', 'LAA3D_ADS 室内口径', 1, False)]
    z = load_json('Z_S1')
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for ax, (key, name, sc, logy) in zip(axes, metrics):
        for arm, col, lab in (('C', '#1f77b4', '纯真实（从零）'), ('T', '#d62728', '迁移（仿真预训练）')):
            xs, ms, ss = [], [], []
            for b in BUDGETS:
                v = [load_json('%s_p%03d_s%d' % (arm, b, s)) for s in (0, 1, 2)]
                v = [d[key] * sc for d in v if d and key in d]
                if v:
                    xs.append(b)
                    ms.append(np.mean(v))
                    ss.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
            if xs:
                ax.errorbar(xs, ms, yerr=ss, marker='o', color=col, label=lab, capsize=3)
                for x, m in zip(xs, ms):
                    ax.annotate(('%.3f' if sc == 1 and key == 'pos_median' else '%.1f') % m, (x, m),
                                textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8, color=col)
        if z and key in z:
            ax.axhline(z[key] * sc, color='#2ca02c', ls='--', label='纯仿真（零样本）%.3g' % (z[key] * sc))
        ax.set_xscale('log')
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels(['%d%%' % b for b in BUDGETS])
        if logy:
            ax.set_yscale('log')
        ax.set_title(name)
        ax.set_xlabel('MAV6D 真实训练数据量')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('MAV6D 测试集 4800 帧（3 种子均值±标准差，按验证集选轮）', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print('写出', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='*', default=[])
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'output', 'camnorm', 'vis', 'compare'))
    ap.add_argument('--curves-out', default=os.path.join(ROOT, 'output', 'camnorm', 'vis', 'curves.png'))
    ap.add_argument('--curves-only', action='store_true')
    args = ap.parse_args()
    curves(args.curves_out)
    if not args.curves_only and args.arms:
        compare_sheet([t for t in args.arms if os.path.exists(arm_ckpt(t))], args.n, args.seed, args.out)


if __name__ == '__main__':
    main()
