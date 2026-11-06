# -*- coding: utf-8 -*-
"""裁窗版蒸馏模型（PREREG_cropdist.md）：检测网络 P0_s{k} 不动，直接改它已有的逐帧结果（仿真全量 / 真实标准口径）。
对每个检测：投影预测 3D 框 → 画面外接框中心 (cx,cy)、画面宽 w；按剪影测宽实验完全相同的流程取窗口
（大窗口 3w、256 像素 → 子窗口边长 s0·w，s0 = √(1.2×2.5) ≈ 1.73，不抖动、不翻转），送进第二段网络（种子 j = k mod 3）：
    CROP_SIL（蒸馏）：剪影网络 M_s{j} 量出剪影宽 → 比例网络 RR_s{j} 换回框宽
    CROP_RB（对照）：RB_s{j} 直接回归框宽（不用剪影）
位置沿视线缩放 p ← p × w / 新框宽；新旧比值超出 [1/1.4, 1.4]、量不出、或 w 不在 [6, 170] 像素（第二段训练范围）时保留原值。
图像：仿真用 full_v2 test 原分辨率 1280×720（与第二段训练同源），真实用 mav6d_cn_obj test 缓存（512×288）。
输出 cropdist/{arm}_{sim|real}_s{k}.pkl（逐帧记录，与 sim_full_recs / irloc_recs 同格式）与 cropdist/summary.json。"""
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
sys.path.insert(0, ROOT + '/tools')
from scipy.spatial.transform import Rotation as R  # noqa: E402
from uavdet3d.utils import pose_eval as pe  # noqa: E402
import cropw_extract as ce  # noqa: E402
import silw_train as sw  # noqa: E402

ce.P, ce.KW = 256, 3.0
OUT = G + '/cropdist'
S0 = float(np.sqrt(1.2 * 2.5))
GUARD = 1.4
WMIN, WMAX = 6.0, 170.0


class Images:
    def __init__(self, which):
        self.which = which
        if which == 'sim':
            b = '@@MMCACHE@@/full_v2/test'
            self.mm = np.memmap(b + '/rgb_jpg.bin', dtype=np.uint8, mode='r')
            self.idx = np.load(b + '/rgb_jpg_index.npy')
        else:
            b = '@@MMCACHE@@/mav6d_cn_obj/test'
            self.arr = np.load(b + '/rgb.npy', mmap_mode='r')
        d = pickle.load(open(b + '/index.pkl', 'rb'))
        self.metas = d['metas']
        self.key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(d['metas'])}

    def get(self, seq, frame):
        i = self.key[(seq, frame)]
        m = self.metas[i]
        if self.which == 'sim':
            o, ln = self.idx[i]
            img = cv2.imdecode(np.asarray(self.mm[int(o):int(o) + int(ln)]), cv2.IMREAD_COLOR)
        else:
            img = np.ascontiguousarray(self.arr[i])
        return img, np.asarray(m['K_in'], np.float64), m


def load_nets(j):
    nets = {}
    for name, cls in (('M', sw.MaskNet), ('RR', sw.Scalar), ('RB', sw.Scalar)):
        n = cls().cuda()
        n.load_state_dict(torch.load('%s/silw/%s_s%d.pt' % (G, name, j), map_location='cuda'))
        nets[name] = n.eval()
    return nets


@torch.no_grad()
def run_nets(nets, views):
    x = sw.to_tensor(views, False, False, None).cuda()
    out = {}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lg = nets['M'](x).float().cpu().numpy()
        out['rr'] = nets['RR'](x).float().cpu().numpy()
        out['rb'] = nets['RB'](x).float().cpu().numpy()
    pm = 1 / (1 + np.exp(-lg))
    out['sil'] = np.array([sw.extent(q) for q in pm])       # 128 像素子窗口里的剪影宽（像素）或 nan
    return out


def process(recs, imgs, nets):
    """→ (CROP_SIL 记录, CROP_RB 记录, 统计)"""
    items = []                                              # (记录号, 检测号, 原画面宽 w, 子窗口边长 L)
    views = []
    for ri, r in enumerate(recs):
        if len(r['pred']) == 0:
            continue
        img, K, _ = imgs.get(r['seq_id'], r['frame_id'])
        for pi, g in enumerate(r['pred']):
            if g[2] <= 1e-3:
                continue
            c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            q = (K @ c.T).T
            if (q[:, 2] <= 1e-3).any():
                continue
            uv = q[:, :2] / q[:, 2:3]
            u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
            w = u1 - u0
            if not (WMIN <= w <= WMAX):
                continue
            patch = ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), ce.KW * w)
            views.append(sw.view_img(patch, S0, 0.0, 0.0, 1.0, w, False))
            items.append((ri, pi, w, S0 * w))
    outs = {'sil': [], 'rr': [], 'rb': []}
    for b in range(0, len(views), 256):
        o = run_nets(nets, views[b:b + 256])
        for k in outs:
            outs[k].append(o[k])
    outs = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in outs.items()}
    sil = [dict(r, pred=r['pred'].copy()) for r in recs]
    rb = [dict(r, pred=r['pred'].copy()) for r in recs]
    st = dict(n_det=sum(len(r['pred']) for r in recs), n_try=len(items), sil_changed=0, rb_changed=0)
    for n, (ri, pi, w, L) in enumerate(items):
        ws = outs['sil'][n]
        if np.isfinite(ws):
            wb = (ws / 128.0 * L) / np.exp(outs['rr'][n])
            f = w / max(wb, 1e-3)
            if 1.0 / GUARD <= f <= GUARD:
                sil[ri]['pred'][pi, :3] *= f
                st['sil_changed'] += 1
        wb2 = np.exp(outs['rb'][n]) * L
        f2 = w / max(wb2, 1e-3)
        if 1.0 / GUARD <= f2 <= GUARD:
            rb[ri]['pred'][pi, :3] *= f2
            st['rb_changed'] += 1
    return sil, rb, st


def summ(recs):
    s = pe.summarize(pe.match_records(recs, seq='xyz', match='greedy', dist_thresh=1.0), n_frames=len(recs))
    return dict(det=100 * s['det_rate'], pos=s.get('pos_median'))


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = json.load(open(OUT + '/summary.json', encoding='utf-8')) if os.path.exists(OUT + '/summary.json') else {}
    srcs = {'sim': G + '/sim_full_recs/P0_s%d.pkl', 'real': G + '/irloc_recs/P0_s%d.pkl'}
    for which in ('real', 'sim'):
        imgs = Images(which)
        for k in range(5):
            key = '%s_s%d' % (which, k)
            if key in summary:
                continue
            t0 = time.time()
            z = pickle.load(open(srcs[which] % k, 'rb'))
            nets = load_nets(k % 3)
            sil, rb, st = process(z['records'], imgs, nets)
            for arm, recs in (('CROP_SIL', sil), ('CROP_RB', rb)):
                with open('%s/%s_%s_s%d.pkl' % (OUT, arm, which, k), 'wb') as f:
                    pickle.dump(dict(z, records=recs), f)
            summary[key] = dict(stats=st, P0=summ(z['records']), CROP_SIL=summ(sil), CROP_RB=summ(rb))
            print(key, json.dumps(summary[key], ensure_ascii=False), '%.0fs' % (time.time() - t0), flush=True)
            json.dump(summary, open(OUT + '/summary.json.tmp', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            os.replace(OUT + '/summary.json.tmp', OUT + '/summary.json')
            del nets
            torch.cuda.empty_cache()
    print('全部完成', flush=True)


if __name__ == '__main__':
    main()
