# -*- coding: utf-8 -*-
"""PREREG_v2.md（sha256 19e95a39）特征（由 uda_feats.py 改：臂 UH / V2A / V2B / V2C / V2Ap / V2Bp / V2Cp，第二段种子 j = 2 + (k mod 3)）。原说明：PREREG_uda.md 特征：P0_s{k} 每个检测（取窗同 dc / softkd），第二段种子 j = 3（k 偶）/ 4（k 奇）。
臂（每个都出 δ̂、log b）：
    UH / UHFT / UHAB / UHDA / UT：吃彩色裁窗
    SBT / SBH：剪影瓶颈，SILIN_s{j} 吃 UT / UH 预测的剪影概率（×255 当灰度图）
    GINV / GTIR：TIRINV_s{j} / TIR_s{j} 吃彩色裁窗转的灰度图
三个集合：真实测试、真实验证、仿真全量测试。输出 uda/feats_{split}_s{k}.pkl（结构同 twohead/feats_*）。"""
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
from scipy.spatial.transform import Rotation as R  # noqa: E402
from uavdet3d.utils import pose_eval as pe  # noqa: E402
import cropw_extract as ce  # noqa: E402
import silw_train as sw  # noqa: E402
import cropdist_eval as cde  # noqa: E402
import dcalign_eval as dae  # noqa: E402
import softkd_train as skt  # noqa: E402
import secondstage_v2 as v2  # noqa: E402

ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
OUT = os.environ.get('V2_OUT', G + '/v2')
SRC = {'test': G + '/irloc_recs/P0_s%d.pkl', 'val': G + '/irloc_val_recs/P0_s%d.pkl', 'sim': G + '/sim_full_recs/P0_s%d.pkl'}
if os.environ.get('V2_SRC') == 'st2':   # PREREG_st2combo：第一段换成 ST2 检测器，只评真实
    SRC = {'test': G + '/st2_recs/test_s%d.pkl', 'val': G + '/st2_recs/val_s%d.pkl'}
if os.environ.get('V2_SRC') == 'st2pp':   # PREREG_st2pp：第二轮去偏自训练的检测器
    SRC = {'test': G + '/st2pp_recs/test_s%d.pkl', 'val': G + '/st2pp_recs/val_s%d.pkl'}
if os.environ.get('V2_SRC') == 'st2plus':   # PREREG_st2plus：第一段换成重新自训练的检测器
    SRC = {'test': G + '/st2plus_recs/test_s%d.pkl', 'val': G + '/st2plus_recs/val_s%d.pkl'}
if os.environ.get('V2_SRC') == 'st2train':   # 用户 10-11 泛化检查：真实训练集（标注只打分）
    SRC = {'train': G + '/st2_train_recs/train_s%d.pkl'}
class _TrainImages:
    def __init__(self):
        self.i = __import__('color_feats').Images('train')

    def get(self, seq, frame):
        img, K = self.i.get(seq, frame)
        return img, K, None


RGB_ARMS = tuple(os.environ.get('V2_ARMS', 'UH,V2A,V2B,V2C,V2Ap,V2Bp,V2Cp').split(','))


def load(name, j):
    n = v2.make(name).cuda()
    n.load_state_dict(torch.load('%s/softkd/%s_s%d.pt' % (G, name, j), map_location='cuda'))
    return n.eval()


@torch.no_grad()
def compute(recs, imgs, j):
    nets = {a: load(a, j) for a in RGB_ARMS}
    items, views, scal, geo = [], [], [], []
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
            if not (6.0 <= w <= 170.0):
                continue
            views.append(sw.view_img(ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), ce.KW * w), S0, 0.0, 0.0, 1.0, w, False))
            scal.append([np.log(w / K[0, 0]), np.log(g[2])] + list(np.log(np.maximum(g[3:6], 1e-3))))
            items.append((ri, pi))
            geo.append([w, v1 - v0, K[0, 0], g[2]] + list(g[3:9]))
    scal = np.array(scal, np.float32)
    res = {a: ([], []) for a in RGB_ARMS}

    def put(a, dh, lb):
        res[a][0].append(dh.float().cpu().numpy())
        res[a][1].append(lb.float().cpu().numpy())
    for b in range(0, len(views), 256):
        vb = np.stack(views[b:b + 256])
        x = skt.dt.prep(vb, None, False).cuda()
        s = torch.from_numpy(scal[b:b + 256]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for a, n in nets.items():
                dh, lb, lg = n(x, s)
                put(a, dh, lb)
    out = dict(items=np.array(items, np.int64).reshape(-1, 2), geo=np.array(geo, np.float32).reshape(-1, 10))
    for a, (d, lb) in res.items():
        out[a] = dict(d=np.concatenate(d) if d else np.zeros(0), lb=np.concatenate(lb) if lb else np.zeros(0))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for split in [sp for sp in ('test', 'val', 'sim', 'train') if sp in SRC]:
        imgs = _TrainImages() if split == 'train' else (dae.ImagesVal() if split == 'val' else cde.Images('real' if split == 'test' else 'sim'))
        for k in range(5):
            fn = '%s/feats_%s_s%d.pkl' % (OUT, split, k)
            if os.path.exists(fn):
                continue
            t0 = time.time()
            recs = pickle.load(open(SRC[split] % k, 'rb'))['records']
            out = compute(recs, imgs, {'identity': k, 'plus5': k + 5}.get(os.environ.get('V2_MAP'), 2 + k % 3))
            pickle.dump(out, open(fn + '.tmp', 'wb'))
            os.replace(fn + '.tmp', fn)
            print(split, k, '检测 %d 个，%.0fs' % (len(out['items']), time.time() - t0), flush=True)
            torch.cuda.empty_cache()
    print('全部完成', flush=True)


if __name__ == '__main__':
    main()
