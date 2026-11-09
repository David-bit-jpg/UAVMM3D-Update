# -*- coding: utf-8 -*-
"""按颜色判别（用户 10-11）：ST2 单模型 k 的每个检测，第二段 V2Pp_s{j}（j = k 与 j = k + 5）给出修正量 δ̂ 和剪影；
用网络自己画的剪影（不用标签）量机身颜色：机身平均亮度 lum（0–255）、机身 − 周围亮度 con、剪影面积 area。
集合：真实测试 / 验证 / 训练。输出 colorfeat/{split}_k{k}_j{j}.pkl。用法：python color_feats.py <split,split> <j 偏移,偏移>"""
import os, pickle, sys, time
import cv2
import numpy as np
import torch
ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT); sys.path.insert(0, G)
from scipy.spatial.transform import Rotation as R
from uavdet3d.utils import pose_eval as pe
import cropw_extract as ce
import silw_train as sw
import softkd_train as skt
import secondstage_v2 as v2
ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
SRC = {'test': G + '/st2_recs/test_s%d.pkl', 'val': G + '/st2_recs/val_s%d.pkl', 'train': G + '/st2_train_recs/train_s%d.pkl'}


class Images:
    def __init__(self, split):
        b = '@@MMCACHE@@/mav6d_cn_obj/' + split
        self.arr = np.load(b + '/rgb.npy', mmap_mode='r')
        d = pickle.load(open(b + '/index.pkl', 'rb'))
        self.metas = d['metas']
        self.key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(self.metas)}

    def get(self, seq, frame):
        i = self.key[(seq, frame)]
        return np.ascontiguousarray(self.arr[i]), np.asarray(self.metas[i]['K_in'], np.float64)


def color_stats(v, p):
    g = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = (p > 0.5).astype(np.uint8)
    if b.sum() < 10:
        return np.nan, np.nan, float(b.mean())
    d1 = cv2.dilate(b, np.ones((9, 9), np.uint8)); d2 = cv2.dilate(b, np.ones((21, 21), np.uint8))
    ring = (d2 > 0) & (d1 == 0)
    lum = float(g[b > 0].mean())
    return lum, lum - float(np.median(g[ring])), float(b.mean())


@torch.no_grad()
def compute(recs, imgs, j):
    net = v2.make('V2Pp').cuda()
    net.load_state_dict(torch.load('%s/softkd/V2Pp_s%d.pt' % (G, j), map_location='cuda'))
    net.eval()
    items, views, scal = [], [], []
    for ri, r in enumerate(recs):
        if len(r['pred']) == 0:
            continue
        img, K = imgs.get(r['seq_id'], r['frame_id'])
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
    scal = np.array(scal, np.float32)
    D, ST = [], []
    for b in range(0, len(views), 256):
        vb = np.stack(views[b:b + 256])
        x = skt.dt.prep(vb, None, False).cuda()
        s = torch.from_numpy(scal[b:b + 256]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dh, _, lg = net(x, s)
        D.append(dh.float().cpu().numpy())
        ps = torch.sigmoid(lg.float()).cpu().numpy()
        ST += [color_stats(v, p) for v, p in zip(vb, ps)]
    ST = np.array(ST, np.float32).reshape(-1, 3)
    return dict(items=np.array(items, np.int64).reshape(-1, 2), d=np.concatenate(D) if D else np.zeros(0), lum=ST[:, 0], con=ST[:, 1], area=ST[:, 2])


def main():
    splits = sys.argv[1].split(',')
    offs = [int(x) for x in sys.argv[2].split(',')]
    for sp in splits:
        imgs = Images(sp)
        for k in range(5):
            for off in offs:
                fn = '%s/colorfeat/%s_k%d_j%d.pkl' % (G, sp, k, k + off)
                if os.path.exists(fn) or not os.path.exists(SRC[sp] % k):
                    continue
                t0 = time.time()
                recs = pickle.load(open(SRC[sp] % k, 'rb'))['records']
                out = compute(recs, imgs, k + off)
                pickle.dump(out, open(fn + '.tmp', 'wb')); os.replace(fn + '.tmp', fn)
                print(sp, k, k + off, len(out['items']), '%.0fs' % (time.time() - t0), flush=True)
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
