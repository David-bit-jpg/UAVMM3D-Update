# -*- coding: utf-8 -*-
"""【11-10 无标签链副本 final_unc_nl.py：检测记录换成新检测器 <arm>（train = nolabel/<arm>/recs/<arm>_train.pkl，val/test = <arm>_ens），
图像与内参只从去真值帧表取（不打开 index.pkl），输出 nolabel/<arm>/unc/。其余逐字相同。用法：python final_unc_nl.py <arm> <标签> <网络名> <种子,...> <split,...>】
wf21/strict：最终管线第二段（5 网）逐检测输出 + 单帧颜色归类亮度（GPU，只推理，不读标签）。
对每条检测记录（检测器 = 全严格 wsfnb，单帧推理）：
  δ̂（5 网均值）、log b（均值）、5 网 δ̂ 标准差 —— 与 finalads/unc_dump.deltas_full 逐行同法（同裁窗、同标量）；
  top-1 检测的机身亮度 lum：用这 5 个第二段网络自己画的剪影（5 网 sigmoid 平均 > 0.5）圈出机身，取灰度均值（color_feats.color_stats 同法）。
  —— 替代旧 color_type_{val,test}.pkl（它用 ST2 单模型**带热图时序背景**的检测裁窗，推理跨帧）。
split：train（recs/wsfnb_train.pkl，间隔 1）/ val / test（wsfnb_ens/ens_<split>.pkl，部署同源）。
输出 unc/unc_<split>_<标签>.pkl：dict(keys, xyz, unc=[dict(d, lb, sd)], lum={(seq, frame): lum})。
用法：python final_unc.py <标签> <网络名> <种子,种子,...> <split,split>   例：python final_unc.py st XFRst 5,6,7,8,9 val,test,train"""
import os, sys, pickle, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G); sys.path.insert(0, G + '/finalads')
import xpaste_final as XF   # noqa: E402  （导入时设 ce.P/KW，与开发一致）
from color_feats import color_stats   # noqa: E402
NL = G + '/gproto/nolabel'


@torch.no_grad()
def run(recs, imgs, members):
    nets = []
    for a, j in members:
        n = XF.v2.make(a).cuda()
        n.load_state_dict(torch.load('%s/softkd/%s_s%d.pt' % (G, a, j), map_location='cuda'))
        nets.append(n.eval())
    items, views, scal = [], [], []
    for ri, r in enumerate(recs):
        if len(r['pred']) == 0:
            continue
        img, K, _ = imgs.get(r['seq_id'], r['frame_id'])
        for pi, g in enumerate(r['pred']):
            if g[2] <= 1e-3:
                continue
            c = (XF.pe.PROTO8 * g[3:6]) @ XF.R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            q = (K @ c.T).T
            if (q[:, 2] <= 1e-3).any():
                continue
            uv = q[:, :2] / q[:, 2:3]
            u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
            w = u1 - u0
            if not (6.0 <= w <= 170.0):
                continue
            views.append(XF.sw.view_img(XF.ce.window(img, 0.5 * (u0 + u1), 0.5 * (v0 + v1), XF.ce.KW * w), XF.S0, 0.0, 0.0, 1.0, w, False))
            scal.append([np.log(w / K[0, 0]), np.log(g[2])] + list(np.log(np.maximum(g[3:6], 1e-3))))
            items.append((ri, pi))
    scal = np.array(scal, np.float32)
    D, B, L = [], [], []
    for b in range(0, len(views), 256):
        vb = np.stack(views[b:b + 256])
        x = XF.v2.skt.dt.prep(vb, None, False).cuda()
        s = torch.from_numpy(scal[b:b + 256]).cuda()
        ds, lbs, ps = [], [], 0
        for n in nets:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dh, lb, lg = n(x, s)
            ds.append(dh.float().cpu().numpy()); lbs.append(lb.float().cpu().numpy())
            ps = ps + torch.sigmoid(lg.float()).cpu().numpy() / len(nets)
        D.append(np.stack(ds, 1)); B.append(np.stack(lbs, 1))
        L += [color_stats(v, p)[0] for v, p in zip(vb, ps)]
    D = np.concatenate(D) if D else np.zeros((0, len(nets)))
    B = np.concatenate(B) if B else np.zeros((0, len(nets)))
    out = [dict(d=np.full(len(r['pred']), np.nan), lb=np.full(len(r['pred']), np.nan), sd=np.full(len(r['pred']), np.nan)) for r in recs]
    lum = {}
    for n, (ri, pi) in enumerate(items):
        out[ri]['d'][pi] = D[n].mean(); out[ri]['lb'][pi] = B[n].mean(); out[ri]['sd'][pi] = D[n].std()
        r = recs[ri]
        if pi == int(np.argmax(r['conf'])) and np.isfinite(L[n]):
            lum[(str(r['seq_id']), str(r['frame_id']))] = float(L[n])
    return out, lum


class NoGTImgs:
    """rgb.npy 行 + 去真值帧表的 K_in（与 color_feats.Images / dae.ImagesVal / cde.Images 取到的图像与内参相同）"""
    def __init__(self, sp):
        T = pickle.load(open('%s/%s_frames_nogt.pkl' % (NL, sp), 'rb'))
        self.key = {k: i for i, k in enumerate(T['keys'])}
        self.K = T['K_in']
        self.arr = np.load('@@MMCACHE@@/mav6d_cn_obj/%s/rgb.npy' % sp, mmap_mode='r')

    def get(self, s, f):
        i = self.key[(str(s), str(f))]
        return np.ascontiguousarray(self.arr[i]), np.asarray(self.K[i], np.float64), None


def main():
    arm, tag, net = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [int(x) for x in sys.argv[4].split(',')]
    od = '%s/%s/unc' % (NL, arm)
    os.makedirs(od, exist_ok=True)
    for sp in sys.argv[5].split(','):
        fn = '%s/unc_%s_%s.pkl' % (od, sp, tag)
        if os.path.exists(fn):
            continue
        t0 = time.time()
        if sp == 'train':
            recs = pickle.load(open('%s/%s/recs/%s_train.pkl' % (NL, arm, arm), 'rb'))['records']
        else:
            recs = pickle.load(open('%s/%s_ens/ens_%s.pkl' % (G, arm, sp), 'rb'))['records']
        u, lum = run(recs, NoGTImgs(sp), [(net, j) for j in seeds])
        pickle.dump(dict(keys=[(r['seq_id'], r['frame_id']) for r in recs], xyz=[np.asarray(r['pred'])[:, :3].copy() if len(r['pred']) else np.zeros((0, 3)) for r in recs],
                         unc=u, lum=lum, members=[(net, j) for j in seeds]), open(fn + '.tmp', 'wb'))
        os.replace(fn + '.tmp', fn)
        print('%s %s %s：%d 帧，有亮度 %d，%.0f s' % (arm, tag, sp, len(recs), len(lum), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
