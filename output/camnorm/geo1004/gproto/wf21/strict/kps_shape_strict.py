# -*- coding: utf-8 -*-
"""wf21/strict：逐帧关键点高度比 r_kp（与 kps_shape_chain.shape 同法：KPS 种子 0、1 在部署裁窗上读结构关键点 → 高度比 = z 向跨度 / 平均(x, y 跨度)，两种子平均），只换三处：
  ① KPS 网络的 AdaBN 统计换成完全单帧版（kps_adabn_strict.py）；
  ② 输出写到 wf21/strict/kps/（不覆盖 lidar/kpsize 下的旧文件）；
  ③ train（只用来求"按颜色原型的训练集常数"）每 3 个窗口取 1 个（窗口本身已是间隔 3 的帧）以省时间；val / test 全部帧。
split ∈ train（wsfnb 单帧检测的真实 train 窗口）/ val / test（wsfnb 部署记录 night1003/repl/wsfnb_<split>_kzt7q 的窗口）。
用法：python kps_shape_strict.py train val test"""
import os, sys, time, pickle
sys.dont_write_bytecode = True
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = '@@ROOT@@/output/camnorm/geo1004'
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst1', SCR + '/repl', '@@ROOT@@/tools', '@@ROOT@@', G + '/lidar/kzbody', G + '/lidar/kpsize'):
    if p not in sys.path:
        sys.path.insert(0, p)
import repl_common as RC   # noqa: E402
import kps_eval as KE      # noqa: E402
ADA = HERE + '/adabn/kps_s%d_adabn.pth'
os.makedirs(HERE + '/kps', exist_ok=True)


def paths(sp):
    if sp == 'train':
        d = HERE + '/tags/wsfnb_train_kz'
        return d + '/kz_members_wsfnb_train_kz.npz'
    return RC.tag_paths('wsfnb_%s_kzt7q' % sp)['npz']


def shape(sp):
    import torch
    KE.register()
    import adabn_kz as AB
    C = AB.C
    dev = torch.device('cuda')
    D, info = C.load_npz_info(paths(sp))
    nm_ = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns_ = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    in_wh = tuple(int(v) for v in info['in_wh'])
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    idx = list(range(0, len(keys), 3)) if sp == 'train' else list(range(len(keys)))
    src = AB.NpzSource(D)
    out = {keys[i]: [] for i in idx}
    for seed in (0, 1):
        m = KE.E.build(ADA % seed, torch, dev)
        t0 = time.time()
        for c0 in range(0, len(idx), 16):
            ch = idx[c0:c0 + 16]
            x = np.stack([AB.to_input(AB.recut(D, i, src, in_wh), nm_, ns_) for i in ch])
            with torch.no_grad():
                tb = {'image': torch.from_numpy(np.ascontiguousarray(x[:, None])).to(dev), 'batch_size': len(ch)}
                for mod in m.module_list:
                    tb = mod(tb)
                f = tb['features_2d']
                kk = m.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:])).float().cpu().numpy()
            for b, i in enumerate(ch):
                rb, cb = (int(v) for v in D['rb_cb'][i])
                lay = kk[b, 16:, rb, cb].reshape(8, 3)
                e = np.ptp(lay, 0)
                out[keys[i]].append(float(e[2] / max(e[:2].mean(), 1e-6)))
        print('%s 种子 %d：%d 帧 %.0f s' % (sp, seed, len(idx), time.time() - t0), flush=True)
        del m
        torch.cuda.empty_cache()
    r = {k: float(np.mean(v)) for k, v in out.items()}
    pickle.dump(r, open(HERE + '/kps/shape_%s_wsfnb.pkl' % sp, 'wb'))
    print('%s 关键点高度比中位（全部）%.3f，帧数 %d' % (sp, float(np.median(list(r.values()))), len(r)), flush=True)


if __name__ == '__main__':
    for sp in sys.argv[1:]:
        if not os.path.exists(HERE + '/kps/shape_%s_wsfnb.pkl' % sp):
            shape(sp)
