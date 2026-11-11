# -*- coding: utf-8 -*-
"""wf21/strict/kps_adabn_strict.py + kps_shape_strict.py 的无标签链副本（11-10），高度网络只用 1 个（KPS 种子 0，与 wf24 最终版 FE_KPSSFX=_k0 相同）。
(1) adabn：KPS 种子 0 的 BN 统计在新检测器 <arm> 的真实 train 单帧检测窗口（nolabel/<arm>/tags/<arm>_train_kz）上重估；
    选帧与 adabn_strict.select('sf') 相同（top-1 在钩子格且置信 ≥0.5，每段视频随机 ≤80 张，rng 种子 0），去掉按机型（cls）统计。
    → nolabel/<arm>/adabn/kps_s0_adabn.pth
(2) shape：逐帧关键点高度比（z 向跨度 / 平均水平跨度）；train 每 3 个窗口取 1（窗口本身间隔 3 帧），val/test 全部帧（<arm>_<split>_kzt7q 部署窗口）。
    → nolabel/<arm>/kps/shape_<split>_<arm>_k0.pkl
用法：python kps_nl.py <arm> adabn | shape train val test"""
import os, sys, time, pickle, json
sys.dont_write_bytecode = True
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'; NL = G + '/gproto/nolabel'
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst1', SCR + '/repl', '@@ROOT@@/tools', '@@ROOT@@', G + '/lidar/kzbody', G + '/lidar/kpsize', NL):
    if p not in sys.path:
        sys.path.insert(0, p)
import repl_common as RC   # noqa: E402
import kps_eval as KE      # noqa: E402
from adabn_all_nl import label_free_table   # noqa: E402
arm, what = sys.argv[1], sys.argv[2]
A = '%s/%s' % (NL, arm)
TAG = '%s_train_kz' % arm
ADA = A + '/adabn/kps_s0_adabn.pth'


def adabn():
    import torch
    import adabn_kz as AB
    import kzst_common as C
    if os.path.exists(ADA):
        print('exists', ADA); return
    KE.register()
    D, info = C.load_npz_info('%s/tags/%s/kz_members_%s.npz' % (A, TAG, TAG))
    assert info.get('split') == 'train' and info.get('dump_finished_ok') is True
    frames, _ = C.load_dump_nogt('%s/tags/%s/real_ens5flip_%s.pkl' % (A, TAG, TAG))
    keys, conf, hook = label_free_table(D, frames, C)
    seq = np.array([k[0] for k in keys])
    m = hook & np.isfinite(conf) & (np.nan_to_num(conf, nan=-1) >= 0.5)
    rng = np.random.default_rng(0)
    idx = []
    for s in sorted(set(seq[m])):                       # 同 adabn_strict.select('sf')：每段视频 ≤80 张
        ii = np.where(m & (seq == s))[0]
        if len(ii) > 80:
            ii = np.sort(rng.choice(ii, 80, replace=False))
        idx += ii.tolist()
    idx = np.array(sorted(idx), np.int64)
    sinfo = dict(n_base=int(m.sum()), n=int(len(idx)), n_seq=int(len(set(seq[idx]))))
    print('[kps-adabn-nl] 选帧', sinfo, flush=True)
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    in_wh = tuple(int(v) for v in info['in_wh'])
    dev = torch.device('cuda')
    os.makedirs(A + '/adabn', exist_ok=True)
    mdl = KE.E.build(KE.E.CK % 0, torch, dev)
    nb, nbn = AB.adabn([mdl], D, idx, AB.NpzSource(D), in_wh, nm, ns, C.ADABN['batch'], C.ADABN['order_seed'], torch, dev, 6)
    torch.save({'model_state': {k: v.detach().cpu() for k, v in mdl.state_dict().items()}, 'src': KE.E.CK % 0,
                'adabn': dict(sinfo, frames=int(len(idx)), batches=nb, bn_layers=nbn, windows=TAG, selection='strict single-frame sf (no-label copy)')}, ADA)
    print('KPS AdaBN 种子 0：%d 帧 %d 批 %d 个 BN 层 → %s' % (len(idx), nb, nbn, ADA), flush=True)


def paths(sp):
    if sp == 'train':
        return '%s/tags/%s/kz_members_%s.npz' % (A, TAG, TAG)
    return RC.tag_paths('%s_%s_kzt7q' % (arm, sp))['npz']


def shape(sp):
    import torch
    out_fn = '%s/kps/shape_%s_%s_k0.pkl' % (A, sp, arm)
    if os.path.exists(out_fn):
        print('exists', out_fn); return
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
    m = KE.E.build(ADA, torch, dev)
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
    r = {k: float(np.mean(v)) for k, v in out.items()}
    os.makedirs(A + '/kps', exist_ok=True)
    pickle.dump(r, open(out_fn, 'wb'))
    print('%s %s 关键点高度比中位 %.3f，帧数 %d，%.0f s' % (arm, sp, float(np.median(list(r.values()))), len(r), time.time() - t0), flush=True)


if __name__ == '__main__':
    if what == 'adabn':
        adabn()
    else:
        for sp in sys.argv[3:]:
            shape(sp)
