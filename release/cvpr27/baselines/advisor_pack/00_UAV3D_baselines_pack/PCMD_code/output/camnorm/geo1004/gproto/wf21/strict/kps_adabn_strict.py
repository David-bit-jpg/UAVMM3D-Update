# -*- coding: utf-8 -*-
"""wf21/strict：关键点高度比网络（KZ-KPS，lidar/kpsize）的 AdaBN 重做成完全单帧版本。
原版（kps_eval.py adabn → kzbody_eval.adabn）选帧用 night1003 kzst1 的 adabn_selection（g0：前后 ±3 帧位置一致，跨帧），
裁窗来自带热图时序背景的检测。这里换成与 adabn_strict.py --sel sf 完全相同的选帧：
  裁窗 = 全严格检测器 wsfnb 单帧检测在真实 train 帧（间隔 3）上的 zoom-95 窗口（tags/wsfnb_train_kz）；
  选帧 = top-1 检测在钩子格且置信 ≥0.5，每序列随机 ≤80 张（种子 0）。不读标签、不更新权重。
输出 adabn/kps_s{seed}_adabn.pth（与原文件同格式）。用法：python kps_adabn_strict.py 0 1"""
import os, sys, json
sys.dont_write_bytecode = True
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = '@@ROOT@@/output/camnorm/geo1004'
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst1', '@@ROOT@@/tools', '@@ROOT@@', G + '/lidar/kzbody', G + '/lidar/kpsize', HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import kps_eval as KE   # noqa: E402
import adabn_strict as AS   # noqa: E402
TAG = 'wsfnb_train_kz'


def main():
    import torch
    import adabn_kz as AB
    import kzst_common as C
    KE.register()
    D, info = C.load_npz_info('%s/tags/%s/kz_members_%s.npz' % (HERE, TAG, TAG))
    assert info.get('split') == 'train' and info.get('dump_finished_ok') is True
    frames, _ = C.load_dump_nogt('%s/tags/%s/real_ens5flip_%s.pkl' % (HERE, TAG, TAG))
    T = C.frame_table(D, frames)
    idx, sinfo = AS.select('sf', T, {}, {}, seed=0)
    print('[kps-adabn] 选帧', sinfo, flush=True)
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    in_wh = tuple(int(v) for v in info['in_wh'])
    dev = torch.device('cuda')
    os.makedirs(HERE + '/adabn', exist_ok=True)
    for seed in [int(a) for a in sys.argv[1:]] or [0, 1]:
        out = '%s/adabn/kps_s%d_adabn.pth' % (HERE, seed)
        if os.path.exists(out):
            continue
        m = KE.E.build(KE.E.CK % seed, torch, dev)
        nb, nbn = AB.adabn([m], D, idx, AB.NpzSource(D), in_wh, nm, ns, C.ADABN['batch'], C.ADABN['order_seed'], torch, dev, 6)
        torch.save({'model_state': {k: v.detach().cpu() for k, v in m.state_dict().items()}, 'src': KE.E.CK % seed,
                    'adabn': dict(sinfo, frames=int(len(idx)), batches=nb, bn_layers=nbn, windows=TAG, selection='strict single-frame sf')}, out)
        print('KPS AdaBN 种子 %d：%d 帧 %d 批 %d 个 BN 层 → %s' % (seed, len(idx), nb, nbn, out), flush=True)


if __name__ == '__main__':
    main()
