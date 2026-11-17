# -*- coding: utf-8 -*-
"""wf21/final：朝向成员的 BN 统计重估——"全部合格帧、不按片段"版（GPU，不读真值、不更新权重、不用任何时序信息）。
与 wf21/orient_uda/adabn_st.py 唯一的区别是选帧：
  adabn_st：合格帧（top-1 检测在钩子格且置信 ≥0.5）里每个视频片段随机取 ≤80 张（按片段均衡，2677 张）；
  本脚本：全部合格帧（4130 张），不看片段编号、不看帧序。裁窗同为全严格检测器 wsfnb 在真实 train（间隔 3 帧）上的单帧检测窗口。
被重估的网络 = wf21/orient_uda/runs/<臂>_s{0..4}/ckpt_last.pth（推荐臂 PSP）。
用法：python adabn_all.py PSP | R5   → adabn_all/<臂>_s{0..4}.pth + adabn_all/sel_<臂>.json（R5 = 合规基线⑤的成员 lidar/irlid/release/kzm_s*.pth）"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import json, time
import numpy as np
SCR1003 = '@@ROOT@@/output/camnorm/night1003/scratch'
OU = '@@ROOT@@/output/camnorm/geo1004/gproto/wf21/orient_uda'
ST = '@@ROOT@@/output/camnorm/geo1004/gproto/wf21/strict'
for p in (SCR1003 + '/kzst1', '@@ROOT@@/tools', '@@ROOT@@', OU, ST):
    if p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = 'wsfnb_train_kz'


def main():
    import torch
    import adabn_kz as AB
    import kzst_common as C
    import teach_dump as TD
    arm = sys.argv[1]
    SEEDS = [int(x) for x in os.environ.get('AB_SEEDS', '0,1,2,3,4').split(',')]   # 11-09：可只重估部分种子
    if arm == 'R5':   # 合规基线 ⑤ 的成员：红外+LiDAR 辅助训练、未做目标域自训练
        cks = ['@@ROOT@@/output/camnorm/geo1004/lidar/irlid/release/kzm_s%d.pth' % k for k in range(5)]
    else:
        cks = ['%s/runs/%s_s%d/ckpt_last.pth' % (OU, arm, k) for k in SEEDS]
    outs = ['%s/adabn_all/%s_s%d.pth' % (HERE, arm, k) for k in (SEEDS if arm != 'R5' else range(5))]
    if all(os.path.exists(o) for o in outs):
        print('exists'); return
    D, info = C.load_npz_info('%s/tags/%s/kz_members_%s.npz' % (ST, TAG, TAG))
    assert info.get('split') == 'train' and info.get('dump_finished_ok') is True
    frames, _ = C.load_dump_nogt('%s/tags/%s/real_ens5flip_%s.pkl' % (ST, TAG, TAG))
    T = C.frame_table(D, frames)
    base = T['hook'] & np.isfinite(T['conf_top']) & (np.nan_to_num(T['conf_top'], nan=-1) >= 0.5)
    idx = np.where(base)[0].astype(np.int64)          # 全部合格帧，不按片段取样
    sinfo = dict(n_base=int(base.sum()), n=int(len(idx)), n_seq=int(len(set(T['seq'][idx]))),
                 per_class={c: int((T['cls'][idx] == c).sum()) for c in ('phantom4', 'mavic2')}, rule='all eligible frames (hook & conf>=0.5), no per-sequence cap')
    print('[adabn_all] 选帧 %s' % sinfo, flush=True)
    in_wh = tuple(int(v) for v in info['in_wh'])
    t0 = time.time()
    crops = dict(zip(idx.tolist(), AB.decode_many(D, idx.tolist(), AB.NpzSource(D), in_wh, 8)))
    print('[adabn_all] decoded %d crops %.0f s' % (len(crops), time.time() - t0), flush=True)
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    dev = torch.device('cuda')
    ms = AB.build_models(torch, dev, cks)[0]
    nb = TD.adabn_on_cache(ms, crops, idx.tolist(), nm, ns, torch, dev)
    os.makedirs(HERE + '/adabn_all', exist_ok=True)
    for km, o, ck in zip(ms, outs, cks):
        torch.save({'model_state': {k: v.detach().cpu() for k, v in km.state_dict().items()},
                    'src': '%s | AdaBN on ALL eligible wsfnb_train_kz windows (%d crops, %d batches, no per-sequence cap)' % (ck, len(idx), nb)}, o)
    json.dump(dict(sinfo, arm=arm, ckpts=outs, n_batches=nb, seconds=time.time() - t0, keys=[list(T['keys'][i]) for i in idx]),
              open(HERE + '/adabn_all/sel_%s.json' % arm, 'w', encoding='utf-8'), ensure_ascii=False)
    print('[adabn_all] wrote %s (%.0f s)' % (outs, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
