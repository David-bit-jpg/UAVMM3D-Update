# -*- coding: utf-8 -*-
"""wf21/final/adabn_all.py 的无标签链副本（11-10）：朝向学生（S6I，种子 0、1，不重训）的 BN 统计在新检测器 <arm> 的真实 train 单帧检测窗口上重估。
选帧规则相同：全部合格帧（top-1 检测在钩子格且置信 ≥0.5），不看片段、不看帧序。改动：窗口 = nolabel/<arm>/tags/<arm>_train_kz；
不再按机型（cls，来自序列路径 = 真值）统计选帧数；输出 nolabel/<arm>/adabn_all/S6I_s{0,1}.pth。
用法：python adabn_all_nl.py <arm> [臂=S6I]"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import json, time
import numpy as np
SCR1003 = '@@ROOT@@/output/camnorm/night1003/scratch'
OU = '@@ROOT@@/output/camnorm/geo1004/gproto/wf21/orient_uda'
NL = '@@ROOT@@/output/camnorm/geo1004/gproto/nolabel'
for p in (SCR1003 + '/kzst1', '@@ROOT@@/tools', '@@ROOT@@', OU):
    if p not in sys.path:
        sys.path.insert(0, p)


def label_free_table(D, frames, C):
    """kzst_common.frame_table 去掉 cls（序列路径前缀 = 机型真值）：只留 keys / top-1 置信 / 钩子标志"""
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    assert set(keys) == set(frames.keys())
    conf = np.full(len(keys), np.nan); hook = np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        fr = frames[k]
        t = C.KFO.top_index(fr)
        if t >= 0:
            conf[i] = float(fr['conf'][t])
        hook[i] = C.hook_top(D, i, fr) >= 0
    return keys, conf, hook


def main():
    import torch
    import adabn_kz as AB
    import kzst_common as C
    import teach_dump as TD
    arm = sys.argv[1]; oarm = sys.argv[2] if len(sys.argv) > 2 else 'S6I'
    SEEDS = [0, 1]
    TAG = '%s_train_kz' % arm
    cks = ['%s/runs/%s_s%d/ckpt_last.pth' % (OU, oarm, k) for k in SEEDS]
    od = '%s/%s/adabn_all' % (NL, arm)
    outs = ['%s/%s_s%d.pth' % (od, oarm, k) for k in SEEDS]
    if all(os.path.exists(o) for o in outs):
        print('exists'); return
    D, info = C.load_npz_info('%s/%s/tags/%s/kz_members_%s.npz' % (NL, arm, TAG, TAG))
    assert info.get('split') == 'train' and info.get('dump_finished_ok') is True
    frames, _ = C.load_dump_nogt('%s/%s/tags/%s/real_ens5flip_%s.pkl' % (NL, arm, TAG, TAG))
    keys, conf, hook = label_free_table(D, frames, C)
    base = hook & np.isfinite(conf) & (np.nan_to_num(conf, nan=-1) >= 0.5)
    idx = np.where(base)[0].astype(np.int64)
    sinfo = dict(n_base=int(base.sum()), n=int(len(idx)), rule='all eligible frames (hook & conf>=0.5), no per-sequence cap')
    print('[adabn_all_nl] %s 选帧 %s' % (arm, sinfo), flush=True)
    in_wh = tuple(int(v) for v in info['in_wh'])
    t0 = time.time()
    crops = dict(zip(idx.tolist(), AB.decode_many(D, idx.tolist(), AB.NpzSource(D), in_wh, 8)))
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    dev = torch.device('cuda')
    ms = AB.build_models(torch, dev, cks)[0]
    nb = TD.adabn_on_cache(ms, crops, idx.tolist(), nm, ns, torch, dev)
    os.makedirs(od, exist_ok=True)
    for km, o, ck in zip(ms, outs, cks):
        torch.save({'model_state': {k: v.detach().cpu() for k, v in km.state_dict().items()},
                    'src': '%s | AdaBN on ALL eligible %s windows (%d crops, %d batches)' % (ck, TAG, len(idx), nb)}, o)
    json.dump(dict(sinfo, arm=arm, ckpts=outs, n_batches=nb, seconds=time.time() - t0, keys=[list(keys[i]) for i in idx]),
              open('%s/sel_%s.json' % (od, oarm), 'w', encoding='utf-8'), ensure_ascii=False)
    print('[adabn_all_nl] wrote %s (%.0f s)' % (outs, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
