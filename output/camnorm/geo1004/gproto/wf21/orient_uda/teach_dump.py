# -*- coding: utf-8 -*-
"""目标域（真实 train）上的教师读数（GPU 只推理，不读真值）。
输入：严格单帧检测器 wanb 在真实 train 帧上的部署裁窗窗口（wf20/repro/tags/wanb_train_kz，间隔 3 共 4289 帧，dump_train_wanb.py）。
四组朝向网络（都只吃彩色裁窗，都换成真实 train 裁窗的 BN 统计 = 严格单帧 AdaBN，选帧同 wf20/repro/adabn/kzm_sf/sel.json）：
  RGB    发布版 KZ ×5（纯仿真、只学彩色）                    wf20/repro/adabn/kz_sf/kz_s{0..4}.pth
  IRLiD  KZ-IRLiD ×5（仿真里多学 LiDAR 深度剪影 + 红外剪影）  wf20/repro/adabn/kzm_sf/kz_s{0..4}.pth
  IR     KZ-KDir ×2（仿真里被红外关键点教师蒸馏）             本目录 adabn/kzir_sf/kz_s{0,1}.pth（这里现做 AdaBN）
  LiD    KZ-LiD ×2（仿真里多学 LiDAR 深度剪影）               本目录 adabn/kzl_sf/kz_s{0,1}.pth（这里现做 AdaBN）
输出：
  <SCR>/train_crops_4289.npy                裁窗缓存（uint8，与部署钩子逐字节同一 warpAffine）
  teach/teach_train.npz                     D 的全部字段 + hm_win (N,14,5,5) / kp_win (N,14,16,5,5)，成员顺序见 teach/teach_train.json
用法（cwd 任意）：python teach_dump.py"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import json, time
import concurrent.futures as cf
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
SCR1003 = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR1003 + '/kzst1', SCR1003 + '/repl', '@@ROOT@@/tools', '@@ROOT@@'):
    if p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__))
G = '@@ROOT@@/output/camnorm/geo1004'
REPRO = G + '/gproto/wf20/repro'
SCR = os.environ.get('OU_SCR', '@@ROOT@@/output/scratch/ou')
MD = '@@ROOT@@/output/models/uavdet_3d/camnorm'
GROUPS = [
    ('RGB', [REPRO + '/adabn/kz_sf/kz_s%d.pth' % k for k in range(5)], None),
    ('IRLiD', [REPRO + '/adabn/kzm_sf/kz_s%d.pth' % k for k in range(5)], None),
    ('IR', [HERE + '/adabn/kzir_sf/kz_s%d.pth' % k for k in range(2)], [MD + '/sim_full_r34_kp_w12_zoom_kdir/KZKDir_s%d/ckpt/checkpoint_epoch_30.pth' % k for k in range(2)]),
    ('LiD', [HERE + '/adabn/kzl_sf/kz_s%d.pth' % k for k in range(2)], [G + '/lidar/irlid/release/kzl_s%d.pth' % k for k in range(2)]),
]
TAG = 'wanb_train_kz'


def load_D():
    import kzst_common as C
    D, info = C.load_npz_info('%s/tags/%s/kz_members_%s.npz' % (REPRO, TAG, TAG))
    assert info.get('split') == 'train' and info.get('dump_finished_ok') is True
    return D, info


def crop_cache(D, info, workers=8):
    import adabn_kz as AB
    os.makedirs(SCR, exist_ok=True)
    N = len(D['seq'])
    f = '%s/train_crops_%d.npy' % (SCR, N)
    done = f + '.done'
    if os.path.exists(done):
        return np.load(f, mmap_mode='r')
    in_wh = tuple(int(v) for v in info['in_wh'])
    out = np.lib.format.open_memmap(f, mode='w+', dtype=np.uint8, shape=(N, in_wh[1], in_wh[0], 3))
    src = AB.NpzSource(D)
    t0 = time.time()
    chunk = 64
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for j0 in range(0, N, chunk):
            idx = list(range(j0, min(N, j0 + chunk)))
            for i, c in zip(idx, ex.map(lambda i: AB.recut(D, i, src, in_wh), idx)):
                out[i] = c
            if (j0 // chunk) % 10 == 0:
                print('[crops] %d / %d  %.0f s' % (idx[-1] + 1, N, time.time() - t0), flush=True)
    out.flush(); del out
    open(done, 'w').write('ok')
    return np.load(f, mmap_mode='r')


def adabn_on_cache(models, crops, idx, nm, ns, torch, dev, batch=16, seed=0):
    """= adabn_kz.adabn（BN 统计清零、momentum=None 累计平均、batch 16、default_rng(0) 顺序），只是裁窗从缓存读"""
    import adabn_kz as AB
    bn_all = []
    for km in models:
        km.eval()
        bns = [m for m in km.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for bn in bns:
            bn.reset_running_stats(); bn.momentum = None; bn.train()
        bn_all.append(bns)
    order = np.asarray(idx)[np.random.default_rng(seed).permutation(len(idx))]
    nb = 0
    for j in range(0, len(order), batch):
        x = np.stack([AB.to_input(np.asarray(crops[i]), nm, ns) for i in order[j:j + batch]])
        AB.forward(models, x, torch, dev)
        nb += 1
    for bns in bn_all:
        for bn in bns:
            bn.eval()
        assert all(int(bn.num_batches_tracked) == nb for bn in bns)
    return nb


def main():
    import torch
    import adabn_kz as AB
    D, info = load_D()
    N = len(D['seq'])
    crops = crop_cache(D, info)
    print('[teach] crops ready', crops.shape, flush=True)
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    dev = torch.device('cuda')
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    kpos = {k: i for i, k in enumerate(keys)}
    sel = json.load(open(REPRO + '/adabn/kzm_sf/sel.json', encoding='utf-8'))
    sel_idx = [kpos[tuple(k)] for k in sel['keys']]
    assert len(sel_idx) == sel['n']
    # 单模态两组现做 AdaBN（同一批 2707 张真实 train 裁窗）
    for name, outs, srcs in GROUPS:
        if srcs is None or all(os.path.exists(o) for o in outs):
            continue
        ms, _, _, _ = AB.build_models(torch, dev, srcs)
        nb = adabn_on_cache(ms, crops, sel_idx, nm, ns, torch, dev)
        os.makedirs(os.path.dirname(outs[0]), exist_ok=True)
        for km, o, s in zip(ms, outs, srcs):
            sd = {kk: v.detach().cpu() for kk, v in km.state_dict().items()}
            torch.save({'model_state': sd, 'src': '%s | strict AdaBN sf on %d train crops (wf21/orient_uda/teach_dump.py)' % (s, len(sel_idx)),
                        'adabn': {'n': len(sel_idx), 'n_batches': nb, 'sel': REPRO + '/adabn/kzm_sf/sel.json'}}, o)
        print('[teach] AdaBN %s done (%d batches)' % (name, nb), flush=True)
        del ms; torch.cuda.empty_cache()
    names, cks = [], []
    for name, outs, _ in GROUPS:
        for k, o in enumerate(outs):
            names.append('%s_s%d' % (name, k)); cks.append(o)
    models, _, _, in_wh = AB.build_models(torch, dev, cks)
    M = len(models)
    S_ = 5
    HM = np.full((N, M, S_, S_), np.nan, np.float32)
    KP = np.full((N, M, 16, S_, S_), np.nan, np.float32)
    t0 = time.time()
    B = 32
    search = int(info.get('search', 2))
    for j0 in range(0, N, B):
        idx = list(range(j0, min(N, j0 + B)))
        x = np.stack([AB.to_input(np.asarray(crops[i]), nm, ns) for i in idx])
        outs = AB.forward(models, x, torch, dev)
        for jj, i in enumerate(idx):
            for m in range(M):
                h, k_, _, _ = AB.window(D, i, [outs[m][0][jj]], [outs[m][1][jj]], search, torch)
                HM[i, m], KP[i, m] = h[0], k_[0]
        if (j0 // B) % 20 == 0:
            print('[teach] %d / %d  %.0f s' % (idx[-1] + 1, N, time.time() - t0), flush=True)
    od = HERE + '/teach'
    os.makedirs(od, exist_ok=True)
    Z = {k: D[k] for k in D.keys() if k not in ('hm_win', 'kp_win', 'rb_cb', 'kp_deployed')}
    Z.update(hm_win=HM, kp_win=KP)
    np.savez(od + '/teach_train.npz', **Z)
    json.dump({'member_names': names, 'ckpts': cks, 'groups': {g: [n for n in names if n.startswith(g + '_')] for g, _, _ in GROUPS},
               'src_tag': TAG, 'n': N, 'adabn_sel': REPRO + '/adabn/kzm_sf/sel.json', 'search': search,
               'KP_SCALE': info.get('KP_SCALE'), 'norm_mean': info['norm_mean'], 'norm_std': info['norm_std'], 'in_wh': info['in_wh'],
               'seconds': time.time() - t0, 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(od + '/teach_train.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[teach] wrote', od, flush=True)


if __name__ == '__main__':
    main()
