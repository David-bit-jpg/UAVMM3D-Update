# -*- coding: utf-8 -*-
"""严格单帧 AdaBN（GPU，不更新权重、不读真值、不用伪标签、不用任何时序信息）。
做法同 night1003 kzst1/adabn_kz.py（函数原样调用：重裁 recut、AB.adabn = 每层 BN 统计清零后 momentum=None 累计平均、batch 16、
default_rng(0) 顺序、5 个成员看同一批），只换两处让它完全单帧：
  ① 裁窗 = 部署同款单帧检测器 wanb（推理不做热图时序背景）在真实 train 帧上的 zoom-95 窗口（tags/wanb_train_kz，dump_train_wanb.py）；
  ② 选帧 = 只看本帧：top-1 检测在钩子格且置信 ≥ 0.5（旧版另有"前后 ±3 帧位置一致"的跨帧筛，这里去掉），
     每个序列随机取 ≤ 80 张（只为各序列均衡；随机而不是按时间均匀，完全不用帧序）。
选帧变体（--sel）：
  sf    = 上面的基础集合；
  gate  = 基础集合 ∩ 红外⇄LiDAR 互验通过：同一帧 top-1 检测上，红外剪影第二段（V2Pp，5 网）与 LiDAR 第二段（V2Lp，5 网）
          各自读出的距离修正量之差 |δ_IR − δ_LiD| 在基础集合的 60% 分位以内（两个特权老师对这一帧的几何读数一致 = 裁窗里是干净的真实机身）；
  p0/p1 = 基础集合按"红外剪影圈机身 → RGB 机身亮度"的 2 个颜色原型分组（原型 = train 自己亮度的 1 维 2 均值，不读标签），每组单独重估一套 BN；
  gp0/gp1 = gate 后再按颜色原型分组。
--base kz（发布版 KZ，纯仿真 RGB）| kzm（KZ-IRLiD：仿真里多学"LiDAR 深度剪影 + LiDAR 区域内红外剪影"两个辅助答案，推理只看 RGB）
输出 adabn/<base>_<sel>/kz_s{k}.pth（部署格式）+ sel.json。用法：python adabn_strict.py --base kz --sel sf"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import argparse, json, pickle, time
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst1', '@@ROOT@@/tools', '@@ROOT@@'):
    if p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__))
G = '@@ROOT@@/output/camnorm/geo1004'
BASES = {'kz': ['@@ROOT@@/output/release_20260923/kz_s%d.pth' % k for k in range(5)],
         'kzm': [G + '/lidar/irlid/release/kzm_s%d.pth' % k for k in range(5)]}
TRAIN_TAG = 'wsfnb_train_kz'   # wf21/strict：裁窗换成全严格检测器 wsfnb 的单帧检测窗口


def train_lum():
    """train 帧机身亮度（color_type.py 同法：ST2 五个单模型 top-1 检测上第二段自己画的红外剪影圈出的机身平均亮度，取中位）"""
    acc = {}
    for k in range(5):
        recs = pickle.load(open(G + '/st2_train_recs/train_s%d.pkl' % k, 'rb'))['records']
        Fc = pickle.load(open('%s/colorfeat/train_k%d_j%d.pkl' % (G, k, k), 'rb'))
        for n, (ri, pi) in enumerate(Fc['items']):
            r = recs[ri]
            if pi != int(np.argmax(r['conf'])) or not np.isfinite(Fc['lum'][n]):
                continue
            acc.setdefault((str(r['seq_id']), str(r['frame_id'])), []).append(float(Fc['lum'][n]))
    return {kk: float(np.median(v)) for kk, v in acc.items()}


def km2(v, it=50):
    c = np.quantile(v, [0.25, 0.75])
    for _ in range(it):
        aa = np.abs(v[:, None] - c[None]).argmin(1); c = np.array([np.median(v[aa == j]) for j in range(2)])
    return np.sort(c)


def irlid_disagree():
    """(seq, frame) → |δ_IR − δ_LiD|（k = 1 处，各自 5 网平均；gproto/fp 的 fp_st2nb[_V2Lp]_c*.npz，训练帧单帧推理）"""
    out = {}
    for c in range(4):
        a = np.load(G + '/gproto/fp/fp_st2nb_c%d.npz' % c, allow_pickle=True)
        b = np.load(G + '/gproto/fp/fp_st2nb_V2Lp_c%d.npz' % c, allow_pickle=True)
        assert np.array_equal(a['seq'], b['seq']) and np.array_equal(a['frame'], b['frame'])
        j = int(np.argmin(np.abs(a['KS'] - 1.0)))
        di = np.nanmean(a['D'][:, j, :], 1); dl = np.nanmean(b['D'][:, j, :], 1)
        for s, f, x, y in zip(a['seq'], a['frame'], di, dl):
            if np.isfinite(x) and np.isfinite(y):
                out[(str(s), str(f))] = abs(float(x) - float(y))
    return out


def select(sel_name, T, lum, dis, cap=80, seed=0):
    base = T['hook'] & np.isfinite(T['conf_top']) & (np.nan_to_num(T['conf_top'], nan=-1) >= 0.5)
    keys = T['keys']
    info = {'n_base': int(base.sum())}
    m = base.copy()
    if sel_name.startswith('g'):
        dv = np.array([dis.get(k, np.nan) for k in keys])
        thr = float(np.nanquantile(dv[base], 0.6))
        m &= np.isfinite(dv) & (dv <= thr)
        info.update(gate_thr=thr, n_gate=int(m.sum()), base_with_irlid=int((base & np.isfinite(dv)).sum()))
    if sel_name.endswith(('p0', 'p1')):
        lv = np.array([lum.get(k, np.nan) for k in keys])
        cc = km2(np.array(list(lum.values())))
        grp = np.where(np.isfinite(lv), np.abs(lv[:, None] - cc[None]).argmin(1), -1)
        m &= grp == int(sel_name[-1])
        info.update(proto_centers=cc.tolist(), n_group=int(m.sum()))
    rng = np.random.default_rng(seed)
    idx = []
    for s in sorted(set(T['seq'][m])):
        ii = np.where(m & (T['seq'] == s))[0]
        if len(ii) > cap:
            ii = np.sort(rng.choice(ii, cap, replace=False))
        idx += ii.tolist()
    idx = np.array(sorted(idx), np.int64)
    info.update(n=int(len(idx)), n_seq=int(len(set(T['seq'][idx]))), per_class={c: int((T['cls'][idx] == c).sum()) for c in ('phantom4', 'mavic2')})
    return idx, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, choices=sorted(BASES))
    ap.add_argument('--sel', required=True, choices=['sf', 'gate', 'p0', 'p1', 'gp0', 'gp1'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0, help='每序列随机取 ≤80 张的随机种子（0 = 主结果；1、2 = 换子样本复核）')
    a = ap.parse_args()
    od = '%s/adabn/%s_%s%s' % (HERE, a.base, a.sel, '' if a.seed == 0 else '_r%d' % a.seed)
    if os.path.exists(od + '/sel.json'):
        print('[adabn] exists', od); return
    os.makedirs(od, exist_ok=True)
    import torch
    import adabn_kz as AB
    import kzst_common as C
    D, info = C.load_npz_info('%s/tags/%s/kz_members_%s.npz' % (HERE, TRAIN_TAG, TRAIN_TAG))
    assert info.get('split') == 'train' and info.get('dump_finished_ok') is True
    frames, _ = C.load_dump_nogt('%s/tags/%s/real_ens5flip_%s.pkl' % (HERE, TRAIN_TAG, TRAIN_TAG))
    T = C.frame_table(D, frames)
    lum = train_lum() if 'p' in a.sel else {}
    dis = irlid_disagree() if a.sel.startswith('g') else {}
    idx, sinfo = select(a.sel, T, lum, dis, seed=a.seed)
    sinfo['subsample_seed'] = a.seed
    print('[adabn] %s_%s 选帧 %s' % (a.base, a.sel, sinfo), flush=True)
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    dev = torch.device('cuda')
    src = AB.NpzSource(D)
    models, _, _, in_wh = AB.build_models(torch, dev, BASES[a.base])
    t0 = time.time()
    nb, nbn = AB.adabn(models, D, idx, src, in_wh, nm, ns, 16, 0, torch, dev, a.workers)
    outs = []
    for k, km in enumerate(models):
        src_ck = torch.load(BASES[a.base][k], map_location='cpu', weights_only=False)
        sd = {kk: v.detach().cpu() for kk, v in km.state_dict().items()}
        for kk, v in src_ck['model_state'].items():
            if kk in sd and not any(kk.endswith(s) for s in ('running_mean', 'running_var', 'num_batches_tracked')):
                assert torch.equal(sd[kk], v.cpu()), kk
        p = '%s/kz_s%d.pth' % (od, k)
        torch.save({'model_state': sd, 'epoch': src_ck.get('epoch'), 'src': '%s | strict AdaBN %s on %d train crops (wf21/strict/adabn_strict.py, windows from wsfnb_train_kz)' % (BASES[a.base][k], a.sel, len(idx)),
                    'adabn': dict(sinfo, n_batches=nb, batch=16, order_seed=0, base=BASES[a.base][k])}, p)
        outs.append(p)
    json.dump(dict(sinfo, base=a.base, sel=a.sel, ckpts=outs, n_bn_layers=nbn, n_batches=nb, seconds=time.time() - t0,
                   keys=[list(T['keys'][i]) for i in idx]), open(od + '/sel.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print('[adabn] wrote %s (%d crops, %d batches, %.0f s)' % (od, len(idx), nb, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
