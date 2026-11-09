# -*- coding: utf-8 -*-
"""【wf23 副本：在 equal5/e5_eval.py 上只加两处——AdaBN 也查 equal5/adabn_out；OU_SWAP=1 时把过滤通过帧的真实机身换到单帧背景库背景上（鲁棒性直测，蒙版 = realcut/masks_kz_<split>.npy，背景按帧固定）】
【wf21/equal5 副本】orient_uda/ou_eval.py 的副本，只改：① AdaBN 权重先找 orient_uda/adabn_out/<名>.pth（只读，RA/PSA 等已有臂不重算），再找 equal5/adabn_out/，都没有才按原法重估并存 equal5/adabn_out/；② 另存逐帧结果 equal5/evals/<OU_TAG>_<split>_perframe.pkl（供按视频段 bootstrap）；③ 结果写 equal5/evals/。读数、打分与原文件逐行相同。
以下为原文件说明。
单网络快速评测（GPU 只推理）：每个朝向网络先做严格单帧 AdaBN（与 wf20/repro/adabn_strict.py --sel sf 同一批 2707 张真实 train 裁窗、
同一顺序 default_rng(0)、batch 16、momentum=None 累计平均），再在真实 val 或 test（严格单帧检测 wanb 的部署裁窗）上读朝向。
读数 = 网络自己在钩子 5×5 窗口里的热图峰值格 → kp2d / z → 部署解码器同款 PnP（位置网络的框尺寸）→ 只换旋转。
四旋翼对称口径：几何 = min_k 测地(R·Rz(90k), R_gt)；航向 = min_k |yaw|（0–45°）；倾斜 = 机体竖轴夹角。真值只打分。
AdaBN 后的权重存到 adabn_out/<name>.pth（部署格式），供 5 成员部署口径评测（recut_multi + eval_orient）直接用。
用法：OU_SPLIT=val python ou_eval.py name=ckpt [name=ckpt ...]   → evals/<OU_TAG>_<split>.json"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import json, time
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
SCR1003 = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR1003 + '/kzst1', SCR1003 + '/repl', '@@ROOT@@/tools', '@@ROOT@@'):
    if p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__))           # equal5
OU = '@@ROOT@@/output/camnorm/geo1004/gproto/wf21/orient_uda'
REPRO = '@@ROOT@@/output/camnorm/geo1004/gproto/wf20/repro'
SCRP = '@@ROOT@@/output/scratch'
import adabn_kz as AB          # noqa: E402
import repl_common as RC       # noqa: E402
from scipy.spatial.transform import Rotation as Rot   # noqa: E402
C = AB.C
KFO = C.KFO
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
SPLIT = os.environ.get('OU_SPLIT', 'val')
SRC_TAG = os.environ.get('OU_SRC', 'wanb_%s_kzt7q' % SPLIT)


def errs(Rp, Rg):
    geo = min(KFO.geodesic(Rp @ RZ[k], Rg) for k in range(4))
    yaw = []
    for k in range(4):
        E = Rg.T @ Rp @ RZ[k]
        yaw.append(abs(np.degrees(np.arctan2(E[1, 0], E[0, 0]))))
    tilt = np.degrees(np.arccos(np.clip(float(Rp[:, 2] @ Rg[:, 2]), -1, 1)))
    return geo, min(yaw), min(yaw[0], yaw[2]), tilt


def adabn_models(torch, dev, cks, names):
    """严格单帧 AdaBN（同 adabn_strict sf 的帧与顺序），裁窗读 orient_uda 的 train 裁窗缓存（与 recut 逐字节同一 warpAffine）。
    已有的 AdaBN 权重（orient_uda/adabn_out 或 equal5/adabn_out）直接读；只对缺的重估（每个网络的 BN 统计互不影响，与一起重估等价）。"""
    sys.path.insert(0, OU)
    import teach_dump as TD
    have = {}
    for n in names:
        for d in (OU + '/adabn_out', '@@ROOT@@/output/camnorm/geo1004/gproto/wf21/equal5/adabn_out', HERE + '/adabn_out'):
            if os.path.exists('%s/%s.pth' % (d, n)):
                have[n] = '%s/%s.pth' % (d, n); break
    miss = [(n, ck) for n, ck in zip(names, cks) if n not in have]
    if miss:
        D, info = TD.load_D()
        crops = TD.crop_cache(D, info)
        keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
        kpos = {k: i for i, k in enumerate(keys)}
        sel = json.load(open(REPRO + '/adabn/kzm_sf/sel.json', encoding='utf-8'))
        sel_idx = [kpos[tuple(k)] for k in sel['keys']]
        nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
        ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
        ms = AB.build_models(torch, dev, [ck for _, ck in miss])[0]
        nb = TD.adabn_on_cache(ms, crops, sel_idx, nm, ns, torch, dev)
        os.makedirs(HERE + '/adabn_out', exist_ok=True)
        for km, (n, ck) in zip(ms, miss):
            o = '%s/adabn_out/%s.pth' % (HERE, n)
            torch.save({'model_state': {k: v.detach().cpu() for k, v in km.state_dict().items()},
                        'src': '%s | strict AdaBN sf (%d crops, %d batches; wf21/equal5/e5_eval.py = orient_uda/ou_eval.py copy)' % (ck, len(sel_idx), nb)}, o)
            have[n] = o
        del ms
    outs = [have[n] for n in names]
    print('[adabn] ' + ' | '.join('%s <- %s' % (n, o) for n, o in zip(names, outs)), flush=True)
    return AB.build_models(torch, dev, outs)[0], outs


def main():
    import torch
    dev = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(0.2)
    pairs = [a.split('=', 1) for a in sys.argv[1:] if '=' in a]
    names = [p[0] for p in pairs]
    raw = os.environ.get('OU_RAW', '0') == '1'
    nets = {}
    if raw:
        for n, m in zip(names, AB.build_models(torch, dev, [p[1] for p in pairs])[0]):
            nets[n] = m
    for n, m in zip(names, adabn_models(torch, dev, [p[1] for p in pairs], names)[0]):
        nets[n + '+BN'] = m
    P = RC.tag_paths(SRC_TAG)
    D, info = C.load_npz_info(P['npz'])
    frames, _ = RC.load_frames(P['pkl'], P['sealed'], with_gt=True)
    nm_ = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns_ = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    ks = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    search = int(info.get('search', 2))
    in_wh = tuple(int(v) for v in info['in_wh'])
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    idx = [i for i, k in enumerate(keys) if k in frames and C.hook_top(D, i, frames[k]) >= 0]
    fva = '%s/kz/val_crops_%s_%d.npy' % (SCRP, SRC_TAG, len(idx))
    if os.path.exists(fva):
        VC = np.load(fva, mmap_mode='r')
    else:
        VC = np.stack(AB.decode_many(D, idx, AB.NpzSource(D), in_wh, 8))
        np.save(fva, VC)
    vpos = {int(i): j for j, i in enumerate(idx)}
    SWAP = os.environ.get('OU_SWAP', '0') == '1'
    if SWAP:
        MKA = np.load('%s/realcut/masks_kz_%s.npy' % (HERE, SPLIT), mmap_mode='r')
        MKQ = np.load('%s/realcut/q_kz_%s.npz' % (HERE, SPLIT))['passed']
        BANK = np.load(SCRP + '/ou/bank_kz_sf_v3.npy', mmap_mode='r')
        assert len(MKA) == len(idx), (len(MKA), len(idx))

    def view_of(i):
        j = vpos[i]
        v = np.asarray(VC[j])
        if SWAP and MKQ[j]:
            r_ = np.random.default_rng(5000 + j)
            bg = np.asarray(BANK[int(r_.integers(len(BANK)))])
            if r_.random() < 0.5:
                bg = bg[:, ::-1]
            a = np.asarray(MKA[j], np.float32)[..., None] / 255.0
            v = np.clip(a * v.astype(np.float32) + (1 - a) * bg.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
        return v
    rows = {a: [] for a in nets}
    t0 = time.time()
    for c0 in range(0, len(idx), 32):
        ch = idx[c0:c0 + 32]
        x = np.stack([AB.to_input(view_of(i), nm_, ns_) for i in ch])
        outs = {}
        with torch.no_grad():
            for a, m in nets.items():
                tb = {'image': torch.from_numpy(np.ascontiguousarray(x[:, None])).to(dev), 'batch_size': len(ch)}
                for mod in m.module_list:
                    tb = mod(tb)
                outs[a] = (tb['pred_center_dict']['kp2d'].float().cpu().numpy(), tb['pred_center_dict']['hm'].float().cpu().numpy()[:, 0])
        for b, i in enumerate(ch):
            k = keys[i]; fr = frames[k]
            t = C.hook_top(D, i, fr)
            p = np.asarray(fr['pred'], np.float64).reshape(-1, 9)[t]
            g = np.asarray(fr['gt'], np.float64).reshape(-1, 9)
            if not len(g):
                continue
            g = g[np.argmin(np.linalg.norm(g[:, :3] - p[:3], axis=1))]
            if np.linalg.norm(g[:3] - p[:3]) > 1.0:
                continue
            geo = KFO.frame_geo(D, i, p[3:6], ks)
            Rg = Rot.from_euler('xyz', g[6:9]).as_matrix()
            rc, cc = (int(v) for v in D['rc_cc'][i])
            for a in nets:
                kp, hm = outs[a]
                r0, r1 = max(0, rc - search), min(hm.shape[1], rc + search + 1)
                c0_, c1 = max(0, cc - search), min(hm.shape[2], cc + search + 1)
                sub = hm[b, r0:r1, c0_:c1]
                jj = int(np.argmax(sub))
                rb, cb = r0 + jj // sub.shape[1], c0_ + jj % sub.shape[1]
                x16 = KFO.f32(kp[b, :, rb, cb].astype(np.float64) / float(D['z'][i]))
                R = KFO.pnp(x16, geo)
                e = errs(R, Rg) if R is not None else (np.nan,) * 4
                rows[a].append((k[0].split('/')[0], k[0], k[1]) + tuple(e) + ((bool(MKQ[vpos[i]]),) if SWAP else (True,)))
    res = {'split': SPLIT, 'src_tag': SRC_TAG, 'nets': dict(pairs), 'arms': {}}
    for a in nets:
        r = rows[a]
        cls = np.array([x[0] for x in r]); seq = np.array([x[1] for x in r]); V = np.array([x[3:7] for x in r], float)
        out = {}
        for c in ('all', 'phantom4', 'mavic2'):
            s = np.ones(len(r), bool) if c == 'all' else cls == c
            v = V[s]
            out[c] = {'n': int(s.sum()), 'fail': int(np.isnan(v[:, 0]).sum()), 'geo_med': float(np.nanmedian(v[:, 0])), 'geo_mean': float(np.nanmean(v[:, 0])),
                      'head_med': float(np.nanmedian(v[:, 1])), 'head_mean': float(np.nanmean(v[:, 1])), 'head_f180_med': float(np.nanmedian(v[:, 2])),
                      'tilt_med': float(np.nanmedian(v[:, 3])), 'tilt_mean': float(np.nanmean(v[:, 3])), 'head_lt10': float(np.nanmean(v[:, 1] < 10))}
        out['per_seq_head_mean'] = {s_: float(np.nanmean(V[seq == s_, 1])) for s_ in sorted(set(seq))}
        out['per_seq_geo_mean'] = {s_: float(np.nanmean(V[seq == s_, 0])) for s_ in sorted(set(seq))}
        res['arms'][a] = out
        print('%-18s %s 几何 中位 %5.2f 均值 %5.2f | 航向 中位 %5.2f 均值 %5.2f | 倾斜 中位 %5.2f | p4 航向均值 %5.2f m2 %5.2f' % (
            a, SPLIT, out['all']['geo_med'], out['all']['geo_mean'], out['all']['head_med'], out['all']['head_mean'],
            out['all']['tilt_med'], out['phantom4']['head_mean'], out['mavic2']['head_mean']), flush=True)
    os.makedirs(HERE + '/evals', exist_ok=True)
    import pickle
    pickle.dump({'split': SPLIT, 'src_tag': SRC_TAG, 'cols': ['cls', 'seq', 'frame', 'geo', 'head', 'head_f180', 'tilt', 'passed'], 'rows': rows},
                open('%s/evals/%s_%s_perframe.pkl' % (HERE, os.environ.get('OU_TAG', 'run'), SPLIT), 'wb'))
    json.dump(res, open('%s/evals/%s_%s.json' % (HERE, os.environ.get('OU_TAG', 'run'), SPLIT), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[eval] %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
