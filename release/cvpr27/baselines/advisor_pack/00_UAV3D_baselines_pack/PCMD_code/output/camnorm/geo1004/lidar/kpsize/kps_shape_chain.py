# -*- coding: utf-8 -*-
"""KZ-KPS 阶段 A'（PREREG_kps f3815298）：老师原链 LiDAR 关键点 → 尺寸（高度比）→ 朝向 PnP 模板，放进现最终版全链。
    shape <split>     KPS 种子 0、1 在部署裁窗上读出预测形状 → 每帧关键点高度比 r_kp（两种子平均）→ kpsize/shape_<split>.pkl
    chain <split>     朝向 PnP 模板高度改成 0.843 × r_kp（片段内只用过去帧滚动中位）× 平均(l, w)，全链在线一致 → 用户口径"""
import json, os, pickle, sys, time
import numpy as np
ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
K_ = G + '/lidar/kpsize'
sys.path.insert(0, K_)
A_MAP = 0.843


def shape(split):
    import torch
    import kps_eval as KE
    KE.register()
    import adabn_kz as AB
    C = AB.C
    import repl_common as RC
    dev = torch.device('cuda')
    P = RC.tag_paths('%s_%s_kzt7q' % (os.environ.get('UAV_POSTAG', 'st2p'), split))
    D, info = C.load_npz_info(P['npz'])
    nm_ = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns_ = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    in_wh = tuple(int(v) for v in info['in_wh'])
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    src = AB.NpzSource(D)
    out = {k: [] for k in keys}
    for seed in (0, 1):
        m = KE.E.build(KE.E.ADA % seed, torch, dev)
        t0 = time.time()
        for c0 in range(0, len(keys), 16):
            ch = list(range(c0, min(c0 + 16, len(keys))))
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
        print('%s 种子 %d：%d 帧 %.0f s' % (split, seed, len(keys), time.time() - t0), flush=True)
    r = {k: float(np.mean(v)) for k, v in out.items()}
    pickle.dump(r, open(K_ + '/shape_%s%s.pkl' % (split, os.environ.get('UAV_SHAPESFX', '')), 'wb'))
    cls = np.array([k[0].split('/')[0] for k in r])
    v = np.array(list(r.values()))
    print('%s 关键点高度比中位：mavic2 %.3f phantom4 %.3f' % (split, np.median(v[cls == 'mavic2']), np.median(v[cls == 'phantom4'])), flush=True)


def running(rk, keys, mode):
    """按片段（同 seq）时间序：causal = 过去（含当前）最近 100 帧中位；offline = 整段中位"""
    by = {}
    for k in keys:
        by.setdefault(k[0], []).append(k)
    out = {}
    for s, ks in by.items():
        ks.sort(key=lambda k: int(os.path.splitext(k[1])[0]))
        v = np.array([rk[k] for k in ks])
        for j, k in enumerate(ks):
            out[k] = float(np.median(v[max(0, j - 99):j + 1])) if mode == 'causal' else float(np.median(v))
    return out


def chain(split):
    sys.path.insert(0, G + '/finalads')
    import temporal_filter as T
    F, RC, C, KFO = T.F, T.RC, T.C, T.KFO
    rk = pickle.load(open(K_ + '/shape_%s%s.pkl' % (split, os.environ.get('UAV_SHAPESFX', '')), 'rb'))
    P = RC.tag_paths('%s_%s_kzt7q' % (os.environ.get('UAV_POSTAG', 'st2p'), split))
    D, info = C.load_npz_info(P['npz'])
    res = {}
    for name, mode, a in (('现最终版', None, None), ("A' 关键点高度比 → 模板（换算、在线）", 'causal', A_MAP),
                          ("描述：不换算（在线）", 'causal', 1.0), ("描述：换算、整段中位", 'offline', A_MAP)):
        frames, _ = RC.load_frames(P['pkl'], with_gt=False)
        if mode:
            rr = running(rk, list(frames.keys()), mode)
            for k, f in frames.items():
                p = np.asarray(f['pred'], np.float64).copy()
                if len(p) and k in rr:
                    p[:, 5] = a * rr[k] * p[:, 3:5].mean(1)
                f['pred'] = p
        keys, B = RC.compute_arms(D, info, frames, members=list(range(5)))
        frames_gt, _ = RC.load_frames(P['pkl'], P['sealed'], with_gt=True)
        Fr = KFO.build_arm_frames(frames_gt, keys, B['SATM'], B['top'])
        votes = T.member_votes(D, info, frames, keys, B['SATM'])
        recs = [{'gt': np.asarray(f['gt'], np.float64).reshape(-1, 9), 'pred': np.asarray(f['pred'], np.float64).reshape(-1, 9),
                 'conf': np.asarray(f['conf'], np.float64), 'K': f['K_in'], 'seq_id': f['seq'], 'frame_id': f['frame']} for f in Fr]
        recs = F.apply(recs, F.factors(split), T.TAU, alpha=T.ALPHA)
        for r, v in zip(recs, votes):
            r['votes'] = v
        out = T.do(recs, 'hmm_causal_g3_w9')
        s = F.summ(out, 'kps_chain_%s_%d' % (split, len(res)), ('indoor', 'laa'))
        _, ang, _ = T.errs(out)
        cls = np.array([x['seq_id'].split('/')[0] for x in out])
        s.update(ang_mavic2=float(np.nanmedian(ang[cls == 'mavic2'])), ang_phantom4=float(np.nanmedian(ang[cls == 'phantom4'])), flip90=float(np.nanmean(ang > 90)))
        res[name] = s
        print('%s %s：ADS indoor %.2f laa %.2f 角度 %.1f°（mavic2 %.1f° / phantom4 %.1f°）反向 %.1f%%' % (
            split, name, s['ads_indoor'], s['ads_laa'], s['ang'], s['ang_mavic2'], s['ang_phantom4'], 100 * s['flip90']), flush=True)
    json.dump(res, open(K_ + '/chain_%s.json' % split, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    L = ['', "## 老师原链：LiDAR 关键点 → 高度比 → 朝向 PnP 模板（PREREG_kps f3815298 A'）· 真实%s · 在线一致" % {'val': '验证', 'test': '测试'}[split], '',
         '| 配置 | 位置中位 | 角度中位 | mavic2 / phantom4 角度 | ADS indoor | ADS laa | 机头反向 |', '|---|---|---|---|---|---|---|']
    for k, s in res.items():
        L.append('| %s | %.3f m | %.1f° | %.1f° / %.1f° | %.2f | %.2f | %.1f%% |' % (k, s['pos'], s['ang'], s['ang_mavic2'], s['ang_phantom4'], s['ads_indoor'], s['ads_laa'], 100 * s['flip90']))
    open(G + '/FINAL_ADS.md', 'a', encoding='utf-8').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
    {'shape': shape, 'chain': chain}[sys.argv[1]](sys.argv[2])
