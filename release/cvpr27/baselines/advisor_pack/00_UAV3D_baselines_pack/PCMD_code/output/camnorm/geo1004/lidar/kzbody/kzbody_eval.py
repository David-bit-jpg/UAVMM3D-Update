# -*- coding: utf-8 -*-
"""KZ-Body 前后裁判评测（PREREG_kzbody 61cb4b2a）。
    adabn <seed>                 真实 train 裁窗重估 BN（与 adabn_kzm 同配方：kzst1 adabn_selection、batch 16、顺序种子 0）→ kzbody/adabn/kzb_s<seed>_adabn.pth
    maps <split> <seeds>         在现最终版的部署裁窗上跑（逐种子），对现最终版逐帧融合朝向的 4 个象限打分 → kzbody/arb_<split>.npz
    eval <split> [rule beta]     裁判准确率 + 票数注入在线一致 → 用户口径 ADS（不给 rule/beta 时跑预登记的 6 组）
全程不读真值（只有 eval 的准确率/ADS 读真值）。"""
import json, os, pickle, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = '@@ROOT@@'
TOOLS = ROOT + '/tools'
G = ROOT + '/output/camnorm/geo1004'
SCR = ROOT + '/output/camnorm/night1003/scratch'
REPL = ROOT + '/output/camnorm/night1003/repl'
B = G + '/lidar/kzbody'
CFG = 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_irlid_body.yaml'
CK = ROOT + '/output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_irlid_body/KZBs%d/ckpt/checkpoint_epoch_8.pth'
ADA = B + '/adabn/kzb_s%d_adabn.pth'
for p in (ROOT, TOOLS, G + '/lidar/kzlid', B, G + '/finalads', SCR + '/repl', SCR + '/kzst1'):
    sys.path.insert(0, p)
PROTO8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]
ONLINE, OFFLINE = 'hmm_causal_g3_w9', 'hmm_offline_g30_w15'


def build(ck, torch, dev):
    import uavdet3d.model.detectors as dets
    from center_det_mm import CenterDetMM
    from center_det_mmb import CenterDetMMB
    dets.__all__['CenterDetMM'] = CenterDetMM
    dets.__all__['CenterDetMMB'] = CenterDetMMB
    from easydict import EasyDict
    from uavdet3d.config import cfg_from_yaml_file
    from uavdet3d.datasets import build_dataloader
    from uavdet3d.model import build_network
    from uavdet3d.utils import common_utils
    cwd = os.getcwd()
    os.chdir(TOOLS)
    cfg = EasyDict()
    cfg_from_yaml_file(CFG, cfg)
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=1, dist=False, workers=0, logger=common_utils.create_logger(), training=False)
    os.chdir(cwd)
    m = build_network(cfg.MODEL, ds)
    sd = torch.load(ck, map_location='cpu', weights_only=False)['model_state']
    m.load_state_dict(sd, strict=True)
    return m.to(dev).eval()


def adabn(seed):
    import torch
    import adabn_kz as AB
    C = AB.C
    dev = torch.device('cuda')
    m = build(CK % seed, torch, dev)
    Dtr, itr = C.load_npz_info(SCR + '/kzst1/kz_members_%s.npz' % C.TRAIN_TAG)
    frames_tr, _ = C.load_dump_nogt(SCR + '/kzst1/real_ens5flip_kpKZavg_zoom_%s.pkl' % C.TRAIN_TAG)
    sel = C.adabn_selection(Dtr, frames_tr)[0]
    nm = np.asarray(itr['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(itr['norm_std'], np.float32).reshape(3, 1, 1)
    in_wh = tuple(int(v) for v in itr['in_wh'])
    nb, nbn = AB.adabn([m], Dtr, sel, AB.NpzSource(Dtr), in_wh, nm, ns, C.ADABN['batch'], C.ADABN['order_seed'], torch, dev, 6)
    os.makedirs(B + '/adabn', exist_ok=True)
    torch.save({'model_state': {k: v.detach().cpu() for k, v in m.state_dict().items()}, 'src': CK % seed,
                'adabn': dict(frames=int(len(sel)), batches=nb, bn_layers=nbn)}, ADA % seed)
    print('AdaBN 种子 %d：%d 帧 %d 批 %d 个 BN 层' % (seed, len(sel), nb, nbn), flush=True)


def load_recs(split):
    return pickle.load(open(G + '/kzst7/kzt7q_%s_recs_v.pkl' % split, 'rb'))   # 新最终版（② 改法 Q；PREREG_kzbody 修订）


def frame_scores(aux, body, rb, cb, Kc, p, st):
    """aux (3,h,w) / body (6,h,w) numpy；p = 最终版逐帧框 9 维；st = 特征步长（裁窗 512 / 图宽 64 = 8）→ (机体图 L1 打分 4，机头余弦 4，有效格数)"""
    t, lwh = p[:3], np.maximum(p[3:6], 1e-3)
    R = Rot.from_euler('xyz', p[6:9]).as_matrix()
    c8 = (PROTO8 * lwh) @ R.T + t
    if (c8[:, 2] < 0.05).any():
        return np.full(4, np.nan), np.full(4, np.nan), 0
    q = c8 @ Kc.T
    uv = q[:, :2] / q[:, 2:3]
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    ex, ey = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
    h, w = aux.shape[1:]
    prob = 1.0 / (1.0 + np.exp(-aux[0]))
    vv, uu = np.meshgrid(np.arange(h) * st + (st - 1) / 2.0, np.arange(w) * st + (st - 1) / 2.0, indexing='ij')
    cell = (prob > 0.5) & (uu >= x0 - ex) & (uu <= x1 + ex) & (vv >= y0 - ey) & (vv <= y1 + ey)
    sb = np.full(4, np.nan)
    n = int(cell.sum())
    if n >= 3:
        I, J = np.nonzero(cell)
        depth = t[2] + aux[1][I, J] * lwh.max()
        P = (np.linalg.inv(Kc) @ np.stack([uu[I, J], vv[I, J], np.ones(n)])) * depth
        wg = prob[I, J]
        for k in range(4):
            Rq = R @ RZ[k]
            X = Rq.T @ (P - t[:, None]) / (0.5 * lwh[:, None])
            sb[k] = float((wg * (np.abs(body[0][I, J] - X[0]) + np.abs(body[1][I, J] - X[1]))).sum() / wg.sum())
    d = body[3:6, rb, cb].astype(np.float64)
    d = d / max(np.linalg.norm(d), 1e-6)
    nc = np.array([float(d @ (R @ RZ[k])[:, 0]) for k in range(4)])
    return sb, nc, n


def maps(split, seeds):
    import torch
    import adabn_kz as AB
    C = AB.C
    sys.path.insert(0, SCR + '/kzst2')
    import kzst2_common as S2
    dev = torch.device('cuda')
    D, info = C.load_npz_info('%s/st2p_%s_kzt7q/kz_members_st2p_%s_kzt7q.npz' % (REPL, split, split))
    _, itr = C.load_npz_info(SCR + '/kzst1/kz_members_%s.npz' % C.TRAIN_TAG)
    assert info.get('norm_mean') == itr.get('norm_mean') and info.get('norm_std') == itr.get('norm_std')
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    in_wh = tuple(int(v) for v in info['in_wh'])
    R = load_recs(split)
    pos = {(str(r['seq_id']), str(r['frame_id'])): n for n, r in enumerate(R)}
    idx = [i for i in range(len(D['seq'])) if (str(D['seq'][i]), str(D['frame'][i])) in pos]
    N = len(R)
    SB = np.full((len(seeds), N, 4), np.nan)
    NC = np.full((len(seeds), N, 4), np.nan)
    NCELL = np.zeros((len(seeds), N), int)
    src = AB.NpzSource(D)
    for si, seed in enumerate(seeds):
        m = build(ADA % seed, torch, dev)
        t0 = time.time()
        for c0 in range(0, len(idx), 16):
            ch = idx[c0:c0 + 16]
            x = np.stack([AB.to_input(AB.recut(D, i, src, in_wh), nm, ns) for i in ch])
            with torch.no_grad():
                tb = {'image': torch.from_numpy(np.ascontiguousarray(x[:, None])).to(dev), 'batch_size': len(ch)}
                for mod in m.module_list:
                    tb = mod(tb)
                f = tb['features_2d']
                f = f.reshape(f.shape[0] * f.shape[1], *f.shape[2:])
                aux = m.aux_head(f).float().cpu().numpy()
                body = m.body_head(f).float().cpu().numpy()
            for b, i in enumerate(ch):
                n = pos[(str(D['seq'][i]), str(D['frame'][i]))]
                if len(R[n]['pred']) == 0:
                    continue
                x0, y0, _, _, s = [float(v) for v in D['crop'][i]]
                Kc = S2.crop_K(D['K_raw_src'][i], x0, y0, s)
                rb, cb = (int(v) for v in D['rb_cb'][i])
                SB[si, n], NC[si, n], NCELL[si, n] = frame_scores(aux[b], body[b], rb, cb, Kc, np.asarray(R[n]['pred'][0], np.float64), in_wh[0] // aux.shape[-1])
            if c0 % 800 == 0:
                print('  种子 %d：%d / %d 帧 %.0f s' % (seed, c0 + len(ch), len(idx), time.time() - t0), flush=True)
    np.savez(B + '/arb_%s.npz' % split, SB=SB, NC=NC, NCELL=NCELL, seeds=np.array(seeds))
    print('%s：%d 帧，有效格 ≥3 的帧 %.1f%%' % (split, N, 100 * np.mean(NCELL >= 3)), flush=True)


def quads(R):
    """每帧 4 象限到真值的角度 (N,4)"""
    out = np.full((len(R), 4), np.nan)
    for n, r in enumerate(R):
        if len(r['pred']) == 0:
            continue
        S = Rot.from_euler('xyz', r['pred'][0, 6:9]).as_matrix()
        Gm = Rot.from_euler('xyz', r['gt'][0, 6:9]).as_matrix()
        out[n] = [np.degrees(np.arccos(np.clip((np.trace((S @ RZ[k]).T @ Gm) - 1) / 2, -1, 1))) for k in range(4)]
    return out


def choice(split, rule):
    Z = np.load(B + '/arb_%s.npz' % split)
    if rule == 'body':
        s = np.nanmean(Z['SB'], 0)
        ok = np.isfinite(s).all(1)
        q = np.where(ok, np.argmin(np.where(np.isfinite(s), s, np.inf), 1), -1)
    else:
        s = np.nanmean(Z['NC'], 0)
        ok = np.isfinite(s).all(1)
        q = np.where(ok, np.argmax(np.where(np.isfinite(s), s, -np.inf), 1), -1)
    return q


def evaluate(split, configs=None, write_md=True):
    import temporal_filter as T
    F = T.F
    R = load_recs(split)
    A = quads(R)
    rel = np.isfinite(A).all(1) & (np.nanmin(A, 1) < 45)
    qgt = np.where(rel, np.nanargmin(np.where(np.isfinite(A), A, np.inf), 1), -1)
    sc = np.array(['/'.join(r['seq_id'].split('/')[:2]) for r in R])
    res = {'n_rel': int(rel.sum()), 'acc': {}, 'ads': {}}
    acc_cur = {s: float(np.mean(qgt[rel & (sc == s)] == 0)) for s in sorted(set(sc))}
    acc_cur['全部'] = float(np.mean(qgt[rel] == 0))
    res['acc']['现最终版逐帧'] = acc_cur
    for rule in ('body', 'nose'):
        q = choice(split, rule)
        m = rel & (q >= 0)
        a = {s: float(np.mean(q[m & (sc == s)] == qgt[m & (sc == s)])) for s in sorted(set(sc))}
        a['全部'] = float(np.mean(q[m] == qgt[m]))
        a['投票覆盖'] = float(np.mean(q[rel] >= 0))
        a['现最终版错的帧里裁判对'] = float(np.mean(q[m & (qgt != 0)] == qgt[m & (qgt != 0)])) if (m & (qgt != 0)).any() else float('nan')
        a['现最终版对的帧里裁判不改'] = float(np.mean(q[m & (qgt == 0)] == 0)) if (m & (qgt == 0)).any() else float('nan')
        res['acc'][rule] = a
        print('裁判 %s：准确率 %s' % (rule, json.dumps({k: round(v, 3) for k, v in a.items()}, ensure_ascii=False)), flush=True)
    print('现最终版逐帧象限准确率：%s' % json.dumps({k: round(v, 3) for k, v in acc_cur.items()}, ensure_ascii=False), flush=True)

    def run(rule, beta, mode):
        q = choice(split, rule) if rule else None
        new = []
        for n, r in enumerate(R):
            r = dict(r)
            if q is not None and q[n] >= 0:
                v = np.asarray(r['votes'], np.float64).copy()
                v[q[n]] += beta
                r['votes'] = v
            new.append(r)
        out = T.do(new, mode)
        s = F.summ(out, 'kzbody_%s_%s_%s_%s' % (split, rule, beta, mode[:8]), ('indoor', 'laa'))
        _, ang, _ = T.errs(out)
        s['flip90'] = float(np.nanmean(ang > 90))
        s['flip90_scene'] = {x: float(np.nanmean(ang[sc == x] > 90)) for x in sorted(set(sc))}
        return s
    res['ads']['现最终版 · 在线一致'] = run(None, 0, ONLINE)
    res['ads']['现最终版 · 离线一致'] = run(None, 0, OFFLINE)
    cfgs = configs or [(r_, b_) for r_ in ('body', 'nose') for b_ in (1, 2, 5)]
    for rule, beta in cfgs:
        for mode, nmode in ((ONLINE, '在线一致'), (OFFLINE, '离线一致')):
            k = '裁判 %s β=%s · %s' % (rule, beta, nmode)
            res['ads'][k] = run(rule, beta, mode)
            x = res['ads'][k]
            print('%s：ADS indoor %.2f laa %.2f 角度 %.1f° 反向 %.1f%%' % (k, x['ads_indoor'], x['ads_laa'], x['ang'], 100 * x['flip90']), flush=True)
    if write_md:
        json.dump(res, open(B + '/eval_%s.json' % split, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    scs = sorted(set(sc))
    L = ['', '## 朝向前后裁判 KZ-Body（PREREG_kzbody 61cb4b2a）· 真实%s · 种子 %s' % ({'val': '验证', 'test': '测试'}[split], list(np.load(B + '/arb_%s.npz' % split)['seeds'])), '',
         '前后/象限准确率（只看离真值 < 45° 的象限存在的帧，%d 帧）：现最终版逐帧 %.1f%%；机体图裁判 %.1f%%（现最终版错的帧里改对 %.0f%%，对的帧里不改 %.1f%%）；机头裁判 %.1f%%（%.0f%% / %.1f%%）' % (
             res['n_rel'], 100 * acc_cur['全部'], 100 * res['acc']['body']['全部'], 100 * res['acc']['body']['现最终版错的帧里裁判对'], 100 * res['acc']['body']['现最终版对的帧里裁判不改'],
             100 * res['acc']['nose']['全部'], 100 * res['acc']['nose']['现最终版错的帧里裁判对'], 100 * res['acc']['nose']['现最终版对的帧里裁判不改']), '',
         '| 配置 | 检出 | 位置中位 | 角度中位 | ADS indoor | ADS laa | 机头反向(>90°) | %s |' % ' | '.join(scs), '|---|---|---|---|---|---|---|' + '---|' * len(scs)]
    for k, x in res['ads'].items():
        L.append('| %s | %.1f%% | %.3f m | %.1f° | %.2f | %.2f | %.1f%% | %s |' % (k, x['det'], x['pos'], x['ang'], x['ads_indoor'], x['ads_laa'], 100 * x['flip90'],
                                                                         ' | '.join('%.0f%%' % (100 * x['flip90_scene'][s]) for s in scs)))
    open(G + '/FINAL_ADS.md', 'a', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    return res


if __name__ == '__main__':
    os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
    if sys.argv[1] == 'adabn':
        adabn(int(sys.argv[2]))
    elif sys.argv[1] == 'maps':
        maps(sys.argv[2], [int(x) for x in sys.argv[3].split(',')])
    else:
        evaluate(sys.argv[2], [(sys.argv[3], float(sys.argv[4]))] if len(sys.argv) > 4 else None)
