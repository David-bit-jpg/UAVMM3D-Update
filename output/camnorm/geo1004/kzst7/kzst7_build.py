# -*- coding: utf-8 -*-
"""KZ-ST7 伪标签（prereg_kzst7.json）：在 KZ-ST6 伪标签上只改两处，分两臂写数据目录。
老师与 KZ-ST6 逐项相同（重算一遍并核对 R_T / 置信与 kzst6/data/pseudo_stage2.npz 一致），另外记下每个成员的单帧朝向，算"老师分歧"。
    臂 Q（data_q）：所有帧 quotient = True（前后不教）
    臂 W（data_w）：每个学生去掉老师分歧最大的 30% 帧（分歧 = 其他 4 成员单帧朝向到老师的角距离中位，取 4 个 90° 象限最小）
裁窗与 KZ-ST6 是同一批帧、同一窗口 → json 里直接写 kzst6/data/crops_train_u8.npy 的绝对路径（E 盘不支持硬链接，不复制 1.9 GB）。全程不读真值。
    python kzst7_build.py"""
import collections, json, os, sys, time
import numpy as np
G = '@@ROOT@@/output/camnorm/geo1004'
sys.path.insert(0, G + '/kzst6')
import kzst6_build as B6        # noqa: E402
S, RC, C1, KFO = B6.S, B6.RC, B6.C1, B6.KFO
OUT = G + '/kzst7'
PREREG = OUT + '/prereg_kzst7.json'
RZ, ang = B6.RZ, B6.ang
DROP = 0.30


def main():
    pre = S.sha256(PREREG)
    assert os.environ.get('UAV_STRICT_HASH') != '1' or pre == open(PREREG + '.sha256').read().split()[0], 'prereg modified'  # [release]
    P = RC.tag_paths('st2p_train_kzma')
    D, info = C1.load_npz_info(P['npz'])
    frames, _ = RC.load_frames(P['pkl'], with_gt=False)          # 不读真值
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    N, M = len(keys), 5
    search = int(info.get('search', 2))
    ks_ = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    seq = [k[0] for k in keys]
    frm = [k[1] for k in keys]
    TR = B6.tracks(seq, frm)
    todo = sorted(i for t in TR for i in t)
    t0 = time.time()
    loo = {}
    for n, i in enumerate(todo):
        r = C1.frame_arms(D, i, frames[keys[i]], list(range(M)), search, ks_, loo=True)
        if r is None:
            loo[i] = None
            continue
        votes = np.zeros((M, 4))
        for k in range(M):
            Rk = r['loo_RT'][k]
            for j, Rm in enumerate(r['member_R']):
                if j == k or Rm is None or Rk is None or not np.isfinite(np.asarray(Rm)).all():
                    continue
                votes[k, int(np.argmin([np.linalg.norm(Rk @ RZ[q] - np.asarray(Rm)) for q in range(4)]))] += 1
        fin = lambda x: None if x is None or not np.isfinite(np.asarray(x)).all() else np.asarray(x, np.float64)
        loo[i] = dict(RT=[fin(x) for x in r['loo_RT']], votes=votes, member=[fin(x) for x in r['member_R']])
        if n % 2000 == 1999:
            print('   LOO 老师 %d / %d, %.0f s' % (n + 1, len(todo), time.time() - t0), flush=True)
    RT = np.full((N, M, 3, 3), np.nan)
    CONF = np.zeros((N, M), bool)
    SPR = np.full((N, M), np.nan)
    for tr in TR:
        for k in range(M):
            Sl = [loo[i]['RT'][k] if loo.get(i) is not None else None for i in tr]
            V = np.stack([loo[i]['votes'][k] if loo.get(i) is not None else np.zeros(4) for i in tr])
            q = B6.hmm_offline(Sl, V)
            Rc = [Sl[t] @ RZ[q[t]] if q[t] >= 0 else None for t in range(len(tr))]
            jump = np.array([ang(Rc[t], Rc[t + 1]) if (Rc[t] is not None and Rc[t + 1] is not None) else 0.0 for t in range(len(tr) - 1)] + [0.0])
            for t, i in enumerate(tr):
                if Rc[t] is None:
                    continue
                w = [j for j in range(max(0, t - B6.W // 2), min(len(tr), t + B6.W // 2 + 1)) if Rc[j] is not None]
                RT[i, k] = B6.rot_avg([Rc[j] for j in w]) if len(w) >= 3 else Rc[t]
                lo, hi = max(0, t - 7), min(len(tr) - 1, t + 7)
                CONF[i, k] = bool(np.all(jump[lo:hi] < 45.0)) and bool(V[t, q[t]] >= 4)
                d = [min(ang(Rc[t], Rm @ RZ[qq]) for qq in range(4)) for j, Rm in enumerate(loo[i]['member']) if j != k and Rm is not None]
                if len(d) >= 2:
                    SPR[i, k] = float(np.median(d))
    Z6 = dict(np.load(G + '/kzst6/data/pseudo_stage2.npz', allow_pickle=False))
    sel = Z6['src_idx']
    sel_new = np.array(sorted(i for tr in TR for t, i in enumerate(tr) if t % B6.STEP == 0 and loo.get(i) is not None), np.int64)
    assert np.array_equal(sel, sel_new), '选帧与 KZ-ST6 不一致'
    a, b = RT[sel], Z6['R_T']
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9), '老师与 KZ-ST6 不一致'
    assert np.array_equal(CONF[sel], Z6['confident']), '置信与 KZ-ST6 不一致'
    J6 = json.load(open(G + '/kzst6/data/pseudo_stage2.json', encoding='utf-8'))
    spr = SPR[sel]
    rep_all = {}
    for arm in ('q', 'w'):
        d = OUT + '/data_' + arm
        os.makedirs(d, exist_ok=True)
        Z = {k: v.copy() for k, v in Z6.items()}
        if arm == 'q':
            Z['quotient'] = np.ones_like(Z6['quotient'])
            rep = {'quotient_frac': 1.0}
        else:
            thr = []
            for k in range(M):
                v = Z['valid'][:, k]
                s = np.where(np.isfinite(spr[:, k]), spr[:, k], np.inf)
                t_ = float(np.quantile(s[v & np.isfinite(s)], 1 - DROP))
                Z['valid'][:, k] = v & (s <= t_)
                thr.append(t_)
            rep = {'spread_thr_deg': thr, 'spread_med_deg': [float(np.nanmedian(spr[:, k])) for k in range(M)],
                   'valid_per_student': [int(Z['valid'][:, k].sum()) for k in range(M)],
                   'confident_frac_kept': [float(Z['confident'][Z['valid'][:, k], k].mean()) for k in range(M)],
                   'kept_frac_by_cls': {c: float(Z['valid'][Z['cls'] == c].mean()) for c in sorted(set(Z['cls'].tolist()))}}
        Z['spread_deg'] = spr
        np.savez(d + '/pseudo_stage2.npz', **Z)
        rep['iters_3ep_b8'] = [int(3 * np.ceil(Z['valid'][:, k].sum() / 8.0)) for k in range(M)]
        J = {'prereg_sha256': pre, 'npz': 'pseudo_stage2.npz', 'npz_sha256': S.sha256(d + '/pseudo_stage2.npz'),
             'crops': G + '/kzst6/data/crops_train_u8.npy', 'crops_sha256': J6['crops_sha256'], 'smoke': False, 'source_tag': 'st2p_train_kzma',
             'arm': arm, 'base': 'kzst6/data（老师核对一致）', 'report_label_free': rep}
        json.dump(J, open(d + '/pseudo_stage2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        rep_all[arm] = rep
    print('[kzst7 build] done %.0f s' % (time.time() - t0), json.dumps(rep_all, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
