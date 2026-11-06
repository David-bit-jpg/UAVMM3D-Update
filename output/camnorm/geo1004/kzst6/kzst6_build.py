# -*- coding: utf-8 -*-
"""KZ-ST6 = KZ-ST4 配方，起点换成 KZ-IRLiD（IR + LiDAR 辅助答案）+ AdaBN 成员（PREREG_irlid B）。
KZ-ST4（10-12，用户：「按你说的做下一轮，只在高置信帧教前后」）= KZ-ST3 + 只在高置信帧教前后：
置信 = 其他 4 个成员都投老师选的象限 且 前后 7 帧内老师朝向相邻帧变化都 < 45°（无跳变）；其余帧 quotient = 前后对称损失（min(|p−g|, |p−P180 g|)）。
以下为 KZ-ST3 原说明。
KZ-ST3（10-12，用户：「位置用真实数据自训练就很好，朝向也试试同样的自训练」）——伪标签 + 训练裁窗，全程不读真值。

和 10-03 的 KZ-ST2（LOO 逐帧集成老师，结果≈AdaBN）的区别：老师 = 相邻帧一致后的朝向（离线 HMM 选机头象限 γ=30 + 居中 15 帧鲁棒平均，
PREREG_temporal 验证选出的离线组合），把"时间上的信息"教给单帧网络；起点 = AdaBN KZ；位置 = 去偏自训练检测器 ST2P。
学生 k 的老师只用其他 4 个成员（LOO：SA-TM 用 4 个、象限投票也只用 4 个），保持多样性。
每段轨迹全帧算老师，训练取每 3 帧 1 帧。目标 = 刚体框（t = 位置网中心，lwh = 部署 PnP 框，R = 老师）在部署裁窗里的投影角点，rho = (1,1)。
输出 → G/kzst3/data：pseudo_stage2.npz、crops_train_u8.npy、pseudo_stage2.json（最后写 = 成功标记，格式同 kzst2_train 要求）。
    python kzst3_build.py --tag st2p_train_adabn [--max-frames N --out-dir ...]"""
import argparse, collections, concurrent.futures as cf, json, os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst2', SCR + '/kzst1', SCR + '/repl', '@@ROOT@@'):
    sys.path.insert(0, p)
import kzst2_common as S        # noqa: E402
import repl_common as RC        # noqa: E402
C1, KFO = S.C1, S.KFO
G = '@@ROOT@@/output/camnorm/geo1004'
OUT = G + '/kzst6'
GAMMA, W, EPS, STEP = 30.0, 15, 0.5, 3
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]


def ang(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


def tracks(seq, frame):
    by = collections.defaultdict(list)
    for i, (s, f) in enumerate(zip(seq, frame)):
        by[s].append(i)
    out = []
    for s, ii in by.items():
        num = lambda i: int(os.path.splitext(frame[i])[0])
        ii.sort(key=num)
        cur = [ii[0]]
        for a, b in zip(ii[:-1], ii[1:]):
            d = num(b) - num(a)
            if (d > 0.2e9) if num(a) > 1e12 else (d > 2):
                out.append(cur)
                cur = []
            cur.append(b)
        out.append(cur)
    return out


def hmm_offline(S_list, V):
    """S_list: [R]（None = 无），V: (n,4) 票数 → 每帧象限（无老师的帧 -1）"""
    idx = [t for t, R in enumerate(S_list) if R is not None]
    ks = np.full(len(S_list), -1)
    if not idx:
        return ks
    n = len(idx)
    E = np.array([-np.log((V[t] + EPS) / (V[t].sum() + 4 * EPS)) for t in idx])
    al = np.zeros((n, 4))
    bp = np.zeros((n, 4), int)
    al[0] = E[0]
    for m in range(1, n):
        A, B = S_list[idx[m - 1]], S_list[idx[m]]
        Tc = np.array([[GAMMA * ang(A @ RZ[a], B @ RZ[b]) / 90.0 for b in range(4)] for a in range(4)])
        c = al[m - 1][:, None] + Tc
        bp[m] = np.argmin(c, 0)
        al[m] = E[m] + c[bp[m], range(4)]
    q = np.zeros(n, int)
    q[-1] = int(np.argmin(al[-1]))
    for m in range(n - 1, 0, -1):
        q[m - 1] = bp[m, q[m]]
    ks[idx] = q
    return ks


def rot_avg(Rs, inl=30.0):
    Q = Rot.from_matrix(np.stack(Rs))
    D = np.stack([(Q[a].inv() * Q).magnitude() for a in range(len(Rs))])
    med = int(np.argmin(D.sum(1)))
    keep = D[med] <= np.radians(inl)
    q = Q[keep].as_quat()
    q = q * np.sign(q @ Q[med].as_quat())[:, None]
    m = q.mean(0)
    return Rot.from_quat(m / np.linalg.norm(m)).as_matrix()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='st2p_train_adabn')
    ap.add_argument('--out-dir', default=OUT + '/data')
    ap.add_argument('--prereg', default=OUT + '/prereg_kzst6.json')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pre = S.sha256(a.prereg)
    assert os.environ.get('UAV_STRICT_HASH') != '1' or pre == open(a.prereg + '.sha256').read().split()[0], 'prereg modified'  # [release]
    P = RC.tag_paths(a.tag)
    D, info = C1.load_npz_info(P['npz'])
    frames, _ = RC.load_frames(P['pkl'], with_gt=False)          # 不读真值
    keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
    N = len(keys)
    search = int(info.get('search', 2))
    ks_ = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    M = 5
    seq = [k[0] for k in keys]
    frm = [k[1] for k in keys]
    TR = tracks(seq, frm)
    if a.max_frames:                                            # 冒烟：只取前几段轨迹
        keep, n = [], 0
        for t in TR:
            if n >= a.max_frames:
                break
            keep.append(t[:a.max_frames - n])
            n += len(keep[-1])
        TR = keep
    todo = sorted(i for t in TR for i in t)
    t0 = time.time()
    loo = {}

    def one(i):
        fr = frames[keys[i]]
        r = C1.frame_arms(D, i, fr, list(range(M)), search, ks_, loo=True)
        if r is None:
            return i, None
        # 学生 k：老师 = 其他 4 个成员的 SA-TM；象限票 = 其他 4 个成员
        votes = np.zeros((M, 4))
        for k in range(M):
            Rk = r['loo_RT'][k]
            for j, Rm in enumerate(r['member_R']):
                if j == k or Rm is None or Rk is None or not np.isfinite(np.asarray(Rm)).all():
                    continue
                votes[k, int(np.argmin([np.linalg.norm(Rk @ RZ[q] - np.asarray(Rm)) for q in range(4)]))] += 1
        return i, dict(RT=[None if x is None or not np.isfinite(np.asarray(x)).all() else np.asarray(x, np.float64) for x in r['loo_RT']],
                       votes=votes, top=int(r['top']), g=r['ctx'].g)
    with cf.ThreadPoolExecutor(max_workers=1) as ex:            # frame_arms 不保证线程安全，顺序算
        for n, (i, v) in enumerate(ex.map(one, todo)):
            loo[i] = v
            if n % 1000 == 999:
                print('   LOO 老师 %d / %d, %.0f s' % (n + 1, len(todo), time.time() - t0), flush=True)
    # 每个学生、每段轨迹：离线 HMM 选象限 + 居中 15 帧鲁棒平均
    RT = np.full((N, M, 3, 3), np.nan)
    CONF = np.zeros((N, M), bool)
    q_hist = collections.Counter()
    for tr in TR:
        for k in range(M):
            Sl = [loo[i]['RT'][k] if loo.get(i) is not None else None for i in tr]
            V = np.stack([loo[i]['votes'][k] if loo.get(i) is not None else np.zeros(4) for i in tr])
            q = hmm_offline(Sl, V)
            Rc = [Sl[t] @ RZ[q[t]] if q[t] >= 0 else None for t in range(len(tr))]
            jump = np.array([ang(Rc[t], Rc[t + 1]) if (Rc[t] is not None and Rc[t + 1] is not None) else 0.0 for t in range(len(tr) - 1)] + [0.0])
            for t, i in enumerate(tr):
                if Rc[t] is None:
                    continue
                q_hist[int(q[t])] += 1
                w = [j for j in range(max(0, t - W // 2), min(len(tr), t + W // 2 + 1)) if Rc[j] is not None]
                RT[i, k] = rot_avg([Rc[j] for j in w]) if len(w) >= 3 else Rc[t]
                lo, hi = max(0, t - 7), min(len(tr) - 1, t + 7)
                stable = bool(np.all(jump[lo:hi] < 45.0))
                agree = bool(V[t, q[t]] >= 4)
                CONF[i, k] = stable and agree
    sel = np.array(sorted(i for tr in TR for t, i in enumerate(tr) if t % STEP == 0 and loo.get(i) is not None), np.int64)
    n = len(sel)
    out = {'src_idx': sel, 'seq': np.array([seq[i] for i in sel]), 'frame': np.array([frm[i] for i in sel]),
           'cls': np.array([str(frames[keys[i]]['cls']) for i in sel]), 'crop': D['crop'][sel].astype(np.float64),
           'z': D['z'][sel].astype(np.float64), 'K_crop': np.zeros((n, 3, 3)), 't': np.zeros((n, 3)), 'lwh': np.zeros((n, 3)),
           'rho': np.ones((n, 2)), 'R_T': RT[sel], 'quotient': ~CONF[sel], 'confident': CONF[sel], 'valid': np.zeros((n, M), bool),
           'target16': np.full((n, M, 16), np.nan), 'rt_deg': np.full((n, M), np.nan)}
    for m, i in enumerate(sel):
        fr = frames[keys[i]]
        top = loo[i]['top']
        t = fr['pred'][top, :3].astype(np.float64)
        lwh = fr['pred'][top, 3:6].astype(np.float64)
        x0, y0, w, h, s = [float(v) for v in D['crop'][i]]
        Kc = S.crop_K(D['K_raw_src'][i], x0, y0, s)
        uvc = S.proj(Kc, t)[0]
        inside = bool(0 <= uvc[0] < S.IN_WH[0] and 0 <= uvc[1] < S.IN_WH[1])
        out['K_crop'][m], out['t'][m], out['lwh'][m] = Kc, t, lwh
        for k in range(M):
            R_ = RT[i, k]
            if not np.isfinite(R_).all():
                continue
            tc = S.g16(Kc, t, lwh, R_)
            out['target16'][m, k] = tc
            Rd = S.decode_crop(tc, (1.0, 1.0), Kc, t, lwh)
            out['rt_deg'][m, k] = KFO.geodesic(Rd, R_) if Rd is not None else np.inf
            out['valid'][m, k] = inside and out['rt_deg'][m, k] < S.MAX_ROUNDTRIP_DEG
    np.savez(a.out_dir + '/pseudo_stage2.npz', **out)
    # 裁窗：部署时 KZ 看到的同一窗口，从原图重切
    import adabn_kz as AB
    src = AB.NpzSource(D)
    crops_p = a.out_dir + '/crops_train_u8.npy'
    arr = np.lib.format.open_memmap(crops_p + '.tmp.npy', mode='w+', dtype=np.uint8, shape=(n, S.IN_WH[1], S.IN_WH[0], 3))

    def cut(m):
        arr[m] = AB.recut(D, int(sel[m]), src, S.IN_WH)
    with cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        list(ex.map(cut, range(n)))
    arr.flush()
    del arr
    os.replace(crops_p + '.tmp.npy', crops_p)
    rep = {'n_track_frames': len(todo), 'n_tracks': len(TR), 'n_selected': n, 'valid_per_student': [int(out['valid'][:, k].sum()) for k in range(M)],
           'per_class': dict(collections.Counter(out['cls'].tolist())), 'quadrant_hist_all_students': dict(q_hist),
           'roundtrip_max_deg': float(np.nanmax(out['rt_deg'][np.isfinite(out['rt_deg'])])) if np.isfinite(out['rt_deg']).any() else None,
           'confident_frac_per_student': [float(CONF[sel][:, k].mean()) for k in range(M)],
           'confident_frac_by_cls': {c: float(CONF[sel][out['cls'] == c].mean()) for c in sorted(set(out['cls'].tolist()))},
           'iters_3ep_b8': [int(3 * np.ceil(out['valid'][:, k].sum() / 8.0)) for k in range(M)], 'seconds': time.time() - t0}
    J = {'prereg_sha256': pre, 'npz': 'pseudo_stage2.npz', 'npz_sha256': S.sha256(a.out_dir + '/pseudo_stage2.npz'),
         'crops': 'crops_train_u8.npy', 'crops_sha256': S.sha256(crops_p), 'smoke': bool(a.max_frames), 'source_tag': a.tag, 'report_label_free': rep}
    json.dump(J, open(a.out_dir + '/pseudo_stage2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[kzst3 build] done', json.dumps(rep, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
