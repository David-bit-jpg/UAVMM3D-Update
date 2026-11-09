# -*- coding: utf-8 -*-
"""10-12 用户：「最差帧那种距离错怎么解决；角度差得多；滤波一下平滑一点」。
在最终版（含 AdaBN 朝向）逐帧结果上做时间滤波：
  位置：对数深度的 Hampel 式离群替换（窗口中位，偏离超过阈值就沿视线拉回中位深度；投影中心不变）
  朝向：窗口内鲁棒旋转平均（先找窗口里离其他帧总角距最小的那帧，再对 30° 以内的帧做四元数平均）
窗口按飞行片段、按帧时间戳排序，时间间隔 > 0.2 s 断开。mode = causal（只用过去帧，在线）或 centered（固定延迟 (k-1)/2 帧）。
python temporal_filter.py diag   → 验证集误差结构（只看验证）
python temporal_filter.py select → 验证集网格（PREREG_temporal.md）→ temporal_select.json
python temporal_filter.py final  → 测试一次 → FINAL_ADS.md 追加"""
import json, os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, G + '/finalads')
import final_eval_ads2 as F      # noqa: E402
RC, C, KFO = F.RC, F.C, F.KFO
O = G + '/finalads'
SEQ = 'xyz'
TAU, ALPHA = 0.1, 0.5
GAP_NS = 0.2e9


def build(split):
    """最终版逐帧记录（ADS 用的 recs 格式）+ 帧次序信息"""
    Pk, Pb = RC.tag_paths('st2p_%s_kz' % split), RC.tag_paths('st2p_%s_adabn' % split)
    Dk, ik = C.load_npz_info(Pk['npz'])
    Db, ib = C.load_npz_info(Pb['npz'])
    frames, _ = RC.load_frames(Pk['pkl'], with_gt=False)
    keys, A = RC.compute_arms(Dk, ik, frames)
    keys_b, B = RC.compute_arms(Db, ib, frames, members=list(range(5)))
    assert keys_b == keys and np.array_equal(A['top'], B['top'])
    frames_gt, _ = RC.load_frames(Pk['pkl'], Pk['sealed'], with_gt=True)
    Fr = KFO.build_arm_frames(frames_gt, keys, B['SATM'], A['top'])
    votes = member_votes(Db, ib, frames, keys, B['SATM'])
    recs = [{'gt': np.asarray(f['gt'], np.float64).reshape(-1, 9), 'pred': np.asarray(f['pred'], np.float64).reshape(-1, 9),
             'conf': np.asarray(f['conf'], np.float64), 'K': f['K_in'], 'seq_id': f['seq'], 'frame_id': f['frame']} for f in Fr]
    recs = F.apply(recs, F.factors(split), TAU, alpha=ALPHA)
    for r, v in zip(recs, votes):
        r['votes'] = v
    return recs


RZ = [Rot.from_euler('z', 90 * k, degrees=True).as_matrix() for k in range(4)]


def member_votes(D, info, frames, keys, S, members=None):
    """每帧 4 个象限（融合朝向绕机体竖轴转 0/90/180/270°）各得几票：5 个成员各投离自己最近的象限"""
    search = int(info.get('search', 2))
    ks = float(info.get('KP_SCALE') or KFO.KP_SCALE_DEFAULT)
    out = np.zeros((len(keys), 4))
    for i, k in enumerate(keys):
        if not np.isfinite(S[i]).all():
            continue
        a = C.frame_arms(D, i, frames[k], list(range(5)) if members is None else list(members), search, ks)   # 11-09：可指定成员（缺省 = 前 5 个，行为不变）
        if a is None:
            continue
        for Rm in a['member_R']:
            if Rm is None or not np.isfinite(np.asarray(Rm)).all():
                continue
            d = [np.linalg.norm(S[i] @ RZ[q] - np.asarray(Rm)) for q in range(4)]
            out[i, int(np.argmin(d))] += 1
    return out


def ang_deg(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


def filt_hmm(recs, gamma, mode, w, eps=0.5):
    """机头象限相邻帧一致（PREREG_temporal A）+ 窗口鲁棒旋转平均"""
    out = [dict(r) for r in recs]
    for tr in tracks(recs):
        S = [Rot.from_euler(SEQ, recs[i]['pred'][0, 6:9]).as_matrix() for i in tr]
        n = len(tr)
        E = np.array([-np.log((recs[i]['votes'] + eps) / (recs[i]['votes'].sum() + 4 * eps)) for i in tr])
        Tc = np.zeros((max(n - 1, 0), 4, 4))
        for t in range(n - 1):
            for a in range(4):
                for b in range(4):
                    Tc[t, a, b] = gamma * ang_deg(S[t] @ RZ[a], S[t + 1] @ RZ[b]) / 90.0
        al = np.zeros((n, 4))
        bp = np.zeros((n, 4), int)
        al[0] = E[0]
        for t in range(1, n):
            c = al[t - 1][:, None] + Tc[t - 1]
            bp[t] = np.argmin(c, 0)
            al[t] = E[t] + c[bp[t], range(4)]
        if mode == 'causal':
            ks = np.argmin(al, 1)
        else:
            ks = np.zeros(n, int)
            ks[-1] = int(np.argmin(al[-1]))
            for t in range(n - 1, 0, -1):
                ks[t - 1] = bp[t, ks[t]]
        Rc = [S[t] @ RZ[ks[t]] for t in range(n)]
        for t, i in enumerate(tr):
            Rf = Rc[t]
            if w > 1:
                ww = win(n, t, w, 'causal' if mode == 'causal' else 'centered')
                if len(ww) >= 3:
                    Rf = rot_avg(np.stack([Rc[j] for j in ww]))
            p = out[i]['pred'].copy()
            p[0, 6:9] = Rot.from_matrix(Rf).as_euler(SEQ)
            out[i]['pred'] = p
    return out


def tracks(recs):
    """[[记录下标, ...], ...]：按片段、按时间戳排序，间隔 > 0.2 s 断开"""
    by = {}
    for i, r in enumerate(recs):
        by.setdefault(r['seq_id'], []).append(i)
    out = []
    for s, ii in by.items():
        ii.sort(key=lambda i: int(os.path.splitext(recs[i]['frame_id'])[0]))
        cur = [ii[0]]
        for a, b in zip(ii[:-1], ii[1:]):
            if int(os.path.splitext(recs[b]['frame_id'])[0]) - int(os.path.splitext(recs[a]['frame_id'])[0]) > GAP_NS:
                out.append(cur)
                cur = []
            cur.append(b)
        out.append(cur)
    return out


def win(n, t, k, mode):
    if mode == 'causal':
        return list(range(max(0, t - k + 1), t + 1))
    h = (k - 1) // 2
    return list(range(max(0, t - h), min(n, t + h + 1)))


def filt_pos(recs, k, T, mode, trunc=False, W=512.0, H=288.0, margin=2.0):
    out = [dict(r) for r in recs]
    for tr in tracks(recs):
        lz = np.array([np.log(recs[i]['pred'][0, 2]) if len(recs[i]['pred']) else np.nan for i in tr])
        for t, i in enumerate(tr):
            if not np.isfinite(lz[t]):
                continue
            w = [j for j in win(len(tr), t, k, mode) if j != t or mode == 'centered']
            v = lz[w]
            v = v[np.isfinite(v)]
            if len(v) < 3:
                continue
            m = float(np.median(v))
            bad = abs(lz[t] - m) > T
            if trunc and not bad:
                p = recs[i]['pred'][0]
                c = (F.pe.PROTO8 * p[3:6]) @ Rot.from_euler(SEQ, p[6:9]).as_matrix().T + p[:3]
                q = c @ recs[i]['K'].T
                uv = q[:, :2] / q[:, 2:3]
                bad = uv[:, 0].min() < margin or uv[:, 1].min() < margin or uv[:, 0].max() > W - margin or uv[:, 1].max() > H - margin
            if bad:
                f = np.exp(m - lz[t])
                p = out[i]['pred'].copy()
                p[0, :3] *= f
                p[0, 3:6] *= f ** ALPHA
                out[i]['pred'] = p
    return out


def rot_avg(Rs, inl_deg=30.0):
    Q = Rot.from_matrix(Rs)
    n = len(Rs)
    D = np.zeros((n, n))
    for a in range(n):
        D[a] = (Q[a].inv() * Q).magnitude()
    med = int(np.argmin(D.sum(1)))
    keep = D[med] <= np.radians(inl_deg)
    q = Q[keep].as_quat()
    q0 = Q[med].as_quat()
    q = q * np.sign(q @ q0)[:, None]
    m = q.mean(0)
    return Rot.from_quat(m / np.linalg.norm(m)).as_matrix()


def filt_rot(recs, k, mode):
    out = [dict(r) for r in recs]
    for tr in tracks(recs):
        Rs = [Rot.from_euler(SEQ, recs[i]['pred'][0, 6:9]).as_matrix() if len(recs[i]['pred']) else None for i in tr]
        for t, i in enumerate(tr):
            if Rs[t] is None:
                continue
            w = [j for j in win(len(tr), t, k, mode) if Rs[j] is not None]
            if len(w) < 3:
                continue
            p = out[i]['pred'].copy()
            p[0, 6:9] = Rot.from_matrix(rot_avg(np.stack([Rs[j] for j in w]))).as_euler(SEQ)
            out[i]['pred'] = p
    return out


def errs(recs):
    pe, an, dz = [], [], []
    for r in recs:
        if len(r['pred']) and len(r['gt']):
            p, g = r['pred'][0], r['gt'][0]
            pe.append(np.linalg.norm(p[:3] - g[:3]))
            dz.append(np.log(p[2] / g[2]))
            an.append(np.degrees((Rot.from_euler(SEQ, p[6:9]).inv() * Rot.from_euler(SEQ, g[6:9])).magnitude()))
        else:
            pe.append(np.nan); an.append(np.nan); dz.append(np.nan)
    return np.array(pe), np.array(an), np.array(dz)


def diag():
    recs = build('val')
    pickle.dump(recs, open(O + '/final_val_recs_v.pkl', 'wb'))
    pe, an, dz = errs(recs)
    TR = tracks(recs)
    print('验证 %d 帧，%d 段连续轨迹，段长中位 %d' % (len(recs), len(TR), int(np.median([len(t) for t in TR]))))
    ts = np.concatenate([np.diff([int(os.path.splitext(recs[i]['frame_id'])[0]) for i in t]) for t in TR if len(t) > 1]) / 1e9
    print('相邻帧间隔中位 %.3f s（p95 %.3f）' % (np.median(ts), np.percentile(ts, 95)))
    print('位置误差中位 %.3f m；>0.5 m %.1f%%；>1 m %.1f%%' % (np.nanmedian(pe), 100 * np.nanmean(pe > 0.5), 100 * np.nanmean(pe > 1.0)))
    big = pe > 0.5
    print('  >0.5 m 的帧里深度错占主导（|对数深度误差|>0.15）：%.0f%%' % (100 * np.mean(np.abs(dz[big]) > 0.15)))
    # 连续段长度
    runs = []
    for t in TR:
        b = [bool(big[i]) for i in t]
        n = 0
        for x in b + [False]:
            if x:
                n += 1
            elif n:
                runs.append(n)
                n = 0
    runs = np.array(runs)
    print('  >0.5 m 连续段：%d 段；长度 1–3 帧的段 %.0f%%，这些段占坏帧 %.0f%%；≥24 帧（1 秒）的段占坏帧 %.0f%%' % (
        len(runs), 100 * np.mean(runs <= 3), 100 * runs[runs <= 3].sum() / runs.sum(), 100 * runs[runs >= 24].sum() / runs.sum()))
    print('角度误差中位 %.1f°；>90° %.1f%%；>150° %.1f%%；30–90° %.1f%%' % (np.nanmedian(an), 100 * np.nanmean(an > 90), 100 * np.nanmean(an > 150), 100 * np.nanmean((an > 30) & (an <= 90))))
    # 相邻帧预测转角 vs 真值转角
    dp, dg = [], []
    for t in TR:
        for a, b in zip(t[:-1], t[1:]):
            ra, rb = recs[a], recs[b]
            dp.append(np.degrees((Rot.from_euler(SEQ, ra['pred'][0, 6:9]).inv() * Rot.from_euler(SEQ, rb['pred'][0, 6:9])).magnitude()))
            dg.append(np.degrees((Rot.from_euler(SEQ, ra['gt'][0, 6:9]).inv() * Rot.from_euler(SEQ, rb['gt'][0, 6:9])).magnitude()))
    dp, dg = np.array(dp), np.array(dg)
    print('相邻帧转角：真值中位 %.1f°（p95 %.1f°）；预测中位 %.1f°（p95 %.1f°）→ 预测抖动' % (np.median(dg), np.percentile(dg, 95), np.median(dp), np.percentile(dp, 95)))
    # 不同平滑窗口的粗看（只描述，选择在 select 里按预登记做）
    for k in (5, 9, 15):
        for mode in ('causal', 'centered'):
            _, a2, _ = errs(filt_rot(recs, k, mode))
            print('  朝向 %s k=%d：角度中位 %.1f°' % (mode, k, np.nanmedian(a2)))


def run_grid():
    recs = pickle.load(open(O + '/final_val_recs_v.pkl', 'rb'))
    res = {'none': F.summ(recs, 'tf_val_none', ('indoor', 'laa'))}
    print('none', res['none']['ads_indoor'], round(res['none']['ang'], 2), flush=True)
    for mode in ('causal', 'offline'):
        for g in (3.0, 10.0, 30.0):
            for w in (1, 9, 15):
                nm = 'hmm_%s_g%g_w%d' % (mode, g, w)
                res[nm] = F.summ(filt_hmm(recs, g, mode, w), 'tf_val_' + nm, ('indoor', 'laa'))
                print(nm, res[nm]['ads_indoor'], round(res[nm]['ang'], 2), flush=True)
    for mode in ('causal', 'centered'):
        for k in (9, 15):
            for T in (0.18, 0.3):
                for tr in (False, True):
                    nm = 'pos_%s_k%d_T%.2f%s' % (mode, k, T, '_trunc' if tr else '')
                    res[nm] = F.summ(filt_pos(recs, k, T, mode, tr), 'tf_val_' + nm, ('indoor', 'laa'))
                    print(nm, res[nm]['ads_indoor'], round(res[nm]['pos'], 4), flush=True)
    sel = {}
    for mode, pm in (('causal', 'causal'), ('offline', 'centered')):
        Hh = [k for k in res if k.startswith('hmm_' + mode)]
        P = [k for k in res if k.startswith('pos_' + pm)]
        sel[mode] = dict(rot=max(Hh, key=lambda k: res[k]['ads_indoor']), pos=max(P, key=lambda k: res[k]['ads_indoor']))
    json.dump(dict(res=res, sel=sel, prereg='PREREG_temporal.md f0e4995c'), open(O + '/temporal_select.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('选中', sel)


def parse(nm):
    a = nm.split('_')
    if a[0] == 'hmm':
        return dict(kind='hmm', mode=a[1], g=float(a[2][1:]), w=int(a[3][1:]))
    if a[0] == 'pos':
        return dict(kind='pos', mode=a[1], k=int(a[2][1:]), T=float(a[3][1:]), trunc=nm.endswith('_trunc'))
    return dict(kind='rot', mode=a[1], k=int(a[2][1:]))


def do(recs, nm):
    p = parse(nm)
    if p['kind'] == 'hmm':
        return filt_hmm(recs, p['g'], p['mode'], p['w'])
    return filt_pos(recs, p['k'], p['T'], p['mode'], p['trunc']) if p['kind'] == 'pos' else filt_rot(recs, p['k'], p['mode'])


def final():
    S = json.load(open(O + '/temporal_select.json', encoding='utf-8'))
    recs = build('test')
    rows = {'最终版（逐帧，不滤波）': F.summ(recs, 'tf_test_none')}
    for mode, nmode in (('causal', '在线（只用过去帧）'), ('offline', '离线（整段）')):
        s = S['sel'][mode]
        rp = do(recs, s['pos'])
        rows['+ 位置滤波 %s %s' % (nmode, s['pos'])] = F.summ(rp, 'tf_test_' + s['pos'])
        rr = do(recs, s['rot'])
        rows['+ 朝向滤波 %s %s' % (nmode, s['rot'])] = F.summ(rr, 'tf_test_' + s['rot'])
        rows['+ 两个都滤 %s' % nmode] = F.summ(do(rp, s['rot']), 'tf_test_both_' + mode)
    base = rows['最终版（逐帧，不滤波）']
    L = ['', '## 时间滤波（PREREG_temporal.md；验证选、测试一次）', '',
         '| 配置 | 检出 | 位置中位 | <20cm | <50cm | 角度中位 | score | ADS indoor | ADS laa | 判据 |', '|---|---|---|---|---|---|---|---|---|---|']
    for nm, r in rows.items():
        ok = '' if r is base else ('成立' if (r['ads_indoor'] > base['ads_indoor'] and r['ads_laa'] > base['ads_laa'] - 0.05) else '判负')
        L.append('| %s | %.1f%% | %.3f m | %.1f%% | %.1f%% | %.1f° | %.4f | %.2f | %.2f | %s |' % (
            nm, r['det'], r['pos'], r['lt20'], r['lt50'], r['ang'], r['score'], r['ads_indoor'], r['ads_laa'], ok))
    open(G + '/FINAL_ADS.md', 'a', encoding='utf-8').write('\n'.join(L) + '\n')
    json.dump(rows, open(O + '/temporal_test_rows.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n'.join(L))


def build_val():
    pickle.dump(build('val'), open(O + '/final_val_recs_v.pkl', 'wb'))


if __name__ == '__main__':
    {'diag': diag, 'select': run_grid, 'final': final, 'build_val': build_val}[sys.argv[1]]()
