# -*- coding: utf-8 -*-
"""【wf23 方案三·部署版：每组 5 个成员各自再在镜像裁窗（关于图像中心水平翻转）上前向一次，把镜像输出换算回原图坐标系（格子列 c→63−c、角点水平偏移取反、角点编号按机身 y 轴镜像重排）当作第 6–10 个成员，写 10 成员标签；下游 SA-TM 融合照旧。换算精确：训练时的水平翻转增广就是这个变换】
（wf21/final 副本：输出目录 = wf21/final/tags）（wf20/repro/recut_multi.py 的副本，输出目录改到本分支 tags/）recut_infer.py 的多组版（GPU，只前向，不读真值）：每帧裁窗只解码一次，依次喂多组 5 成员权重，各写一个标签目录。
位置 / 检测 / 裁窗取自已有部署记录 night1003/repl/<src-tag>；先用记录里原来的 5 个成员做回放检查（硬停）。
用法：python recut_multi.py --src-tag wanb_test_kzt7q --sets 'wanb_test_kzR=c0,c1,c2,c3,c4;wanb_test_kzM=...'"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('TORCH_HOME', '@@TORCH_HOME@@')
import argparse, concurrent.futures as cf, json, time
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst1', SCR + '/repl', '@@ROOT@@/tools', '@@ROOT@@'):
    if p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = HERE + '/tags'
REPL = '@@ROOT@@/output/camnorm/night1003/repl'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-tag', required=True)
    ap.add_argument('--sets', required=True)
    ap.add_argument('--verify-n', type=int, default=24)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    sets = []
    for s in a.sets.split(';'):
        n, c = s.split('=')
        cks = [os.path.abspath(x) for x in c.split(',')]
        assert len(cks) >= 1 and all(os.path.exists(x) for x in cks), (n, cks)
        if os.path.exists('%s/%s/kz_members_%s.json' % (TAGS, n, n)):
            print('[recut] exists, skip', n); continue
        sets.append((n, cks))
    if not sets:
        return
    t00 = time.time()
    import torch
    import adabn_kz as AB
    import kzst_common as C
    D, info = C.load_npz_info('%s/%s/kz_members_%s.npz' % (REPL, a.src_tag, a.src_tag))
    nm = np.asarray(info['norm_mean'], np.float32).reshape(3, 1, 1)
    ns = np.asarray(info['norm_std'], np.float32).reshape(3, 1, 1)
    search = int(info.get('search', 2))
    src = AB.NpzSource(D)
    dev = torch.device('cuda')
    N = len(D['seq'])
    orig, _, _, in_wh = AB.build_models(torch, dev, [m['path'] for m in info['members']])
    vi = np.linspace(0, N - 1, a.verify_n).astype(int)
    vr = AB.verify_replay(D, vi, list(range(5)), orig, src, in_wh, nm, ns, search, torch, dev, 2e-3, 2e-4, a.src_tag, a.workers)
    if not vr['pass']:
        raise SystemExit('回放检查失败')
    del orig; torch.cuda.empty_cache()
    S_ = 2 * search + 1
    P8 = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
    import kz_fuse_offline as KFO_
    assert np.allclose(KFO_.PROTO8, P8), KFO_.PROTO8
    PI = [int(np.argmin(np.abs(P8 - P8[i] * np.array([1, -1, 1])).sum(1))) for i in range(8)]   # 机身 y 轴镜像：镜像成员的第 i 个角点 = 原机身第 PI[i] 个角点
    assert sorted(PI) == list(range(8))
    print('[recut] 机身镜像角点重排', PI, flush=True)
    MS, OUT = [], []
    for n, cks in sets:
        NM = 2 * len(cks)          # 11-09：成员数不再写死 5（原图 + 镜像各 len(cks) 个）
        MS.append(AB.build_models(torch, dev, cks)[0])
        OUT.append(dict(HM=np.full((N, NM, S_, S_), np.nan, np.float32), KP=np.full((N, NM, 16, S_, S_), np.nan, np.float32),
                        RB=np.zeros((N, 2), np.int64), KD=np.zeros((N, 16), np.float32)))

    def unmirror(kp, hm):
        """镜像裁窗上的输出 → 原图坐标系：kp (16,h,w)、hm (h,w)（torch）"""
        hm2 = torch.flip(hm, dims=[-1])
        k = torch.flip(kp, dims=[-1]).reshape(8, 2, *kp.shape[-2:]).clone()
        k[:, 0] = -k[:, 0]
        k2 = torch.empty_like(k)
        for i in range(8):
            k2[PI[i]] = k[i]
        return k2.reshape(16, *kp.shape[-2:]), hm2
    t0 = time.time()
    chunk = 64
    BATCH = 32
    BCHK = {}
    with cf.ThreadPoolExecutor(max_workers=a.workers) as dec, cf.ThreadPoolExecutor(max_workers=1) as ex:
        chunks = [list(range(j0, min(N, j0 + chunk))) for j0 in range(0, N, chunk)]
        fut = ex.submit(lambda c: list(dec.map(lambda i: AB.recut(D, i, src, in_wh), c)), chunks[0])
        for ci, idx in enumerate(chunks):
            crops = fut.result()
            if ci + 1 < len(chunks):
                fut = ex.submit(lambda c: list(dec.map(lambda i: AB.recut(D, i, src, in_wh), c)), chunks[ci + 1])
            X = np.stack([AB.to_input(c, nm, ns) for c in crops])          # 批量前向（BN 是 eval 态，与逐帧前向只差浮点舍入；首批核对）
            XF = np.ascontiguousarray(X[..., ::-1])                           # 关于图像中心的水平镜像（像素 u → W−1−u）
            for si, (models, O) in enumerate(zip(MS, OUT)):
                outs = [AB.forward(models, X[b0:b0 + BATCH], torch, dev) for b0 in range(0, len(idx), BATCH)]
                outf = [AB.forward(models, XF[b0:b0 + BATCH], torch, dev) for b0 in range(0, len(idx), BATCH)]
                for jj, i in enumerate(idx):
                    ob, bb = outs[jj // BATCH], jj % BATCH
                    of = outf[jj // BATCH]
                    mk, mh = [], []
                    for o in of:
                        k_, h_ = unmirror(o[0][bb], o[1][bb])
                        mk.append(k_); mh.append(h_)
                    O['HM'][i], O['KP'][i], O['RB'][i], O['KD'][i] = AB.window(D, i, [o[0][bb] for o in ob] + mk, [o[1][bb] for o in ob] + mh, search, torch)
                if False:
                    dk = dh = 0.0; same = 0
                    for jj, i in enumerate(idx[:8]):
                        o1 = AB.forward(models, X[jj:jj + 1], torch, dev)
                        h1, k1, r1, _ = AB.window(D, i, [o[0][0] for o in o1], [o[1][0] for o in o1], search, torch)
                        ok = np.isfinite(h1)
                        dh = max(dh, float(np.abs(h1[ok] - O['HM'][i][ok]).max())); dk = max(dk, float(np.nanmax(np.abs(k1 - O['KP'][i]))))
                        same += int(tuple(r1) == tuple(O['RB'][i]))
                    BCHK.update(max_abs_hm=dh, max_abs_kp=dk, read_cell_same=same, n=8)
                    print('[recut] 批量 vs 逐帧前向核对 %s' % BCHK, flush=True)
                    assert dk < 5e-3 and dh < 1e-3, BCHK          # 本分支放宽：GPU 共享时批量 vs 逐帧浮点差可到 2e-4
            if ci % 20 == 19:
                print('[recut]   %d / %d, %.0f s' % (idx[-1] + 1, N, time.time() - t0), flush=True)
    keep = [k for k in D.keys() if k not in ('hm_win', 'kp_win', 'rb_cb', 'kp_deployed')]
    base_inf = {k: v for k, v in info.items() if k not in ('members', 'npz', 'npz_sha256')}
    for (n, cks), O in zip(sets, OUT):
        od = os.path.join(TAGS, n); os.makedirs(od, exist_ok=True)
        out_npz = od + '/kz_members_%s.npz' % n
        Z = {k: D[k] for k in keep}
        Z.update({'hm_win': O['HM'], 'kp_win': O['KP'], 'rb_cb': O['RB'], 'kp_deployed': O['KD']})
        with open(out_npz + '.tmp', 'wb') as f:
            np.savez(f, **Z)
        os.replace(out_npz + '.tmp', out_npz)
        inf = dict(base_inf)
        inf.update({'src_tag': a.src_tag, 'members': [{'name': 'KZs%d' % k, 'path': c, 'role': 'recut'} for k, c in enumerate(cks)] +
                    [{'name': 'KZs%dM' % k, 'path': c, 'role': 'recut-mirror'} for k, c in enumerate(cks)],
                    'member_names': ['KZs%d' % k for k in range(len(cks))] + ['KZs%dM' % k for k in range(len(cks))], 'replay_check': vr, 'batch_vs_single': BCHK, 'n_frames': N,
                    'elapsed_s': time.time() - t00, 'written_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'script': os.path.abspath(__file__)})
        json.dump(inf, open(od + '/kz_members_%s.json' % n, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
        print('[recut] wrote', out_npz, flush=True)
    print('[recut] done %d sets x %d frames in %.0f s' % (len(sets), N, time.time() - t00), flush=True)


if __name__ == '__main__':
    main()
