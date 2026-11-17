# -*- coding: utf-8 -*-
"""对比实验统一评测集：现最终版（st2p_{split}_kzt7q）在真实 val/test 上每帧的部署裁窗（与 KZ 看到的窗口逐像素相同），
加上裁窗内参 K_crop、部署位置 t / 尺寸 lwh（现最终版第一段检测器给的）、真值框（封存文件，只评估用）。
输出 cmpeval/{split}_crops_u8.npy + {split}_meta.npz。python build_eval_crops.py val|test"""
import sys, pickle, concurrent.futures as cf
import numpy as np
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
for p in (SCR + '/kzst2', SCR + '/kzst1', SCR + '/repl', '@@ROOT@@'):
    sys.path.insert(0, p)
import kzst2_common as S
import repl_common as RC
import adabn_kz as AB
C1 = S.C1
split = sys.argv[1]
O = '@@ROOT@@/output/camnorm/geo1004/cmpeval'
P = RC.tag_paths('st2p_%s_kzt7q' % split)
D, info = C1.load_npz_info(P['npz'])
frames, _ = RC.load_frames(P['pkl'], P['sealed'], with_gt=True)
keys = [(str(s), str(f)) for s, f in zip(D['seq'], D['frame'])]
rows = []
for i, k in enumerate(keys):
    fr = frames[k]
    t_ = C1.hook_top(D, i, fr)
    if t_ < 0:
        continue
    gt = np.asarray(fr['gt'], np.float64).reshape(-1, 9)
    x0, y0, w, h, s = [float(v) for v in D['crop'][i]]
    Kc = S.crop_K(D['K_raw_src'][i], x0, y0, s)
    rows.append(dict(i=i, seq=k[0], frame=k[1], cls=str(fr.get('cls', k[0].split('/')[0])), K=Kc, t=np.asarray(fr['pred'][t_, :3], np.float64),
                     lwh=np.asarray(fr['pred'][t_, 3:6], np.float64), gt=gt))
n = len(rows)
src = AB.NpzSource(D)
arr = np.lib.format.open_memmap(O + '/%s_crops_u8.npy' % split, mode='w+', dtype=np.uint8, shape=(n, S.IN_WH[1], S.IN_WH[0], 3))
def cut(m):
    arr[m] = AB.recut(D, rows[m]['i'], src, S.IN_WH)
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    list(ex.map(cut, range(n)))
arr.flush()
G = np.zeros((n, 9)); ng = np.zeros(n, int)
for m, r in enumerate(rows):
    if len(r['gt']):
        j = int(np.argmin(np.linalg.norm(r['gt'][:, :3] - r['t'], axis=1))); G[m] = r['gt'][j]; ng[m] = len(r['gt'])
np.savez(O + '/%s_meta.npz' % split, seq=np.array([r['seq'] for r in rows]), frame=np.array([r['frame'] for r in rows]),
         cls=np.array([r['cls'] for r in rows]), K=np.stack([r['K'] for r in rows]), t=np.stack([r['t'] for r in rows]),
         lwh=np.stack([r['lwh'] for r in rows]), gt=G, n_gt=ng, src_idx=np.array([r['i'] for r in rows]))
d = np.linalg.norm(G[:, :3] - np.stack([r['t'] for r in rows]), axis=1)
print(split, '帧', n, '有真值', int((ng > 0).sum()), '部署位置误差中位 %.3f m' % np.median(d[ng > 0]), dict(zip(*np.unique([r['cls'] for r in rows], return_counts=True))))
