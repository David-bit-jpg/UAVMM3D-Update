# -*- coding: utf-8 -*-
"""PREREG_bgs.md 第 1 步（sha256 2f584b93）：背景差分剪影零训练诊断 → DIAG_BGS.md / bgs/diag.json / bgs/gallery.jpg。
真实测试 / 验证每段时间中位背景（非因果，诊断用）；第一段 ST2 单模型 k = 0..4；第二段 j = 3（k 偶）/ 4（k 奇）。
① SILIN_s{j} 吃背景差分软剪影（质量差回退 V2Ap）② V2Ap 学生 ③ SILIN 吃学生剪影。"""
import collections
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch

ROOT = '@@ROOT@@'
G = ROOT + '/output/camnorm/geo1004'
sys.path.insert(0, ROOT)
sys.path.insert(0, G)
from scipy.spatial.transform import Rotation as R  # noqa: E402
from uavdet3d.utils import pose_eval as pe  # noqa: E402
import cropw_extract as ce  # noqa: E402
import silw_train as sw  # noqa: E402
import softkd_train as skt  # noqa: E402
import softkd_analyze as ska  # noqa: E402
import secondstage_v2 as v2  # noqa: E402
from anchor_eval import deploy  # noqa: E402

ce.P, ce.KW = 256, 3.0
S0 = float(np.sqrt(1.2 * 2.5))
O = G + '/bgs'
SRC = {'test': G + '/st2_recs/test_s%d.pkl', 'val': G + '/st2_recs/val_s%d.pkl'}
NAME = {'test': '真实测试', 'val': '真实验证'}


class Frames:
    def __init__(self, split):
        b = '@@MMCACHE@@/mav6d_cn_obj/' + split
        self.arr = np.load(b + '/rgb.npy', mmap_mode='r')
        d = pickle.load(open(b + '/index.pkl', 'rb'))
        self.metas = d['metas']
        self.key = {(str(m['seq']), str(m['frame'])): i for i, m in enumerate(self.metas)}
        fn = '%s/bg_%s.npz' % (O, split)
        if os.path.exists(fn):
            z = np.load(fn)
            self.bg = {k: z[k] for k in z.files}
        else:
            by = collections.defaultdict(list)
            for i, m in enumerate(self.metas):
                by[str(m['seq'])].append(i)
            self.bg = {}
            for s, ii in by.items():
                sel = ii[::max(1, len(ii) // 60)][:60]
                self.bg[s] = np.median(np.stack([self.arr[i] for i in sel]), 0).astype(np.uint8)
            np.savez(fn, **self.bg)

    def get(self, seq, frame):
        i = self.key[(seq, frame)]
        return np.ascontiguousarray(self.arr[i]), self.bg[seq], np.asarray(self.metas[i]['K_in'], np.float64)


def bgs_mask(view, bview):
    """→ 软掩码 (128,128) float32 [0,1]、是否可用、外圈激活率"""
    D = np.abs(view.astype(np.int16) - bview.astype(np.int16)).max(-1).astype(np.float32)
    ring = np.ones_like(D, bool)
    ring[10:-10, 10:-10] = False
    t0 = 10.0 + 2.0 * float(np.median(D[ring]))
    p = np.clip((D - t0) / 15.0, 0.0, 1.0)
    act = float((p[ring] > 0.5).mean())
    n, lab = cv2.connectedComponents((p > 0.5).astype(np.uint8), connectivity=8)
    keep = sorted(set(np.unique(lab[43:85, 43:85])) - {0})
    if not keep:
        return np.zeros_like(p), False, act
    m = cv2.dilate(np.isin(lab, keep).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    p = cv2.GaussianBlur(p * m, (0, 0), 1.0)
    good = act <= 0.02 and float((p > 0.5).sum()) >= 10
    return p.astype(np.float32), good, act


def load(name, j):
    n = v2.make(name).cuda()
    n.load_state_dict(torch.load('%s/softkd/%s_s%d.pt' % (G, name, j), map_location='cuda'))
    return n.eval()


@torch.no_grad()
def compute(recs, fr, j, gal=None):
    stu, sil = load('V2Ap', j), load('SILIN', j)
    items, views, bviews, scal, ty = [], [], [], [], []
    for ri, r in enumerate(recs):
        if len(r['pred']) == 0:
            continue
        img, bg, K = fr.get(r['seq_id'], r['frame_id'])
        for pi, g in enumerate(r['pred']):
            if g[2] <= 1e-3:
                continue
            c = (pe.PROTO8 * g[3:6]) @ R.from_euler('xyz', g[6:9]).as_matrix().T + g[:3]
            q = (K @ c.T).T
            if (q[:, 2] <= 1e-3).any():
                continue
            uv = q[:, :2] / q[:, 2:3]
            u0, u1, v0, v1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
            w = u1 - u0
            if not (6.0 <= w <= 170.0):
                continue
            cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            views.append(sw.view_img(ce.window(img, cx, cy, ce.KW * w), S0, 0.0, 0.0, 1.0, w, False))
            bviews.append(sw.view_img(ce.window(bg, cx, cy, ce.KW * w), S0, 0.0, 0.0, 1.0, w, False))
            scal.append([np.log(w / K[0, 0]), np.log(g[2])] + list(np.log(np.maximum(g[3:6], 1e-3))))
            items.append((ri, pi))
            ty.append(r['seq_id'].split('/')[0])
    scal = np.array(scal, np.float32)
    masks = [bgs_mask(a, b) for a, b in zip(views, bviews)]
    P = np.stack([m[0] for m in masks]) if masks else np.zeros((0, 128, 128), np.float32)
    good = np.array([m[1] for m in masks], bool)
    dA, dB, dC, iou = [], [], [], []
    for b in range(0, len(views), 256):
        x = skt.dt.prep(np.stack(views[b:b + 256]), None, False).cuda()
        s = torch.from_numpy(scal[b:b + 256]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            da, _, lg = stu(x, s)
            ps = torch.sigmoid(lg.float()).cpu().numpy()
            xb = skt.dt.prep(skt.gray3((P[b:b + 256] * 255).round().astype(np.uint8)), None, False).cuda()
            db, _, _ = sil(xb, s)
            xc = skt.dt.prep(skt.gray3((ps * 255).round().astype(np.uint8)), None, False).cuda()
            dc, _, _ = sil(xc, s)
        dA.append(da.float().cpu().numpy()); dB.append(db.float().cpu().numpy()); dC.append(dc.float().cpu().numpy())
        a, m = ps > 0.5, P[b:b + 256] > 0.5
        iou += list((a & m).sum((1, 2)) / np.maximum((a | m).sum((1, 2)), 1))
        if gal is not None:
            gal.append((np.stack(views[b:b + 256]), (P[b:b + 256] * 255).astype(np.uint8), (ps * 255).astype(np.uint8), good[b:b + 256], ty[b:b + 256]))
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)
    return dict(items=np.array(items, np.int64).reshape(-1, 2), good=good, ty=np.array(ty), iou=np.array(iou),
                act=np.array([m[2] for m in masks]), dA=cat(dA), dB=cat(dB), dC=cat(dC))


def pr_of(recs, items, d):
    pr = [(np.full(len(r['pred']), np.nan), np.zeros(len(r['pred']))) for r in recs]
    for n, (ri, pi) in enumerate(items):
        pr[ri][0][pi] = d[n]
    return pr


def gallery(gal):
    views, P, ps, good, ty = [np.concatenate([g[i] for g in gal]) for i in range(5)]
    rows = []
    for t in ('phantom4', 'mavic2'):
        idx = np.nonzero(ty == t)[0]
        pick = idx[np.linspace(0, len(idx) - 1, 8).astype(int)] if len(idx) else []
        for i in pick:
            m1 = cv2.cvtColor(P[i], cv2.COLOR_GRAY2BGR)
            m2 = cv2.cvtColor(ps[i], cv2.COLOR_GRAY2BGR)
            if not good[i]:
                cv2.rectangle(m1, (0, 0), (127, 127), (0, 0, 255), 2)
            rows.append(np.concatenate([views[i], m1, m2], 1))
    cols = [np.concatenate(rows[:8], 0), np.concatenate(rows[8:16], 0)] if len(rows) == 16 else [np.concatenate(rows, 0)]
    cv2.imwrite(O + '/gallery.jpg', np.concatenate(cols, 1), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    os.makedirs(O, exist_ok=True)
    FR = {sp: Frames(sp) for sp in ('test', 'val')}
    res = {}
    gal = []
    for k in range(5):
        j = 3 if k % 2 == 0 else 4
        for sp in ('test', 'val'):
            recs = pickle.load(open(SRC[sp] % k, 'rb'))['records']
            out = compute(recs, FR[sp], j, gal if (k == 0 and sp == 'test') else None)
            arms = {'A': out['dA'], 'C': out['dC'], 'B': np.where(out['good'], out['dB'], out['dA'])}
            rows = {}
            for a, d in arms.items():
                pr = pr_of(recs, out['items'], d)
                rows[a] = ska.top1(deploy(recs, pr, ska.cval(pr)))
            keys = sorted(set.intersection(*[set(v) for v in rows.values()]))
            tk = np.array([q[0].split('/')[0] for q in keys])
            r = res.setdefault(sp, {})
            for a in arms:
                r.setdefault(a, []).append(ska.debiased(rows[a], keys))
                for t in ('mavic2', 'phantom4'):
                    r.setdefault(a + '_' + t, []).append(ska.debiased(rows[a], keys, tk == t))
            for t in ('mavic2', 'phantom4'):
                sel = out['ty'] == t
                r.setdefault('good_' + t, []).append(100 * float(out['good'][sel].mean()))
                r.setdefault('iou_' + t, []).append(float(np.median(out['iou'][sel & out['good']])) if (sel & out['good']).any() else float('nan'))
            print(k, sp, '逐帧 ②学生 %.1f ①背景差分 %.1f ③学生剪影 %.1f | phantom4 %.1f→%.1f | 可用 p4 %.0f%% m2 %.0f%%' % (
                r['A'][-1], r['B'][-1], r['C'][-1], r['A_phantom4'][-1], r['B_phantom4'][-1], r['good_phantom4'][-1], r['good_mavic2'][-1]), flush=True)
        if k == 0:
            gallery(gal)
            gal = None
    A = np.array
    L = ['# 背景差分剪影零训练诊断（PREREG_bgs.md 第 1 步，sha256 2f584b93）', '',
         '第一段 ST2 单模型 k = 0..4；第二段 j = 3（k 偶）/ 4（k 奇）。扣全局倍数逐帧位置中位（cm），5 个检测种子平均。', '',
         '① = 只看剪影的深度网络（SILIN）吃背景差分剪影（质量差的回退学生）；② = 现管线学生 A+；③ = SILIN 吃学生自己画的剪影。', '']
    dec = {}
    for sp in ('test', 'val'):
        r = res[sp]
        L += ['## ' + NAME[sp], '', '| | 全部 | mavic2 | phantom4 |', '|---|---|---|---|']
        for a, nm in (('A', '② 学生 A+'), ('B', '① 背景差分剪影'), ('C', '③ 学生剪影')):
            L.append('| %s | %.1f | %.1f | %.1f |' % (nm, A(r[a]).mean(), A(r[a + '_mavic2']).mean(), A(r[a + '_phantom4']).mean()))
        dd = {t: A(r['B' + t]) - A(r['A' + t]) for t in ('', '_mavic2', '_phantom4')}
        L += ['', '① − ②：全部 %+.2f（%d/5）、mavic2 %+.2f（%d/5）、phantom4 %+.2f（%s；%d/5）' % (
            dd[''].mean(), (dd[''] < 0).sum(), dd['_mavic2'].mean(), (dd['_mavic2'] < 0).sum(), dd['_phantom4'].mean(),
            ' '.join('%+.2f' % v for v in dd['_phantom4']), (dd['_phantom4'] < 0).sum()),
              '背景差分可用比例：mavic2 %.0f%%、phantom4 %.0f%%；学生剪影与背景差分剪影交并比中位：mavic2 %.2f、phantom4 %.2f' % (
            A(r['good_mavic2']).mean(), A(r['good_phantom4']).mean(), np.nanmean(r['iou_mavic2']), np.nanmean(r['iou_phantom4'])), '']
        dec[sp] = bool(dd['_phantom4'].mean() <= -1.0 and (dd['_phantom4'] < 0).sum() >= 4)
    go = dec['test'] and dec['val']
    L += ['## 决策', '', '**phantom4 ① − ② ≤ −1.0 cm 且 ≥ 4/5，两集合：%s → %s**（测试 %s / 验证 %s）' % (
        '成立' if go else '不成立', '做第 2 步训练' if go else '这条线停', '✓' if dec['test'] else '✗', '✓' if dec['val'] else '✗'),
          '', '图：bgs/gallery.jpg（每行：彩色裁窗 | 背景差分剪影（红框 = 判差） | 学生剪影；左 phantom4、右 mavic2，检测种子 0 测试集）']
    open(G + '/DIAG_BGS.md', 'w', encoding='utf-8').write('\n'.join(L))
    json.dump(dict(res=res, go=go), open(O + '/diag.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
