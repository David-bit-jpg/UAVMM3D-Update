# -*- coding: utf-8 -*-
"""核查某个 MAV6D 模型的结果是不是来自泄露或过拟合（默认 25% 档 C_p025_s0）。

泄露
  L1 划分：train / val / test 序列名、帧是否重叠；训练实际用到的帧（SAMPLED_INTERVAL 抽样）是否全在 train 里
  L2 时间相邻：帧名是纳秒时间戳。每个测试序列与同机型同场景的训练/验证序列的时间间隔 —— 同一次飞行被切开会表现为间隔很小
  L3 近重复：每个测试帧在「训练实际用到的帧」里找最近邻（同机型）：
       位姿最近邻（位置差 + 旋转角） 与 目标裁块外观最近邻（目标中心 48px 裁块，灰度归一化后余弦相似度），
       对照组 = 训练帧在【其他训练序列】里的最近邻（跨序列本来就有多像）与【同序列相邻帧】（真近重复有多像）
  L4 误差是否依赖「训练里有没有近重复」：按位姿最近邻距离分箱看测试误差
过拟合
  O1 同一权重在 训练实际用到的帧 / 验证集 / 测试集 上的位姿指标（增广关、同一解码）
  O2 验证曲线：选中第几轮、之后是否下降

    python check_leak_overfit.py --tag C_p025_s0 --interval 4
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavdet3d.config import cfg_from_yaml_file   # noqa: E402
from uavdet3d.datasets import build_dataloader   # noqa: E402
from uavdet3d.model import build_network   # noqa: E402
from uavdet3d.utils import common_utils, pose_eval   # noqa: E402

CACHE = 'E:/mmcache/mav6d_cn'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_split(sp):
    idx = pickle.load(open(os.path.join(CACHE, sp, 'index.pkl'), 'rb'))
    return idx, np.load(os.path.join(CACHE, sp, 'rgb.npy'), mmap_mode='r')


def frame_info(idx, rgb, sel):
    out = []
    for i in sel:
        m = idx['metas'][int(i)]
        cls, scene, seq = m['seq'].split('/')
        b = m['boxes9d'][0].astype(np.float64)
        K = np.asarray(m['K_in'], np.float64)
        u = K @ b[:3]
        out.append({'i': int(i), 'cls': cls, 'scene': scene, 'seq': m['seq'], 't': int(os.path.splitext(m['frame'])[0]),
                    'pos': b[:3], 'quat': R.from_euler('xyz', b[6:9]).as_quat(), 'uv': (u[0] / u[2], u[1] / u[2])})
    return out


def crop_vec(rgb, f, half=24):
    img = np.asarray(rgb[f['i']])
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 少数测试帧真值中心在画面外（MAV6D 原图就出画），裁块中心夹到图内
    cu = int(np.clip(round(f['uv'][0]), 0, g.shape[1] - 1))
    cv_ = int(np.clip(round(f['uv'][1]), 0, g.shape[0] - 1))
    pad = cv2.copyMakeBorder(g, half, half, half, half, cv2.BORDER_REFLECT)
    c = cv2.resize(pad[cv_:cv_ + 2 * half, cu:cu + 2 * half], (32, 32), interpolation=cv2.INTER_AREA).reshape(-1)
    c = c - c.mean()
    return c / (np.linalg.norm(c) + 1e-6)


def pose_nn(A, B, same_seq_mask=None):
    """A、B 为 frame_info 列表（同机型）。返回 A 中每帧到 B 的最近邻（位置差 m、旋转角 °），按 位置差/0.1m + 角度/10° 取最近。"""
    pa = np.stack([f['pos'] for f in A])
    pb = np.stack([f['pos'] for f in B])
    qa = np.stack([f['quat'] for f in A])
    qb = np.stack([f['quat'] for f in B])
    dpos = np.zeros(len(A))
    dang = np.zeros(len(A))
    for s in range(0, len(A), 256):
        dp = np.linalg.norm(pa[s:s + 256, None] - pb[None], axis=2)
        da = np.degrees(2 * np.arccos(np.clip(np.abs(qa[s:s + 256] @ qb.T), 0, 1)))
        cost = dp / 0.1 + da / 10.0
        if same_seq_mask is not None:
            cost = np.where(same_seq_mask[s:s + 256], np.inf, cost)
        k = np.argmin(cost, axis=1)
        dpos[s:s + 256] = dp[np.arange(len(k)), k]
        dang[s:s + 256] = da[np.arange(len(k)), k]
    return dpos, dang


def evaluate(tag, split, interval):
    cfg = EasyDict()
    cfg_from_yaml_file('cfgs/models/uavdet_3d/camnorm/mav6d.yaml', cfg)
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = split
    cfg.DATA_CONFIG.SAMPLED_INTERVAL['test'] = interval
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, batch_size=8, dist=False, workers=0,
                                     logger=common_utils.create_logger(), training=False)
    model = build_network(cfg.MODEL, ds)
    ck = os.path.join(ROOT, 'output', 'models', 'uavdet_3d', 'camnorm', 'mav6d', tag, 'ckpt', 'best.pth')
    n_loaded, n_total = model.load_params_from_file(ck, to_cpu=False)
    assert n_loaded == n_total
    model.cuda().eval()
    recs = pose_eval.run_inference(model, loader)
    pairs = pose_eval.match_records(recs, seq='xyz', match='top1')
    return pose_eval.summarize(pairs, n_frames=len(recs)), recs, [int(i) for i in ds.valid_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='C_p025_s0')
    ap.add_argument('--interval', type=int, default=4, help='该模型训练时的 SAMPLED_INTERVAL.train')
    ap.add_argument('--out', default=os.path.join(ROOT, 'output', 'camnorm', 'check_leak_overfit'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {}

    tr_idx, tr_rgb = load_split('train')
    va_idx, _ = load_split('val')
    te_idx, te_rgb = load_split('test')
    used = tr_idx['valid_idx'][::args.interval]

    # ---------------- L1 划分 ----------------
    def keys(ix, sel):
        return set(ix['metas'][int(i)]['seq'] for i in sel), set((ix['metas'][int(i)]['seq'], ix['metas'][int(i)]['frame']) for i in sel)
    s_tr, f_tr = keys(tr_idx, tr_idx['valid_idx'])
    s_va, f_va = keys(va_idx, va_idx['valid_idx'])
    s_te, f_te = keys(te_idx, te_idx['valid_idx'])
    s_used, f_used = keys(tr_idx, used)
    L1 = {'train_seq': len(s_tr), 'val_seq': len(s_va), 'test_seq': len(s_te), 'used_frames': len(f_used),
          'seq_train&val': len(s_tr & s_va), 'seq_train&test': len(s_tr & s_te), 'seq_val&test': len(s_va & s_te),
          'frame_used&test': len(f_used & f_te), 'frame_used&val': len(f_used & f_va), 'used_subset_of_train': f_used <= f_tr}
    report['L1'] = L1
    print('L1 划分:', L1)

    # ---------------- L2 时间相邻 ----------------
    # 时间戳取标签文件第一列（纳秒）：mavic2 帧名就是时间戳，phantom4 帧名是 1.jpg 2.jpg... 不能用帧名。
    # 每个序列只读首尾两帧的标签（按帧名数值排序）
    frames_of = defaultdict(list)
    split_of = {}
    for sp, ix in (('train', tr_idx), ('val', va_idx), ('test', te_idx)):
        for i in ix['valid_idx']:
            m = ix['metas'][int(i)]
            frames_of[m['seq']].append(m['frame'])
            split_of[m['seq']] = sp

    def label_ts(seq, frame):
        cls, scene, sq = seq.split('/')
        with open(os.path.join('E:/MAV6D', cls, 'labels', scene, sq, os.path.splitext(frame)[0] + '.txt')) as fh:
            return int(fh.readline().split()[0])
    span = {}
    for sq, frs in frames_of.items():
        frs = sorted(frs, key=lambda x: int(os.path.splitext(x)[0]))
        span[sq] = [label_ts(sq, frs[0]), label_ts(sq, frs[-1])]
    gaps = []
    for sq, (a0, a1) in span.items():
        if split_of[sq] != 'test':
            continue
        cls, scene, _ = sq.split('/')
        best = (np.inf, None)
        for sq2, (b0, b1) in span.items():
            if split_of[sq2] == 'test' or not sq2.startswith(cls + '/' + scene + '/'):
                continue
            gap = max(b0 - a1, a0 - b1) / 1e9           # 负数 = 时间区间重叠
            if abs(gap) < abs(best[0]):
                best = (gap, sq2)
        gaps.append((sq, (a1 - a0) / 1e9, best[0], best[1], split_of.get(best[1])))
    # 对照：训练序列之间（同机型同场景）的最小时间间隔分布 —— 官方就是把连续录制切成序列
    tr_gaps = []
    for sq, (a0, a1) in span.items():
        if split_of[sq] != 'train':
            continue
        cls, scene, _ = sq.split('/')
        g = [max(b0 - a1, a0 - b1) / 1e9 for sq2, (b0, b1) in span.items()
             if sq2 != sq and split_of[sq2] == 'train' and sq2.startswith(cls + '/' + scene + '/')]
        if g:
            tr_gaps.append(min(g, key=abs))
    print('    对照：训练序列到最近的其他训练序列 时间间隔 中位 %.1f s（p10 %.1f，p90 %.1f）'
          % (np.median(tr_gaps), np.percentile(tr_gaps, 10), np.percentile(tr_gaps, 90)))
    print('L2 每个测试序列与最近的训练/验证序列（同机型同场景）的时间间隔：')
    for sq, dur, gap, sq2, sp2 in sorted(gaps):
        print('    %-22s 时长 %6.1f s | 最近 %-22s (%s) 间隔 %s' % (sq, dur, sq2, sp2,
              ('%.1f s' % gap) if np.isfinite(gap) else '无同场景序列'))
    report['L2'] = [{'test_seq': g[0], 'dur_s': g[1], 'nearest_seq': g[3], 'nearest_split': g[4], 'gap_s': g[2]} for g in gaps]

    # ---------------- L3 近重复（位姿 + 目标外观） ----------------
    F_used = frame_info(tr_idx, tr_rgb, used)
    F_test = frame_info(te_idx, te_rgb, te_idx['valid_idx'])
    L3 = {}
    nn_test = {}
    for cls in sorted(set(f['cls'] for f in F_test)):
        A = [f for f in F_test if f['cls'] == cls]
        B = [f for f in F_used if f['cls'] == cls]
        dpos, dang = pose_nn(A, B)
        seq_b = np.array([f['seq'] for f in B])
        same = np.stack([seq_b == f['seq'] for f in B])
        np.fill_diagonal(same, True)
        bpos_x, bang_x = pose_nn(B, B, same_seq_mask=same)           # 训练帧 -> 其他训练序列
        ca = np.stack([crop_vec(te_rgb, f) for f in A])
        cb = np.stack([crop_vec(tr_rgb, f) for f in B])
        sim_te = (ca @ cb.T).max(1)
        sim_bb = cb @ cb.T
        sim_cross = np.where(same, -np.inf, sim_bb).max(1)
        diffseq_or_self = ~same | np.eye(len(B), dtype=bool)
        sim_within = np.where(diffseq_or_self, -np.inf, sim_bb).max(1)
        for f, a, b, s in zip(A, dpos, dang, sim_te):
            nn_test[f['i']] = (a, b, s)
        L3[cls] = {
            'n_test': len(A), 'n_used': len(B),
            'test->used 位置差中位 m': float(np.median(dpos)), 'test->used 角度差中位 °': float(np.median(dang)),
            'test 有 5cm&5° 内训练近邻的比例': float(np.mean((dpos < 0.05) & (dang < 5))),
            '对照 used->其他训练序列 位置差中位 m': float(np.median(bpos_x)), '对照 角度差中位 °': float(np.median(bang_x)),
            '对照 used 有 5cm&5° 内其他序列近邻的比例': float(np.mean((bpos_x < 0.05) & (bang_x < 5))),
            'test->used 目标裁块相似度中位': float(np.median(sim_te)),
            '对照 used->其他训练序列 相似度中位': float(np.median(sim_cross)),
            '对照 used->同序列其他帧 相似度中位（真近重复水平）': float(np.median(sim_within[np.isfinite(sim_within)])),
            'test 相似度 >= 同序列近重复中位 的比例': float(np.mean(sim_te >= np.median(sim_within[np.isfinite(sim_within)]))),
            '对照 used->其他训练序列 相似度 >= 同序列近重复中位 的比例': float(np.mean(sim_cross >= np.median(sim_within[np.isfinite(sim_within)]))),
        }
        print('L3 [%s]' % cls)
        for k, v in L3[cls].items():
            print('    %-44s %s' % (k, ('%.4f' % v) if isinstance(v, float) else v))
    report['L3'] = L3

    # ---------------- O1 训练帧 / 验证 / 测试 ----------------
    O1 = {}
    for name, split, iv in (('训练实际用到的帧', 'train', args.interval), ('验证集', 'val', 1), ('测试集', 'test', 1)):
        s, recs, vidx = evaluate(args.tag, split, iv)
        O1[name] = {k: s.get(k) for k in ('n_gt', 'det_rate', 'pos_median', 'z_median', 'ang_median', 'ACC_pos_0.1',
                                          'ACC_rot_10', 'ACC_0.1m10deg', 'score')}
        print('O1 %-10s %s' % (name, pose_eval.format_summary(name, s)))
        if split == 'test':
            test_recs, test_vidx = recs, vidx
    report['O1'] = O1

    # ---------------- L4 误差 vs 近邻距离 ----------------
    rows = []
    for r, i in zip(test_recs, test_vidx):
        g, p = r['gt'][0], r['pred'][int(np.argmax(r['conf']))]
        pe = float(np.linalg.norm(p[:3] - g[:3]))
        ae = float(np.degrees((R.from_euler('xyz', p[6:9]).inv() * R.from_euler('xyz', g[6:9])).magnitude()))
        a, b, s = nn_test[i]
        rows.append((a / 0.1 + b / 10.0, s, pe, ae))
    rows = np.array(rows)
    qs = np.percentile(rows[:, 0], [0, 25, 50, 75, 100])
    print('L4 按「最近训练近邻的位姿距离」四分位分箱的测试误差（近邻越近误差越小才可疑）：')
    L4 = []
    for k in range(4):
        sel = (rows[:, 0] >= qs[k]) & (rows[:, 0] <= qs[k + 1])
        L4.append({'nn_cost_range': [float(qs[k]), float(qs[k + 1])], 'n': int(sel.sum()),
                   'pos_median': float(np.median(rows[sel, 2])), 'ang_median': float(np.median(rows[sel, 3]))})
        print('    近邻代价 %.2f~%.2f  %4d 帧  位置误差中位 %.3f m  角度误差中位 %.1f°'
              % (qs[k], qs[k + 1], sel.sum(), np.median(rows[sel, 2]), np.median(rows[sel, 3])))
    from scipy.stats import spearmanr
    rho_p = spearmanr(rows[:, 0], rows[:, 2]).correlation
    rho_s = spearmanr(rows[:, 1], rows[:, 2]).correlation
    print('    Spearman(近邻位姿距离, 位置误差) = %.3f；Spearman(裁块相似度, 位置误差) = %.3f' % (rho_p, rho_s))
    report['L4'] = {'bins': L4, 'spearman_nnpose_poserr': float(rho_p), 'spearman_cropsim_poserr': float(rho_s)}

    # ---------------- O2 验证曲线 ----------------
    h = json.load(open(os.path.join(ROOT, 'output', 'models', 'uavdet_3d', 'camnorm', 'mav6d', args.tag, 'val_history.json'),
                       encoding='utf-8'))
    report['O2'] = [{'epoch': x['epoch'], 'score': x['score'], 'pos_median': x.get('pos_median'),
                     'ang_median': x.get('ang_median')} for x in h]
    print('O2 验证曲线:', ', '.join('%d:%.3f' % (x['epoch'], x['score']) for x in h))
    json.dump(report, open(os.path.join(args.out, '%s.json' % args.tag), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False, default=float)
    print('写出', os.path.join(args.out, '%s.json' % args.tag))


if __name__ == '__main__':
    main()
