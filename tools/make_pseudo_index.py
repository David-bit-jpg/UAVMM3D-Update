# -*- coding: utf-8 -*-
"""真实无标签帧自训练的伪标签索引（2026-09-24，用户拍板「试试真实无标签帧的自训练」）。

输入：部署管线（发布版 5 集成 + 翻转 + 时序背景 + 无标签自标定尺寸 + KZ 旋转）在 MAV6D **train 划分**上的逐帧预测 dump
（annot_audit/dump_real_preds.py --split train）。真值标签只用于最后打印伪标签质量（诊断），**不参与筛选和训练**。

筛选（全部只用无标签信息）：
    1. 置信度：top-1 预测 conf >= --conf
    2. 时序一致性：同一序列按帧号排序，取 ±--win 帧邻域里其它帧的 top-1 位置中位数，离它 > --pos-tol m 的丢掉；邻域少于 --min-nb 帧的丢掉
       （相机固定、无人机连续飞，位置不会跳）
    3. 航向一致性（只标记，不丢帧）：与邻域航向中位数折 180 相差 > --yaw-tol 的帧记 rot_ok=False —— 位置网络自训练时 rot 权重为 0，用不到
输出：<cache>/train/index_pseudo_<tag>.pkl —— 与原 index.pkl 同结构，metas 里 boxes9d 换成伪标签（尺寸 = 自标定尺寸），
valid_idx 只含通过筛选的帧；训练配置用 DATA_CONFIG.INDEX_FILE 指向它，图像数据不变。

    python make_pseudo_index.py --dump ../output/camnorm/st/real_train_preds.pkl --tag st1
"""
import argparse
import collections
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from eval_camnorm import frame_key   # noqa: E402


def heading_deg(euler_xyz):
    """水平航向（相机 OpenCV 系，y 朝下）：机体 x 轴（机头）在水平面 x-z 上的方向角，0 = 远离相机，+90 = 朝画面右。
    注意不是 atan2(R[1,0], R[0,0])（那是绕光轴的像面内滚转，飞行时几乎恒定，2026-09-24 第一版就错在这）。"""
    Rm = R.from_euler('xyz', euler_xyz).as_matrix()
    return float(np.degrees(np.arctan2(Rm[0, 0], Rm[2, 0])))


def fold180(d):
    d = abs((d + 180.0) % 360.0 - 180.0)
    return min(d, 180.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--cache', default='@@MMCACHE@@/mav6d_cn_obj')
    ap.add_argument('--split', default='train')
    ap.add_argument('--tag', default='st1')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--win', type=int, default=3)
    ap.add_argument('--min-nb', type=int, default=2)
    ap.add_argument('--pos-tol', type=float, default=0.3)
    ap.add_argument('--yaw-tol', type=float, default=20.0)
    a = ap.parse_args()

    D = pickle.load(open(a.dump, 'rb'))
    assert D['info'].get('split', 'test') == a.split, D['info'].get('split')
    lwh = np.asarray(D['info']['calib_lwh'], np.float64)
    idx_path = os.path.join(a.cache, a.split, 'index.pkl')
    idx = pickle.load(open(idx_path, 'rb'))
    metas = idx['metas']
    key2i = {(m['seq'], m['frame']): i for i, m in enumerate(metas) if m}
    frames = D['frames']
    # top-1 预测（按置信度）
    top = {}
    for f in frames:
        if f['pred'] is None or len(f['conf']) == 0:
            continue
        j = int(np.argmax(f['conf']))
        if float(f['conf'][j]) < a.conf:
            continue
        p = np.asarray(f['pred'][j], np.float64)[:9]
        top[(f['seq'], f['frame'])] = (p, float(f['conf'][j]))
    # 按序列 / 帧号排序做时序一致性
    by_seq = collections.defaultdict(list)
    for f in frames:
        by_seq[f['seq']].append(f['frame'])
    accepted, rot_ok, reasons = {}, {}, collections.Counter()
    for seq, frs in by_seq.items():
        frs = sorted(frs, key=frame_key)
        for k, fr in enumerate(frs):
            key = (seq, fr)
            if key not in top:
                reasons['no_pred_or_low_conf'] += 1
                continue
            p, c = top[key]
            nb = [top[(seq, frs[j])][0] for j in range(max(0, k - a.win), min(len(frs), k + a.win + 1)) if j != k and (seq, frs[j]) in top]
            if len(nb) < a.min_nb:
                reasons['few_neighbors'] += 1
                continue
            nb = np.stack(nb)
            med = np.median(nb[:, :3], axis=0)
            if np.linalg.norm(p[:3] - med) > a.pos_tol:
                reasons['pos_jump'] += 1
                continue
            yaw_nb = np.median([heading_deg(q[6:9]) for q in nb])
            rot_ok[key] = fold180(heading_deg(p[6:9]) - yaw_nb) <= a.yaw_tol
            accepted[key] = p
            reasons['accepted'] += 1
    # 写索引
    new_metas = [None] * len(metas)
    valid, q_pos, q_yaw = [], [], []
    for key, p in accepted.items():
        i = key2i.get(key)
        if i is None:
            reasons['not_in_index'] += 1
            continue
        m = dict(metas[i])
        b = np.concatenate([p[:3], lwh, p[6:9]]).astype(np.float32)[None]
        gt = np.asarray(metas[i]['boxes9d'], np.float64).reshape(-1, 9)
        if len(gt):                                       # 诊断用：伪标签 vs 真值（不参与筛选）
            q_pos.append(float(np.linalg.norm(gt[0, :3] - p[:3])))
            q_yaw.append(fold180(heading_deg(gt[0, 6:9]) - heading_deg(p[6:9])))
        m['boxes9d'] = b
        m['names'] = ['drone']
        m['qualified'] = np.array([True])
        m['pseudo'] = True
        m['rot_ok'] = bool(rot_ok[key])
        new_metas[i] = m
        valid.append(i)
    out = dict(idx)
    out['metas'] = new_metas
    out['valid_idx'] = np.asarray(sorted(valid), dtype=idx['valid_idx'].dtype)
    out['pseudo'] = {'dump': os.path.abspath(a.dump), 'pipeline': D['info'].get('pipeline'), 'conf': a.conf, 'win': a.win,
                     'pos_tol': a.pos_tol, 'yaw_tol': a.yaw_tol, 'calib_lwh': lwh.tolist()}
    op = os.path.join(a.cache, a.split, 'index_pseudo_%s.pkl' % a.tag)
    with open(op + '.tmp', 'wb') as f:
        pickle.dump(out, f)
    os.replace(op + '.tmp', op)
    q_pos, q_yaw = np.asarray(q_pos), np.asarray(q_yaw)
    per_cls = collections.Counter(metas[i]['cls'] for i in valid)
    stats = {'index': op, 'n_frames': len(frames), 'n_accepted': len(valid), 'accept_rate': len(valid) / max(1, len(frames)),
             'reasons': dict(reasons), 'per_class': dict(per_cls), 'rot_ok_frac': float(np.mean([rot_ok[k] for k in accepted])) if accepted else None,
             'diag_pos_median_m': float(np.median(q_pos)) if len(q_pos) else None,
             'diag_pos_lt_0.2': float((q_pos < 0.2).mean()) if len(q_pos) else None,
             'diag_pos_lt_0.5': float((q_pos < 0.5).mean()) if len(q_pos) else None,
             'diag_yaw_fold180_median_deg': float(np.median(q_yaw)) if len(q_yaw) else None}
    with open(op.replace('.pkl', '.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
