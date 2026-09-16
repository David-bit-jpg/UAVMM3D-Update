# -*- coding: utf-8 -*-
"""nose_det：合并 run_heading.py 的多次运行（3 个真实微调模型 x 2 个裁窗），按 n 加权 = 逐样本直接拼接。

    python annot_audit/nose_det/merge_heading.py --tags T_p025_s0,T_p010_s0,T_p005_s0 --zooms 1.0,2.5 --control S1
输出 heading_merged.md / heading_merged.json 到 OUT_ROOT。
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nose_common import OUT_ROOT, table, stats   # noqa: E402


def load(tag, zoom, out):
    p = os.path.join(out, 'heading_%s_z%s.csv' % (tag, str(zoom).replace('.', 'p')))
    if not os.path.exists(p):
        return None
    rows = []
    with open(p, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r['matched'] = int(r['matched'])
            for k in ('yaw', 'e2d_px', 'conf', 'tilt', 'ang', 'gt_z', 'gt_maxdim'):
                r[k] = float(r[k]) if r[k] not in ('', 'nan') else np.nan
            rows.append(r)
    return rows


def per_type(rows, key='name'):
    per = {}
    for r in rows:
        if r['matched'] and np.isfinite(r['yaw']):
            per.setdefault(r[key], []).append(r['yaw'])
    return per


def verdict(s, others_pm):
    """判定规则（任务给定）：±(60~120) 占比明显高于其他机型且 |中位| > 45° -> 约差 90°（给符号）；|中位| < 30° -> 一致。"""
    if s['n'] == 0:
        return '无样本'
    if abs(s['median']) < 30:
        return '一致（标签 x = 视觉机头）'
    ref = np.median(others_pm) if len(others_pm) else 0.0
    if s['pm60_120'] > ref + 0.15 and abs(s['median']) > 45:
        sign = '+' if s['median'] > 0 else '-'
        return '约差 90°：视觉机头 = 标签 x 绕机体 z 转 %s90°（中位 %+.0f°，±(60~120) 占 %.0f%% vs 其他机型中位 %.0f%%）' % (
            sign, s['median'], 100 * s['pm60_120'], 100 * ref)
    return '不明（中位 %+.0f°，±(60~120) 占 %.0f%%，>150 占 %.0f%%）' % (s['median'], 100 * s['pm60_120'], 100 * s['over150'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='T_p025_s0,T_p010_s0,T_p005_s0')
    ap.add_argument('--zooms', default='1.0,2.5')
    ap.add_argument('--control', default='S1')
    ap.add_argument('--out', default=OUT_ROOT)
    args = ap.parse_args()
    tags = args.tags.split(',')
    zooms = args.zooms.split(',')
    md = ['# nose_det：标签 x 轴是否指向视觉机头 —— 仿真测试集上的合并统计', '',
          '口径：真实微调模型（MAV6D 上学到「x = 视觉机头」）在仿真测试集上预测的机体 x 轴，投到 GT 机体 z 的水平面，'
          '与标签 x 的带符号夹角 yaw = atan2((x_gt × x_pred)·z_gt, x_gt·x_pred)。'
          'yaw>0 = 视觉机头在标签 x 的逆时针方向（俯视，机体 z 朝上）。每个 GT 取 2D 最近的检测，中心误差 > 0.5 倍粗略表观大小的不计。',
          '机型归属按 batch GT 框精确匹配回 meta（原 diag 脚本按位置 zip，裁窗丢框时多机帧会错位）。', '']
    summary = {'runs': {}, 'merged': {}, 'control': {}, 'verdict': {}}
    merged_rows = []
    med_matrix = {}
    for tag in tags:
        for z in zooms:
            rows = load(tag, z, args.out)
            if rows is None:
                md.append('- 缺 %s zoom %s 的结果' % (tag, z))
                continue
            merged_rows += rows
            per = per_type(rows)
            n_gt = len(rows)
            n_det = sum(r['matched'] for r in rows)
            n_mis = sum(1 for r in rows if r['name'] != r['name_zip'])
            md += ['## %s · 裁窗 %s 倍：GT %d，2D 对上 %d（%.1f%%），zip 错位 %d（%.1f%%）'
                   % (tag, z, n_gt, n_det, 100.0 * n_det / max(n_gt, 1), n_mis, 100.0 * n_mis / max(n_gt, 1)), '',
                   '```', table(per, ''), '```', '']
            summary['runs']['%s_z%s' % (tag, z)] = {k: stats(v) for k, v in per.items()}
            for k, v in per.items():
                med_matrix.setdefault(k, {})['%s_z%s' % (tag, z)] = stats(v)['median']
    per_m = per_type(merged_rows)
    md += ['## 合并（%s x 裁窗 %s，按 n 加权 = 逐样本拼接）' % ('+'.join(tags), '/'.join(zooms)), '',
           '```', table(per_m, ''), '```', '']
    summary['merged'] = {k: stats(v) for k, v in per_m.items()}
    # 各次运行的中位一致性
    md += ['### 各次运行的带符号偏航差中位（°）', '', '| 机型 | ' + ' | '.join(sorted(next(iter(med_matrix.values())).keys())) + ' | 合并 |',
           '|---|' + '---|' * (len(next(iter(med_matrix.values()))) + 1)]
    for k in sorted(med_matrix):
        cols = sorted(med_matrix[k].keys())
        md.append('| %s | ' % k + ' | '.join('%+.0f' % med_matrix[k][c] for c in cols) + ' | %+.0f |' % summary['merged'][k]['median'])
    md.append('')
    # 判定
    md += ['### 判定（规则：±(60~120) 占比明显高于其他机型且 |中位|>45° -> 约差 90°；|中位|<30° -> 一致）', '']
    for k in sorted(per_m):
        s = summary['merged'][k]
        others = [summary['merged'][o]['pm60_120'] for o in per_m if o != k]
        v = verdict(s, others)
        summary['verdict'][k] = v
        md.append('- **%s**（n=%d）：中位 %+.1f°，<30° %.0f%%，±(60~120)° %.0f%%（+ %.0f%% / - %.0f%%），>150° %.0f%% -> %s'
                  % (k, s['n'], s['median'], 100 * s['within30'], 100 * s['pm60_120'], 100 * s['pos60_120'],
                     100 * s['neg60_120'], 100 * s['over150'], v))
    md.append('')
    # 对照
    for z in zooms:
        rows = load(args.control, z, args.out)
        if rows is None:
            md.append('- 缺对照 %s zoom %s' % (args.control, z))
            continue
        per = per_type(rows)
        n_gt = len(rows)
        n_det = sum(r['matched'] for r in rows)
        md += ['## 对照 %s（纯仿真）· 裁窗 %s 倍：GT %d，2D 对上 %d（%.1f%%）' % (args.control, z, n_gt, n_det, 100.0 * n_det / max(n_gt, 1)),
               '', '```', table(per, ''), '```', '']
        summary['control']['%s_z%s' % (args.control, z)] = {k: stats(v) for k, v in per.items()}
    with open(os.path.join(args.out, 'heading_merged.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(args.out, 'heading_merged.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
