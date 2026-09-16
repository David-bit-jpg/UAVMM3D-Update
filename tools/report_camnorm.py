# -*- coding: utf-8 -*-
"""汇总 camnorm 全量实验：纯真实（从零）/ 纯仿真（零样本）/ 迁移，各预算档 3 种子均值 ± 标准差。

    python report_camnorm.py [--md ../docs/results/camnorm_sweep.md]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, 'output', 'camnorm', 'json')
BUDGETS = [1, 5, 10, 25, 50, 100]
N_TRAIN = 12866          # MAV6D camnorm train 帧数（去掉验证序列后）

ROWS = [
    ('det_rate', '检出率', '%', 100),
    ('pos_median', '位置误差中位', 'm', 1),
    ('z_median', '深度误差中位', 'm', 1),
    ('ang_median', '角度误差中位', '°', 1),
    ('ACC_pos_0.1', '位置<10cm（全部帧）', '%', 100),
    ('ACC_rot_10', '朝向<10°（全部帧）', '%', 100),
    ('ACC_0.1m10deg', '位置<10cm且朝向<10°', '%', 100),
    ('ads_indoor', 'ADS 室内口径', '', 1),
    ('ads_laa', 'ADS 原口径(laa)', '', 1),
]


def load(tag):
    p = os.path.join(JS, '%s_test.json' % tag)
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding='utf-8'))
    for prof, txt in (d.get('ads_reports') or {}).items():
        m = re.search(r'LAA3D_ADS_drone \(%\) : ([0-9.]+)', txt)
        if m:
            d['ads_' + prof] = float(m.group(1))
    return d


def cell(vals, scale, digits):
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return '-'
    if len(vals) == 1:
        return ('%.' + str(digits) + 'f') % vals[0]
    return ('%.' + str(digits) + 'f±%.' + str(digits) + 'f') % (np.mean(vals), np.std(vals, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=os.path.join(ROOT, 'docs', 'results', 'camnorm_sweep.md'))
    args = ap.parse_args()
    z = load('Z_S1')
    lines = ['# camnorm 全量实验：纯真实 / 纯仿真 / 迁移（MAV6D test 4800 帧，类别无关）', '',
             '每格 = 3 个种子的均值±标准差；每个种子按验证集（MAV6D 训练序列留出 15%）选轮，测试集只测一次。',
             '纯仿真 = 只在新室内仿真上训练、不看任何真实数据的零样本结果（与预算无关）。',
             '位置/深度/角度误差中位只在检出的帧上算；带「全部帧」的准确率以 4800 帧为分母，没检出算失败。', '']
    head = '| 指标 | 臂 | ' + ' | '.join('%d%%（%d 帧）' % (b, len(range(0, N_TRAIN, max(1, 100 // b)))) for b in BUDGETS) + ' |'
    sep = '|' + '---|' * (len(BUDGETS) + 2)
    for key, name, unit, scale in ROWS:
        digits = 1 if unit in ('%', '°', '') else 3
        lines += ['### %s%s' % (name, '（%s）' % unit if unit else ''), '', head, sep]
        for arm, label in (('C', '纯真实（从零）'), ('T', '迁移（仿真预训练）')):
            cells = []
            for b in BUDGETS:
                vals = []
                for s in (0, 1, 2):
                    d = load('%s_p%03d_s%d' % (arm, b, s))
                    vals.append(d.get(key) if d else None)
                cells.append(cell(vals, scale, digits))
            lines.append('| %s | %s | %s |' % (name, label, ' | '.join(cells)))
        zc = cell([z.get(key) if z else None], scale, digits)
        lines.append('| %s | 纯仿真（零样本） | %s |' % (name, ' | '.join([zc] * len(BUDGETS))))
        lines.append('')
    # 原版 ADS 打印：零样本 + 5% / 100% 档种子 0
    lines += ['## 原版 LAA3D_ADS 打印（室内口径，种子 0）', '']
    for tag in ['Z_S1', 'C_p005_s0', 'T_p005_s0', 'C_p100_s0', 'T_p100_s0']:
        d = load(tag)
        if d and 'ads_reports' in d:
            lines += ['#### %s' % tag, '```', d['ads_reports'].get('indoor', '').strip(), '```', '']
    txt = '\n'.join(lines)
    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    open(args.md, 'w', encoding='utf-8').write(txt + '\n')
    print(txt)
    print('\n写出', args.md)


if __name__ == '__main__':
    main()
