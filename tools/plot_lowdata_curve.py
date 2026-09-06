# -*- coding: utf-8 -*-
"""把低数据量迁移曲线画出来：B(仿真预训练+微调) vs C(目标域从零) vs A(零样本下限)。

读 tools/../output/bench_json/*.json（由 eval_on_mav6d.py --json 产出）。

用法：
    python tools/plot_lowdata_curve.py --out ../output/lowdata_curve.png
"""
import argparse
import json
import os
import sys

import numpy as np

# 训练集降采样间隔 -> 实际使用的目标域数据比例
FRACS = [('p01', 100, 1.0), ('p05', 20, 5.0), ('p10', 10, 10.0),
         ('p25', 4, 25.0), ('full', 1, 100.0)]
# 100% 那两个臂是之前跑的，tag 名不同
TAG = {('B', 'full'): 'B_transfer', ('C', 'full'): 'C_real_only'}


def load(js_dir, tag):
    p = os.path.join(js_dir, tag + '.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def run_custom(args, arms, fracs):
    """任意多臂的表 + 图（K0/K1/K2 这种命名：<前缀>_<档位>）。"""
    n_train_full = 15202
    xs = [pct for _, _, pct in fracs]
    table = {}
    for prefix, label in arms:
        table[prefix] = [load(args.json_dir, '%s_%s' % (prefix, key)) for key, _, _ in fracs]
    unit = 'm' if args.metric.startswith(('pos', 'z')) else ('deg' if 'ang' in args.metric else '')
    better_low = not args.metric.startswith('acc')
    print('\n目标域数据量 -> %s (%s)' % (args.metric, unit))
    head = '%-8s %-8s' % ('比例', '帧数') + ''.join('%14s' % l for _, l in arms)
    print(head); print('-' * len(head))
    for i, (key, iv, pct) in enumerate(fracs):
        vals = [table[p][i].get(args.metric) if table[p][i] else None for p, _ in arms]
        row = '%-8s %-8d' % ('%.0f%%' % pct, int(round(n_train_full * pct / 100.0)))
        best = (min if better_low else max)([v for v in vals if v is not None], default=None)
        for v in vals:
            cell = '-' if v is None else ('%.4f' % v + ('*' if v == best else ' '))
            row += '%14s' % cell
        print(row)
    print('(* = 该档位最优)')
    for tag, label in (('A_sim_only', 'A 纯仿真零样本'), ('Z0_rgb', 'Z0 S0 零样本'),
                       ('Z1_kdmm', 'Z1 KD-mm 零样本'), ('Z2_kdrgb', 'Z2 KD-rgb 零样本')):
        d = load(args.json_dir, tag)
        if d and args.metric in d:
            print('%-18s %.4f' % (label, d[args.metric]))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('(未画图: %s)' % e); return 0
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for (prefix, label), color in zip(arms, ('tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')):
        pts = [(x, d.get(args.metric)) for x, d in zip(xs, table[prefix]) if d and d.get(args.metric) is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=color, lw=2, ms=6, label=label)
    # 旧的 C（从零）当参照虚线
    ref = [(x, load(args.json_dir, 'C_%s' % key)) for (key, _, x) in fracs]
    ref = [(x, d.get(args.metric)) for x, d in ref if d]
    if ref:
        ax.plot([r[0] for r in ref], [r[1] for r in ref], 's--', color='gray', lw=1.2, ms=4, label='C: target only (scratch)')
    ax.set_xscale('log'); ax.set_xticks(xs); ax.set_xticklabels(['%g%%' % x for x in xs])
    ax.set_xlabel('target-domain training data'); ax.set_ylabel(args.metric); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(args.title or ('MAV6D transfer: %s' % args.metric))
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150); print('\n图写出 %s' % args.out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', default='../output/bench_json')
    ap.add_argument('--out', default='../output/lowdata_curve.png')
    ap.add_argument('--metric', default='pos_median',
                    choices=['pos_median', 'ang_median', 'acc_0.2', 'z_median'])
    ap.add_argument('--arms', default=None,
                    help='自定义臂，格式 "K0:S0 plain,K1:KD-mm,K2:KD-rgb"；缺省画 B/C 两臂')
    ap.add_argument('--fracs', default=None,
                    help='自定义档位键，如 "p01,p05,p10"；缺省全部五档')
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    # 自定义臂 -> 直接改写 series 的构造方式
    if args.arms:
        arms = [a.split(':', 1) for a in args.arms.split(',')]
        fracs = [f for f in FRACS if (args.fracs is None or f[0] in args.fracs.split(','))]
        return run_custom(args, arms, fracs)

    n_train_full = 15202
    xs, series = [], {'B': [], 'C': []}
    for key, iv, pct in FRACS:
        xs.append(pct)
        for arm in ('B', 'C'):
            tag = TAG.get((arm, key), '%s_%s' % (arm, key))
            d = load(args.json_dir, tag)
            series[arm].append(None if d is None else d.get(args.metric))

    a = load(args.json_dir, 'A_sim_only')
    probe = load(args.json_dir, 'probe_frozen_full')

    # ---- 文本表 ----
    unit = 'm' if args.metric.startswith(('pos', 'z')) else ('deg' if 'ang' in args.metric else '')
    print('\n目标域数据量 -> %s (%s)' % (args.metric, unit))
    print('%-12s %-10s %12s %12s %10s' % ('比例', '帧数', 'B 迁移', 'C 从零', 'B 相对增益'))
    print('-' * 60)
    for i, (key, iv, pct) in enumerate(FRACS):
        b, c = series['B'][i], series['C'][i]
        n = int(round(n_train_full * pct / 100.0))
        # acc_* 是越大越好，误差类是越小越好，增益的符号要跟着变
        sign = -1.0 if args.metric.startswith('acc') else 1.0
        gain = '' if (b is None or c is None or c == 0) else '%+.1f%%' % (sign * 100.0 * (c - b) / c)
        print('%-12s %-10d %12s %12s %10s' % (
            '%.0f%%' % pct, n,
            '-' if b is None else '%.4f' % b,
            '-' if c is None else '%.4f' % c, gain))
    if a:
        print('\nA 纯仿真零样本 (下限): %.4f' % a.get(args.metric, float('nan')))
    if probe:
        print('冻结骨干线性探针 (100%% 数据): %.4f' % probe.get(args.metric, float('nan')))

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('\n(未画图: %s)' % e)
        return 0

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for arm, color, lab in (('B', 'tab:red', 'B: sim pretrain -> finetune'),
                            ('C', 'tab:blue', 'C: target only (scratch)')):
        x = [p for p, v in zip(xs, series[arm]) if v is not None]
        y = [v for v in series[arm] if v is not None]
        if x:
            ax.plot(x, y, 'o-', color=color, label=lab, lw=2, ms=6)
    if a is not None and args.metric in a:
        ax.axhline(a[args.metric], color='gray', ls='--', lw=1.2,
                   label='A: sim zero-shot (%.2f)' % a[args.metric])
    if probe is not None and args.metric in probe:
        ax.plot([100.0], [probe[args.metric]], '*', color='tab:green', ms=14,
                label='frozen-backbone probe')
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels(['%g%%' % p for p in xs])
    ax.set_xlabel('target-domain training data')
    ax.set_ylabel(args.metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Sim -> Real transfer on MAV6D: %s' % args.metric)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print('\n图写出 %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
