# -*- coding: utf-8 -*-
"""训练域 x 测试域 交叉表：纯仿真 / 纯真实 模型分别在 仿真测试集 与 MAV6D 真实测试集上的 LAA3D_ADS 指标。

    python table_cross_domain.py [--md ../docs/results/cross_domain_table.md]

指标全部取自 eval_camnorm.py 写出的 JSON：LAA3D_ADS 原版报告（两套口径）+ 类别无关位姿指标（分母 = 全部 GT）。
"""
import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J1 = os.path.join(ROOT, 'output', 'camnorm', 'json')
J2 = os.path.join(ROOT, 'output', 'camnorm', 'json_simtest')

ROWS = [
    ('纯仿真 S1', '仿真测试集（整幅）', os.path.join(J2, 'S1__simtest_z1.json')),
    ('纯仿真 S1', '仿真测试集（2.5 倍长焦裁窗）', os.path.join(J2, 'S1__simtest_z2.5.json')),
    ('纯仿真 S1', 'MAV6D 真实测试集（零样本）', os.path.join(J1, 'Z_S1_test.json')),
    ('纯真实 25% 种子0', '仿真测试集（整幅）', os.path.join(J2, 'C_p025_s0__simtest_z1.json')),
    ('纯真实 25% 种子0', 'MAV6D 真实测试集', os.path.join(J1, 'C_p025_s0_test.json')),
]


def parse_ads(txt):
    def num(pat, s):
        m = re.search(pat, s)
        return float(m.group(1)) if m else float('nan')
    head, _, rest = txt.partition('detection 3D AP:')
    ap3d, _, ap2d = rest.partition('detection 2D AP:')
    return {
        'orin_med': num(r'orin_error_median\(degree\)[:：]\s*([0-9.eE+-]+)', head),
        'posi_med': num(r'posi_error_median\(m\)[:：]\s*([0-9.eE+-]+)', head),
        'size_med': num(r'size_error_median\(m\)[:：]\s*([0-9.eE+-]+)', head),
        'orin_acc': num(r'orin_accuracy\(%\)[:：]\s*([0-9.eE+-]+)', head),
        'posi_acc': num(r'posi_accuracy\(%\)[:：]\s*([0-9.eE+-]+)', head),
        'size_acc': num(r'size_accuracy\(%\)[:：]\s*([0-9.eE+-]+)', head),
        'ap3d': num(r'AP_R40_eval_distance_\d+\(%\)[:：]\s*([0-9.eE+-]+)', ap3d),
        'ap2d': num(r'AP_R40_eval_distance_\d+\(%\)[:：]\s*([0-9.eE+-]+)', ap2d),
        'ads': num(r'LAA3D_ADS_drone \(%\)\s*:\s*([0-9.]+)', txt),
    }


def fmt(v, d=1):
    return '-' if v != v else ('%.' + str(d) + 'f') % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=os.path.join(ROOT, 'docs', 'results', 'cross_domain_table.md'))
    args = ap.parse_args()
    lines = ['# 训练域 × 测试域：LAA3D_ADS 指标', '',
             '仿真测试集 = 新室内仿真 test 划分（与训练序列不重叠，4814 帧，多目标）；MAV6D 真实测试集 = 官方 test 4800 帧。',
             '纯仿真 S1 从未见过真实数据；纯真实 = 只用 25% MAV6D 训练帧从零训练。两个模型结构完全相同。', '']
    for prof, pname in (('indoor', '室内口径（3D AP 匹配 0.05/0.1/0.2/0.5 m，位置归一化上限 1 m，朝向 30°，尺寸 0.2 m）'),
                        ('laa', '原口径 laa（与 laam6d.yaml 相同：3D AP 匹配 1/2/4/6 m，位置上限 8 m，朝向 30°，尺寸 1 m）')):
        lines += ['## LAA3D_ADS · %s' % pname, '',
                  '| 模型 | 测试集 | ADS | orin_error_median (°) | posi_error_median (m) | size_error_median (m) | '
                  'orin_acc (%) | posi_acc (%) | size_acc (%) | 3D AP_R40 (%) | 2D AP_R40 (%) |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for model, test, path in ROWS:
            if not os.path.exists(path):
                lines.append('| %s | %s | 未完成 | | | | | | | | |' % (model, test))
                continue
            d = json.load(open(path, encoding='utf-8'))
            a = parse_ads((d.get('ads_reports') or {}).get(prof, ''))
            lines.append('| %s | %s | **%s** | %s | %s | %s | %s | %s | %s | %s | %s |'
                         % (model, test, fmt(a['ads'], 2), fmt(a['orin_med']), fmt(a['posi_med'], 3), fmt(a['size_med'], 3),
                            fmt(a['orin_acc']), fmt(a['posi_acc']), fmt(a['size_acc']), fmt(a['ap3d']), fmt(a['ap2d'])))
        lines.append('')
    lines += ['## 位姿指标（类别无关，分母 = 全部 GT）', '',
              '| 模型 | 测试集 | GT 数 | 检出率 | 位置误差中位 (m) | 深度误差中位 (m) | 角度误差中位 (°) | 位置<10cm | 朝向<10° | 位置<10cm 且 朝向<10° |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for model, test, path in ROWS:
        if not os.path.exists(path):
            lines.append('| %s | %s | 未完成 | | | | | | | |' % (model, test))
            continue
        d = json.load(open(path, encoding='utf-8'))
        lines.append('| %s | %s | %d | %.1f%% | %.3f | %.3f | %.1f | %.1f%% | %.1f%% | %.1f%% |'
                     % (model, test, d['n_gt'], 100 * d['det_rate'], d['pos_median'], d['z_median'], d['ang_median'],
                        100 * d['ACC_pos_0.1'], 100 * d['ACC_rot_10'], 100 * d['ACC_0.1m10deg']))
    txt = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    open(args.md, 'w', encoding='utf-8').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
