# -*- coding: utf-8 -*-
"""把 output/ads_metrics/<tag>/<profile>/report.txt 里的 LAA3D_ADS 指标汇成一张表。
    D:/Miniconda3/envs/city/python.exe tools/print_ads_table.py --profile indoor --out docs/results/ads_table_indoor.md
"""
import argparse
import os
import re

ROWS = [
    ('Z_M0',        '只用仿真（3000 生成 + 1436 原版）',        '仿真 4436 / 真实 0'),
    ('Z_MT',        '只用仿真 + 多任务',                        '仿真 4436 / 真实 0'),
    ('Z_S1',        '只用仿真 + 蒸馏',                          '仿真 4436 / 真实 0'),
    ('Z_S1MT',      '只用仿真 + 蒸馏 + 多任务',                 '仿真 4436 / 真实 0'),
    ('C_p01',       '1% 真实从零',                              '仿真 0 / 真实 152'),
    ('M0_p01',      '1% 仿真预训练 + 微调',                     '仿真 4436 / 真实 152'),
    ('S1_p01',      '1% 蒸馏预训练 + 微调',                     '仿真 4436 / 真实 152'),
    ('S1MT_p01',    '1% 蒸馏+多任务预训练 + 微调',              '仿真 4436 / 真实 152'),
    ('Cn_p05',      '5% 真实从零',                              '仿真 0 / 真实 761'),
    ('B_p05',       '5% near15 仿真预训练 + 微调',              '仿真 near15 / 真实 761'),
    ('M0_p05',      '5% 仿真预训练 + 微调',                     '仿真 4436 / 真实 761'),
    ('MT_p05',      '5% 多任务预训练 + 微调',                   '仿真 4436 / 真实 761'),
    ('S1_p05',      '5% 蒸馏预训练 + 微调',                     '仿真 4436 / 真实 761'),
    ('S1MT_p05',    '5% 蒸馏+多任务预训练 + 微调',              '仿真 4436 / 真实 761'),
    ('RM0_p05',     '5% 仿真预训练 + 保留式微调',               '仿真 4436 / 真实 761'),
    ('ST_S1MT_p05', '5% 蒸馏+多任务 + 微调 + 自训练',           '仿真 4436 / 真实 761+14346 无标签'),
    ('ST_Cn_p05',   '5% 真实从零 + 自训练',                     '仿真 0 / 真实 761+14364 无标签'),
    ('C_p10',       '10% 真实从零',                             '仿真 0 / 真实 1520'),
    ('M0_p10',      '10% 仿真预训练 + 微调',                    '仿真 4436 / 真实 1520'),
    ('MT_p10',      '10% 多任务预训练 + 微调',                  '仿真 4436 / 真实 1520'),
    ('S1_p10',      '10% 蒸馏预训练 + 微调',                    '仿真 4436 / 真实 1520'),
    ('S1MT_p10',    '10% 蒸馏+多任务预训练 + 微调',             '仿真 4436 / 真实 1520'),
]

PATS = {
    'orin_mean': r'orin_error_mean\(degree\)[：:]\s*([\d.eE+-]+)',
    'posi_mean': r'posi_error_mean\(m\):\s*([\d.eE+-]+)',
    'size_mean': r'size_error_mean\(m\):\s*([\d.eE+-]+)',
    'orin_med': r'orin_error_median\(degree\):\s*([\d.eE+-]+)',
    'posi_med': r'posi_error_median\(m\):\s*([\d.eE+-]+)',
    'size_med': r'size_error_median\(m\):\s*([\d.eE+-]+)',
    'orin_acc': r'orin_accuracy\(%\):\s*([\d.eE+-]+)',
    'posi_acc': r'posi_accuracy\(%\):\s*([\d.eE+-]+)',
    'size_acc': r'size_accuracy\(%\):\s*([\d.eE+-]+)',
    'ads': r'LAA3D_ADS_drone\s*\(%\)\s*:\s*([\d.eE+-]+)',
}


def parse(path):
    t = open(path, encoding='utf-8').read()
    d = {}
    for k, p in PATS.items():
        m = re.search(p, t)
        d[k] = float(m.group(1)) if m else float('nan')
    ap3 = re.findall(r'AP_R40_matching_distance_threshold_([\d.]+)\(%\):\s*([\d.eE+-]+)', t)
    d['ap3d'] = [(a, float(b)) for a, b in ap3]
    m = re.search(r'detection 3D AP:\s*\n\s*AP_R40_eval_distance_\d+\(%\):\s*([\d.eE+-]+)', t)
    d['ap3d_all'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'Recall_eval_distance_\d+\(%\):\s*([\d.eE+-]+)', t)
    d['recall'] = float(m.group(1)) if m else float('nan')
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='output/ads_metrics')
    ap.add_argument('--profile', default='indoor')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    got = []
    for tag, name, data in ROWS:
        p = os.path.join(a.root, tag, a.profile, 'report.txt')
        if os.path.exists(p):
            got.append((tag, name, data, parse(p)))
    if not got:
        print('没有报告'); return
    ths = [t for t, _ in got[0][3]['ap3d']]
    L = []
    L.append('测试集：MAV6D test 4800 帧（mavic2 1290 + phantom4 3510），全程未参与训练。类别无关（统一记作 drone）。')
    L.append('指标口径 = %s%s' % (a.profile, '：AP3D 匹配阈值 %s m，位置归一化上限 1 m、朝向 30°、尺寸 0.2 m' % '/'.join(ths)
             if a.profile == 'indoor' else '：与源域 laam6d.yaml 同参数（AP3D 匹配阈值 %s m，位置归一化上限 8 m）' % '/'.join(ths)))
    L.append('')
    hdr = ['实验', '训练数据（帧）', 'ADS', 'Recall2D', 'AP3D'] + ['AP3D@%sm' % t for t in ths] + \
          ['posi_err_med(m)', 'orin_err_med(°)', 'size_err_med(m)', 'posi_acc', 'orin_acc', 'size_acc']
    L.append('| ' + ' | '.join(hdr) + ' |')
    L.append('|' + '---|' * len(hdr))
    for tag, name, data, d in got:
        row = [name, data, '**%.2f**' % d['ads'], '%.1f' % d['recall'], '%.2f' % d['ap3d_all']]
        row += ['%.2f' % v for _, v in d['ap3d']]
        row += ['%.3f' % d['posi_med'], '%.1f' % d['orin_med'], '%.3f' % d['size_med'],
                '%.1f' % d['posi_acc'], '%.1f' % d['orin_acc'], '%.1f' % d['size_acc']]
        L.append('| ' + ' | '.join(row) + ' |')
    txt = '\n'.join(L)
    print(txt)
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(txt + '\n')
        print('\n写出 %s' % a.out)


if __name__ == '__main__':
    main()
