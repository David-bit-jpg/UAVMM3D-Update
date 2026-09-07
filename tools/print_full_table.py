# -*- coding: utf-8 -*-
"""把 output/bench_json/full_*.json 打成一张「实验名 + 训练数据 + 性能指标」的表（Markdown）。
    D:/Miniconda3/envs/city/python.exe tools/print_full_table.py [--out docs/results/full_table.md]
"""
import argparse
import json
import os

# 实验名 -> (中文说明, 训练数据)
ROWS = [
    ('Z_M0',        '只用仿真：3000 生成 + 1436 原版，纯 RGB',           '仿真 4436 / 真实 0'),
    ('Z_MT',        '只用仿真：同上 + 多任务',                            '仿真 4436 / 真实 0'),
    ('Z_S1',        '只用仿真：同上 + 多模态蒸馏',                        '仿真 4436 / 真实 0'),
    ('Z_S1MT',      '只用仿真：同上 + 蒸馏 + 多任务',                     '仿真 4436 / 真实 0'),
    ('C_p01',       '1% 真实从零',                                        '仿真 0 / 真实 152'),
    ('M0_p01',      '1% 仿真预训练 + 真实微调',                           '仿真 4436 / 真实 152'),
    ('S1_p01',      '1% 蒸馏预训练 + 真实微调',                           '仿真 4436 / 真实 152'),
    ('S1MT_p01',    '1% 蒸馏+多任务预训练 + 真实微调',                    '仿真 4436 / 真实 152'),
    ('C_p05',       '5% 真实从零（旧流程）',                              '仿真 0 / 真实 761'),
    ('Cn_p05',      '5% 真实从零（现行流程）',                            '仿真 0 / 真实 761'),
    ('B_p05',       '5% near15 仿真预训练 + 真实微调',                    '仿真 near15 / 真实 761'),
    ('M0_p05',      '5% 仿真预训练 + 真实微调',                           '仿真 4436 / 真实 761'),
    ('MT_p05',      '5% 多任务预训练 + 真实微调',                         '仿真 4436 / 真实 761'),
    ('S1_p05',      '5% 蒸馏预训练 + 真实微调',                           '仿真 4436 / 真实 761'),
    ('S1MT_p05',    '5% 蒸馏+多任务预训练 + 真实微调',                    '仿真 4436 / 真实 761'),
    ('RM0_p05',     '5% 仿真预训练 + 保留式微调',                         '仿真 4436 / 真实 761'),
    ('ST_S1MT_p05', '5% 蒸馏+多任务 + 微调 + 自训练',                     '仿真 4436 / 真实 761 有标签 + 14346 无标签'),
    ('ST_Cn_p05',   '5% 真实从零 + 自训练',                               '仿真 0 / 真实 761 有标签 + 14364 无标签'),
    ('C_p10',       '10% 真实从零',                                       '仿真 0 / 真实 1520'),
    ('M0_p10',      '10% 仿真预训练 + 真实微调',                          '仿真 4436 / 真实 1520'),
    ('MT_p10',      '10% 多任务预训练 + 真实微调',                        '仿真 4436 / 真实 1520'),
    ('S1_p10',      '10% 蒸馏预训练 + 真实微调',                          '仿真 4436 / 真实 1520'),
    ('S1MT_p10',    '10% 蒸馏+多任务预训练 + 真实微调',                   '仿真 4436 / 真实 1520'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', default='output/bench_json')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    L = []
    L.append('测试集：MAV6D test 4800 帧，全程未参与训练。准确率的分母是全部 4800 帧，没检出算失败。')
    L.append('')
    L.append('| 实验 | 训练数据（帧） | 检出率 | 位置<0.1m | 位置<0.2m | 朝向<10° | 朝向<20° | 0.1m&10° | ADD<10%直径 | 位置中位 m | 朝向中位 ° |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for tag, name, data in ROWS:
        p = os.path.join(a.json_dir, 'full_%s.json' % tag)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        L.append('| %s | %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.3f | %.1f |' % (
            name, data, 100 * d['det_rate'], 100 * d['ACC_pos_0.1'], 100 * d['ACC_pos_0.2'],
            100 * d['ACC_rot_10'], 100 * d['ACC_rot_20'], 100 * d['ACC_0.1m10deg'],
            100 * d['ACC_add_10'], d['pos_median'], d['ang_median']))
    txt = '\n'.join(L)
    print(txt)
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(txt + '\n')
        print('\n写出 %s' % a.out)


if __name__ == '__main__':
    main()
