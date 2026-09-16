# -*- coding: utf-8 -*-
"""纯仿真训练对照表：S1（原标签）/ S2（机头修正）/ S3（机头 + 尺度）/ R3（S3 + ImageNet ResNet-34）……

    cd E:/Open3DUAVDet/tools && python report_simfix.py [--md ../docs/results/simfix_table.md]

读 output/camnorm/json(_simtest) 与 output/camnorm/simfix/json 里 eval_camnorm.py 写的 JSON；
缺的臂显示 '-'。MAV6D 指标分母 = 全部 GT；ADS 是用户原来的 LAA3D_ADS 打印（室内 / laa 两个口径）。
"""
import argparse
import json
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
from table_cross_domain import parse_ads   # noqa: E402

CN = os.path.join(ROOT, 'output', 'camnorm')
ARMS = [
    ('S1', '原标签（ResNet8x）', os.path.join(CN, 'json', 'Z_S1_test.json'), os.path.join(CN, 'json_simtest', 'S1__simtest_z1.json')),
    ('S2', '机头修正', os.path.join(CN, 'simfix', 'json', 'Z_S2_test.json'), os.path.join(CN, 'simfix', 'json', 'S2__simtest.json')),
    ('S3', '机头+尺度修正', os.path.join(CN, 'simfix', 'json', 'Z_S3_test.json'), os.path.join(CN, 'simfix', 'json', 'S3__simtest.json')),
    ('R3', 'S3 + ImageNet ResNet-34', os.path.join(CN, 'simfix', 'json', 'Z_R3_test.json'), os.path.join(CN, 'simfix', 'json', 'R3__simtest.json')),
    ('P1', '重采 PowerPlant 真机尺寸 + 机头 +x（单场景）', os.path.join(CN, 'pp_realsize', 'json', 'Z_P1_test.json'), None),
    ('TINY_old', 'tiny（每 30 帧取 1）旧标签', os.path.join(CN, 'simfix', 'json', 'Z_TINY_old_test.json'), None),
    ('TINY_fix', 'tiny（每 30 帧取 1）机头修正（从已修正 D 盘建）', os.path.join(CN, 'simfix', 'json', 'Z_TINY_fix_test.json'), None),
]


def load(p):
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))


def f(v, d=3):
    return '-' if v is None or v != v else ('%.' + str(d) + 'f') % v


def row(tag, desc, d):
    if d is None:
        return '| %s | %s |' % (tag, desc) + ' - |' * 11
    ads = {k: parse_ads(v) for k, v in (d.get('ads_reports') or {}).items()}
    return '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        tag, desc, f(d['det_rate'], 3), f(d['pos_median']), f(d['z_median']), f(d['ang_median'], 1),
        f(d['ACC_pos_0.2'], 3), f(d['ACC_rot_20'], 3), f(d['ACC_0.2m20deg'], 3),
        f(ads.get('indoor', {}).get('ads'), 1), f(ads.get('laa', {}).get('ads'), 1),
        f(ads.get('indoor', {}).get('orin_med'), 1), f(ads.get('indoor', {}).get('size_med'), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=os.path.join(ROOT, 'docs', 'results', 'simfix_table.md'))
    args = ap.parse_args()
    head = ('| 臂 | 说明 | 检出率 | 位置中位 m | 深度误差中位 m | 角度中位 ° | ACC 0.2 m | ACC 20° | ACC 0.2 m&20° | ADS 室内 | ADS laa | ADS 朝向中位 ° | ADS 尺寸中位 m |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    lines = ['# 纯仿真训练：标签修正与模型对照', '',
             '全部只用仿真数据训练（同协议 24 轮、种子 0、仿真 test 抽帧选轮），没有见过任何真实帧。', '',
             '## 零样本：MAV6D 真实 test（4800 帧）', '', head]
    for tag, desc, zt, _ in ARMS:
        lines.append(row(tag, desc, load(zt)))
    txt = '\n'.join(lines) + '\n'
    print(txt)
    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    open(args.md, 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
