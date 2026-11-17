# -*- coding: utf-8 -*-
"""把 inst/online_fuse.py 写出的 online_<CMP_OUT>.pkl 整理成 Markdown 表（真实测试集 / 验证集，测距用部署版）。
用法：python release/print_table.py <online_xxx.pkl> [test|val]"""
import pickle, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
rows = pickle.load(open(sys.argv[1], 'rb'))
sp = sys.argv[2] if len(sys.argv) > 2 else 'test'
print('| 模型（%s，%s） | 推理时实例原型 | 平均精度 | 平均位置误差 m | 平均尺寸误差 cm | 平均朝向误差 ° | 检出率 %% | UDS* |' % ('真实测试集' if sp == 'test' else '真实验证集', '测距用部署版'))
print('|---|---|---|---|---|---|---|---|')
for r in rows:
    if r['split'] != sp or r['strength'] != '强':
        continue
    a = r['metrics']['all']
    print('| %s | %s | %.2f | %.3f | %.2f | %.2f | %.1f | %.2f |' % (
        r['name'], '无' if r['online'] == '原样' else '有（等后 5 帧）', a['AP'], a['ATE'], a['ASE'], a['AOE'], a['mDR'], a['UDS']))
print('\n*UDS：MAV6D 原文没给公式，按 nuScenes NDS 的写法自定义（output/camnorm/geo1004/cmpeval/paper_table.py 的 metrics()）：'
      '0.5×平均精度 + 0.5×平均(1−min(1,位置/1 m), 1−min(1,尺寸/10 cm), 1−min(1,朝向/1 rad), 检出率)。')
