# -*- coding: utf-8 -*-
"""11-05 实例原型修正伪标签：inst（正常增强）/ wainst（弱增强）× 种子 0–2 → 推理 → 测距强弱 → compare_gen。"""
import os, time
import run_proto as RP
idx = '@@MMCACHE@@/mav6d_cn_obj/train/index_pseudo_st2inst.pkl'
while not os.path.exists(idx):
    time.sleep(60)
for k in range(3):
    for n in ('inst', 'wainst'):
        RP.train(n, k)
    RP.say('inst: 种子 %d 两组训练完' % k)
for n in ('inst', 'wainst'):
    RP.infer(n)
    RP.run([RP.G + '/ablation/ablation.py', 'unc', 'V2Qp', n + 'wq'], '%s_uncwq.log' % n, env={'UAV_ENSDIR': n + '_ens'}, check=RP.G + '/finalads/unc_val_%swq.pkl' % n)
RP.run([RP.O + '/compare_gen.py'], 'compare_gen_inst.log', env={'CUDA_VISIBLE_DEVICES': '', 'CMP_OUT': 'inst',
       'CMP_SPEC': 'dep3:部署版(3种子),nd3:噪声对照(3种子),wa:弱增强 无原型,zgt:上限 伪距离换真值,inst:实例原型修正伪距离,wainst:弱增强+实例原型'})
RP.say('inst: 全部结束')
