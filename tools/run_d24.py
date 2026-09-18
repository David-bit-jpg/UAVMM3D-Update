# -*- coding: utf-8 -*-
"""D 臂正式训练（2026-09-18）：全量训练集、24 轮，训完做全套评测。

D = ImageNet ResNet-34 + size2d 显式监督 2D 跨度 + 几何解深度【只用水平跨度】
    + 目标级随机化（外圈随机衰减 / 目标运动模糊）+ 裁窗按机身倍数对齐。
6 轮短 test 的 MAV6D 全量结果：已知尺寸 0.464 m、不给尺寸 0.527 m（旧基线 R1 是 4.591 m）。

评测三个口径：known（给机型尺寸）/ pred（完全零样本）/ both（同权重切回宽高联立，作为消融），
再跑常数先验对照与纯 2D 指标。

    python run_d24.py [--epochs 24] [--tag D24]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'd24')
STEM = 'sim_full_r34_geo_tr'
EVAL_CFG = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'


def log(msg):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M'), msg)
    print(line, flush=True)
    with open(os.path.join(OUT, 'driver.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd, name):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=os.path.dirname(TOOLS), TORCH_HOME='E:/torch_home')
    with open(os.path.join(OUT, name + '.log'), 'w', encoding='utf-8') as f:
        rc = subprocess.call(cmd, cwd=TOOLS, env=env, stdout=f, stderr=subprocess.STDOUT)
    log('%s rc=%d' % (name, rc))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--tag', default='D24')
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    log('==== D 臂正式训练开始：全量训练集，%d 轮 ====' % a.epochs)
    ckpt = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm', STEM, a.tag, 'ckpt', 'best.pth')
    if not os.path.exists(ckpt):
        rc = run([PY, 'train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % STEM,
                  '--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', '0', '--cudnn_benchmark',
                  '--use_amp', '--epochs', str(a.epochs), '--extra_tag', a.tag, '--max_ckpt_save_num', '1',
                  '--logger_iter_interval', '500', '--val_split', 'test', '--val_interval', '5', '--val_every', '3',
                  '--val_match', 'greedy', '--skip_test_eval'], a.tag + '.train')
        if rc or not os.path.exists(ckpt):
            log('训练失败，停止')
            return
    for src, mode in (('known', 'width'), ('pred', 'width'), ('known', 'both')):
        tag = 'Z_%s_%s_%s' % (a.tag, src, mode)
        run([PY, 'eval_camnorm.py', '--cfg', EVAL_CFG, '--ckpt', ckpt, '--tag', tag, '--split', 'test', '--workers', '0',
             '--json', os.path.join(OUT, 'json', tag + '.json'), '--ads',
             '--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', src, 'MODEL.POST_PROCESSING.SIZE2D_MODE', mode], tag)
        try:
            d = json.load(open(os.path.join(OUT, 'json', tag + '.json'), encoding='utf-8'))
            log('  %-22s 位置中位 %.3f m | 深度 %.3f | <0.2 m %.1f%% | <0.5 m %.1f%% | 角度 %.1f°(折 %.1f) | uv %.1f px' % (
                tag, d['pos_median'], d['z_median'], 100 * d['ACC_pos_0.2'], 100 * d['ACC_pos_0.5'],
                d['ang_median'], d['ang_fold_median'], d['uv_median']))
        except Exception as e:   # noqa: BLE001
            log('  %s 读结果失败 %r' % (tag, e))
    log('ALL DONE')


if __name__ == '__main__':
    main()
