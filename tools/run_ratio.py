# -*- coding: utf-8 -*-
"""Q2 驱动（2026-09-17）：机身倍数深度（DEPTH_TARGET=log_ratio）+ ImageNet ResNet-34，pp_realsize 上训练，
训完在 MAV6D test 上评两种尺寸口径：pred（网络自己预测尺寸）/ known（已知机型尺寸 0.34x0.34x0.23）。
与 R1（同骨干、同数据、同协议）对照。脱离会话启动：PowerShell Start-Process python run_ratio.py

    python run_ratio.py [--tag Q2] [--epochs 24]
"""
import argparse
import datetime
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'ratio')


def log(msg):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M'), msg)
    print(line, flush=True)
    with open(os.path.join(OUT, 'driver.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd, logfile):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=os.path.dirname(TOOLS), TORCH_HOME='E:/torch_home')
    with open(logfile, 'w', encoding='utf-8') as f:
        return subprocess.call(cmd, cwd=TOOLS, env=env, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='Q2')
    ap.add_argument('--epochs', type=int, default=24)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    stem = 'sim_pp_r34_ratio'
    log('开始训练 %s（%s）' % (a.tag, stem))
    rc = run([PY, 'train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % stem,
              '--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', '0', '--cudnn_benchmark', '--use_amp',
              '--epochs', str(a.epochs), '--extra_tag', a.tag, '--max_ckpt_save_num', '1', '--logger_iter_interval', '200',
              '--val_split', 'test', '--val_interval', '5', '--val_every', '2', '--val_match', 'greedy', '--skip_test_eval',
              '--set', 'DATA_CONFIG.TRAIN_REPEAT', '4'], os.path.join(OUT, '%s.train.log' % a.tag))
    log('%s 训练结束 rc=%d' % (a.tag, rc))
    ckpt = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm', stem, a.tag, 'ckpt', 'best.pth')
    for src in ('pred', 'known'):
        tag = 'Z_%s_%s' % (a.tag, src)
        rc = run([PY, 'eval_camnorm.py', '--cfg', 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_ratio_pp.yaml', '--ckpt', ckpt,
                  '--tag', tag, '--split', 'test', '--workers', '0', '--json', os.path.join(OUT, 'json', tag + '_test.json'),
                  '--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', src], os.path.join(OUT, tag + '_test.log'))
        log('%s MAV6D test 评测 rc=%d' % (tag, rc))
    log('ALL DONE')


if __name__ == '__main__':
    main()
