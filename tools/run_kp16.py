# -*- coding: utf-8 -*-
"""K 臂正式验证（2026-09-18）：关键点 + PnP，16 轮、保存每一轮，然后逐轮看真实域表现随训练怎么变。

要回答两个问题：
  1. 关键点这条路在真实图上行不行 —— 关键点相对误差落到 1~2 px，位置就该是 0.07~0.13 m、角度 3~5°
     （PnP 敏感度由 annot_audit/check_kp2d.py 实测）。
  2. 它会不会像回归深度那样「训得越久跨域越差」 —— D 臂 6 轮 0.464 m、24 轮 0.766 m，
     强随机化（E24 0.854）和稳健性选轮（D24b 0.762）都没治好。所以逐轮评测，看趋势。

    python run_kp16.py [--epochs 16] [--eval-epochs 2,4,6,8,12,16]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'kp16')
CK = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm', 'sim_full_r34_kp', 'K16', 'ckpt')


def log(msg):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M'), msg)
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'driver.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def env():
    return dict(os.environ, PYTHONUTF8='1', PYTHONPATH=os.path.dirname(TOOLS), TORCH_HOME='E:/torch_home')


def run(cmd, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + '.log'), 'w', encoding='utf-8') as f:
        rc = subprocess.call(cmd, cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT)
    log('%s rc=%d' % (name, rc))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--eval-epochs', default='2,4,6,8,12,16')
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    log('==== K 臂 %d 轮开始 ====' % a.epochs)
    if not os.path.exists(os.path.join(CK, 'checkpoint_epoch_%d.pth' % a.epochs)):
        rc = run([PY, 'train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp.yaml',
                  '--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', '0', '--cudnn_benchmark',
                  '--use_amp', '--epochs', str(a.epochs), '--extra_tag', 'K16', '--max_ckpt_save_num', '30',
                  '--logger_iter_interval', '500', '--val_split', 'test', '--val_interval', '10', '--val_every', '4',
                  '--val_match', 'greedy', '--skip_test_eval'], 'K16.train')
        if rc:
            log('训练失败，停止')
            return
    for ep in [int(x) for x in a.eval_epochs.split(',')]:
        ck = os.path.join(CK, 'checkpoint_epoch_%d.pth' % ep)
        if not os.path.exists(ck):
            continue
        run([PY, 'annot_audit/kp_error.py', '--ckpt', ck], 'kperr_ep%d' % ep)
        for ln in open(os.path.join(OUT, 'kperr_ep%d.log' % ep), encoding='utf-8', errors='replace'):
            if '角点绝对误差' in ln:
                log('  第 %2d 轮 %s' % (ep, ln.strip()))
        for src in ('known', 'pred'):
            name = 'Z_K16_ep%d_%s' % (ep, src)
            js = os.path.join(OUT, 'json', name + '.json')
            run([PY, 'eval_camnorm.py', '--cfg', 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml', '--ckpt', ck,
                 '--tag', name, '--split', 'test', '--workers', '0', '--json', js,
                 '--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', src], name)
            try:
                d = json.load(open(js, encoding='utf-8'))
                log('  第 %2d 轮 %-6s 位置 %.3f m | 深度 %.3f | <0.2m %.1f%% | <0.5m %.1f%% | 角度 %.1f°(折 %.1f) | uv %.1f px' % (
                    ep, src, d['pos_median'], d['z_median'], 100 * d['ACC_pos_0.2'], 100 * d['ACC_pos_0.5'],
                    d['ang_median'], d['ang_fold_median'], d['uv_median']))
            except Exception as e:                       # noqa: BLE001
                log('  第 %d 轮 %s 读结果失败 %r' % (ep, src, e))
    log('ALL DONE')


if __name__ == '__main__':
    main()
