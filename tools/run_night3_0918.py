# -*- coding: utf-8 -*-
"""第三批（2026-09-18 晚）：关键点路线的两项改进，等第二批跑完自动开始。

K30 域内已证明关键点 + PnP 比回归路线强（第 25 轮 位置 0.335 m / 角度 8.9°，D 系列最好 34.6°），
但暴露两个问题：检出只有 0.52（D 系列 0.70），且每个目标只有 1 个格子有监督。

KA = K30 + 热力图损失权重 1.0 -> 2.0          （补检测）
KC = KA  + 关键点监督扩到 3x3 邻域（KP_NEIGHBOR 1）（补定位精度；PnP 敏感度 1 px -> 位置 0.067 m）
KA vs K30 单独衡量热力图权重，KC vs KA 单独衡量邻域监督。

    python run_night3_0918.py --wait-pid <第二批 pid>
"""
import argparse
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'night3')
MODELS = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm')
COMMON = ['--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', '0', '--cudnn_benchmark',
          '--use_amp', '--max_ckpt_save_num', '40', '--logger_iter_interval', '500', '--val_split', 'test',
          '--val_interval', '10', '--val_every', '5', '--val_match', 'greedy', '--skip_test_eval']
ARMS = [('KA', 'sim_full_r34_kp_hm2'), ('KC', 'sim_full_r34_kp_hm2_n1')]
EVAL_EPOCHS = (10, 20, 30)


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


def alive(pid):
    h = ctypes.windll.kernel32.OpenProcess(0x00100000, False, int(pid))
    if not h:
        return False
    r = ctypes.windll.kernel32.WaitForSingleObject(h, 0)
    ctypes.windll.kernel32.CloseHandle(h)
    return r == 258


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-pid', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    log('==== 第三批开始 ====')
    while a.wait_pid and alive(a.wait_pid):
        time.sleep(120)
    if a.wait_pid:
        log('第二批已结束')

    procs = []
    for tag, stem in ARMS:
        if os.path.exists(os.path.join(MODELS, stem, tag, 'ckpt', 'checkpoint_epoch_%d.pth' % a.epochs)):
            log('%s 已训过，跳过' % tag)
            continue
        f = open(os.path.join(OUT, '%s.train.log' % tag), 'w', encoding='utf-8')
        cmd = [PY, 'train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % stem,
               '--epochs', str(a.epochs), '--extra_tag', tag] + COMMON
        procs.append((tag, subprocess.Popen(cmd, cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT), f))
        log('%s 训练启动（%s，%d 轮）' % (tag, stem, a.epochs))
    for tag, p, f in procs:
        rc = p.wait()
        f.close()
        log('%s 训练结束 rc=%d' % (tag, rc))

    for tag, stem in ARMS:
        hist = os.path.join(MODELS, stem, tag, 'val_history.json')
        if os.path.exists(hist):
            h = json.load(open(hist, encoding='utf-8'))
            log('%s 域内：%s' % (tag, ' | '.join('第%d轮 位置%.3f 角度%.1f 检出%.2f' % (
                r['epoch'], r['pos_median'], r['ang_median'], r['det_rate']) for r in h)))
        for ep in EVAL_EPOCHS:
            ck = os.path.join(MODELS, stem, tag, 'ckpt', 'checkpoint_epoch_%d.pth' % ep)
            if not os.path.exists(ck):
                continue
            run([PY, 'annot_audit/kp_error.py', '--ckpt', ck], '%s_kperr_ep%d' % (tag, ep))
            for ln in open(os.path.join(OUT, '%s_kperr_ep%d.log' % (tag, ep)), encoding='utf-8', errors='replace'):
                if '角点绝对误差' in ln:
                    log('  %s 第 %2d 轮 %s' % (tag, ep, ln.strip()))
            for src in ('known', 'pred'):
                name = '%s_ep%d_%s' % (tag, ep, src)
                js = os.path.join(OUT, 'json', name + '.json')
                run([PY, 'eval_camnorm.py', '--cfg', 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml', '--ckpt', ck,
                     '--tag', name, '--split', 'test', '--workers', '0', '--json', js,
                     '--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', src], name)
                try:
                    d = json.load(open(js, encoding='utf-8'))
                    log('  %s 第 %2d 轮 %-6s 位置 %.3f m | 深度 %.3f | <0.2m %.1f%% | <0.5m %.1f%% | 角度 %.1f°(折 %.1f) | uv %.1f px | 检出 %.2f' % (
                        tag, ep, src, d['pos_median'], d['z_median'], 100 * d['ACC_pos_0.2'], 100 * d['ACC_pos_0.5'],
                        d['ang_median'], d['ang_fold_median'], d['uv_median'], d['det_rate']))
                except Exception as e:                        # noqa: BLE001
                    log('  %s 读结果失败 %r' % (name, e))
    log('ALL DONE')


if __name__ == '__main__':
    main()
