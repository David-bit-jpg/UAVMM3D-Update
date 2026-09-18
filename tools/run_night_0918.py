# -*- coding: utf-8 -*-
"""夜间驱动（2026-09-18）：等在跑的两条训练结束 -> 稳健性选轮 -> 关键点臂与温和增广臂 -> 全部评测汇总。

背景与判据：
  * 深度的失败已定位为「模型觉得目标在图上多大」这一个 2D 量：size2d 头域内 0.99，真实域宽 0.85 高 0.53。
  * 能迁移的是定位类的 2D 量（中心 9.7 px、检出 85%），所以 K 臂把位姿整个交给 8 角点 + PnP。
    往返自检已过（1e-7 m）；PnP 敏感度：角点 1 px -> 位置 0.067 m / 角度 3.4°，2 px -> 0.129 m / 5.4°。
  * 强随机化保护深度但削弱朝向（域内角度 B 79.1° vs D 89.7°；D24 24 轮域内 34.6°，旧基线 14~18°），
    所以 M 臂（温和增广、无关键点）作对照，用来分清「关键点头」和「增广强度」各值多少。
  * 选轮只用仿真侧的稳健性判据（annot_audit/robust_val.py），真实数据只在最后报数。

    python run_night_0918.py [--wait-pids 772,21064]
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'night0918')
MODELS = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm')
COMMON = ['--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', '0', '--cudnn_benchmark',
          '--use_amp', '--max_ckpt_save_num', '30', '--logger_iter_interval', '500', '--val_split', 'test',
          '--val_interval', '10', '--val_every', '3', '--val_match', 'greedy', '--skip_test_eval']


def log(msg):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M'), msg)
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'driver.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def env():
    return dict(os.environ, PYTHONUTF8='1', PYTHONPATH=os.path.dirname(TOOLS), TORCH_HOME='E:/torch_home')


def run(cmd, name):
    with open(os.path.join(OUT, name + '.log'), 'w', encoding='utf-8') as f:
        rc = subprocess.call(cmd, cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT)
    log('%s rc=%d' % (name, rc))
    return rc


def alive(pid):
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x00100000, False, int(pid))
    if not h:
        return False
    r = ctypes.windll.kernel32.WaitForSingleObject(h, 0)
    ctypes.windll.kernel32.CloseHandle(h)
    return r == 258


def eval_real(stem, tag, ckpt, eval_cfg, sets, name):
    js = os.path.join(OUT, 'json', '%s.json' % name)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    rc = run([PY, 'eval_camnorm.py', '--cfg', eval_cfg, '--ckpt', ckpt, '--tag', name, '--split', 'test',
              '--workers', '0', '--json', js, '--set'] + sets, 'eval_' + name)
    try:
        d = json.load(open(js, encoding='utf-8'))
        log('  %-26s 位置 %.3f m | 深度 %.3f | <0.2m %.1f%% | <0.5m %.1f%% | 角度 %.1f°(折 %.1f) | uv %.1f px | 检出 %.2f' % (
            name, d['pos_median'], d['z_median'], 100 * d['ACC_pos_0.2'], 100 * d['ACC_pos_0.5'],
            d['ang_median'], d['ang_fold_median'], d['uv_median'], d['det_rate']))
        return d
    except Exception as e:                                   # noqa: BLE001
        log('  %s 读结果失败 %r' % (name, e))
        return None


def robust_pick(stem, tag):
    """跑稳健性验证，返回按扰动下位置中位选出的 checkpoint 路径。"""
    name = 'robustval_%s' % tag
    run([PY, 'annot_audit/robust_val.py', '--stem', stem, '--tag', tag, '--interval', '12'], name)
    best_ep, best_val = None, 1e9
    for ln in open(os.path.join(OUT, name + '.log'), encoding='utf-8', errors='replace'):
        m = re.search(r'第\s*(\d+) 轮 \| 干净 位置 ([\d.]+) 深度 ([\d.]+) .*扰动 位置 ([\d.]+)', ln)
        if m and float(m.group(4)) < best_val:
            best_ep, best_val = int(m.group(1)), float(m.group(4))
    if best_ep is None:
        log('  %s 稳健性验证没解析到结果，回退 best.pth' % tag)
        return os.path.join(MODELS, stem, tag, 'ckpt', 'best.pth'), None
    log('  %s 稳健性选轮 -> 第 %d 轮（扰动下位置 %.3f m）' % (tag, best_ep, best_val))
    return os.path.join(MODELS, stem, tag, 'ckpt', 'checkpoint_epoch_%d.pth' % best_ep), best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-pids', default='')
    ap.add_argument('--epochs', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    log('==== 夜间驱动开始 ====')
    pids = [int(x) for x in a.wait_pids.split(',') if x.strip()]
    while any(alive(p) for p in pids):
        time.sleep(120)
    if pids:
        log('在跑的训练已结束：%s' % pids)

    # 1) 对已经训完的两条做稳健性选轮 + 真实域评测
    D_EVAL = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'
    for stem, tag in (('sim_full_r34_geo_tr', 'D24b'), ('sim_full_r34_geo_tr2', 'E24')):
        if not os.path.isdir(os.path.join(MODELS, stem, tag, 'ckpt')):
            log('%s/%s 没有权重，跳过' % (stem, tag))
            continue
        ck, ep = robust_pick(stem, tag)
        for src in ('known', 'pred'):
            eval_real(stem, tag, ck, D_EVAL,
                      ['MODEL.POST_PROCESSING.SIZE_SOURCE', src, 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'],
                      '%s_robust_%s' % (tag, src))
        last = os.path.join(MODELS, stem, tag, 'ckpt', 'best.pth')
        if os.path.exists(last):
            eval_real(stem, tag, last, D_EVAL,
                      ['MODEL.POST_PROCESSING.SIZE_SOURCE', 'known', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'],
                      '%s_domainbest_known' % tag)

    # 2) K 臂（关键点 + PnP）与 M 臂（温和增广、无关键点）并行训练
    arms = [('K', 'sim_full_r34_kp'), ('M', 'sim_full_r34_geo_mild')]
    procs = []
    for tag, stem in arms:
        if not os.path.exists(os.path.join(TOOLS, 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % stem)):
            log('%s 配置缺失，跳过' % stem)
            continue
        f = open(os.path.join(OUT, '%s.train.log' % tag), 'w', encoding='utf-8')
        cmd = [PY, 'train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % stem,
               '--epochs', str(a.epochs), '--extra_tag', tag] + COMMON
        procs.append((tag, stem, subprocess.Popen(cmd, cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT), f))
        log('%s 臂训练启动（%s，%d 轮）' % (tag, stem, a.epochs))
    for tag, stem, p, f in procs:
        rc = p.wait()
        f.close()
        log('%s 臂训练结束 rc=%d' % (tag, rc))

    # 3) 两条新臂：稳健性选轮 + 真实域评测
    for tag, stem in arms:
        if not os.path.isdir(os.path.join(MODELS, stem, tag, 'ckpt')):
            continue
        ck, ep = robust_pick(stem, tag)
        if tag == 'K':
            for src in ('known', 'pred'):
                eval_real(stem, tag, ck, 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml',
                          ['MODEL.POST_PROCESSING.SIZE_SOURCE', src], 'K_robust_%s' % src)
        else:
            for src in ('known', 'pred'):
                eval_real(stem, tag, ck, D_EVAL,
                          ['MODEL.POST_PROCESSING.SIZE_SOURCE', src, 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'],
                          'M_robust_%s' % src)
    log('ALL DONE')


if __name__ == '__main__':
    main()
