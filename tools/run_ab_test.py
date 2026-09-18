# -*- coding: utf-8 -*-
"""短 test：验证「裁窗按机身倍数对齐」+「显式监督 2D 表观大小」两项改动是否真的让深度在真实图上可用（2026-09-18）。

判据（都在 MAV6D test 上、只看 2D 命中的帧）：位置中位必须明显低于「机身倍数固定成训练均值」的常数对照。
Q2（PowerPlant + 随机裁窗）实测：网络 0.942 m，常数对照 0.754 m —— 网络输不过常数，等于没读图。

A 臂 sim_full_r34_ratio_t：深度学 log(Zv/L)，裁窗按 t 对齐
B 臂 sim_full_r34_geo_t  ：size2d 头显式监督 2D 跨度 + 几何解深度，裁窗按 t 对齐
两臂同数据（full_v2）、同骨干（ImageNet ResNet-34）、同轮数，并行训练。

    python run_ab_test.py [--epochs 6] [--interval 3]
"""
import argparse
import datetime
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'ab_test')
ARMS = [
    ('A', 'sim_full_r34_ratio_t', 'mav6d_r34_ratio_full_t'),
    ('B', 'sim_full_r34_geo_t', 'mav6d_r34_geo_full_t'),
]


def log(msg):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M'), msg)
    print(line, flush=True)
    with open(os.path.join(OUT, 'driver.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def env():
    return dict(os.environ, PYTHONUTF8='1', PYTHONPATH=os.path.dirname(TOOLS), TORCH_HOME='E:/torch_home')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--interval', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    log('==== 短 test 开始：%d 轮，训练集抽帧间隔 %d ====' % (a.epochs, a.interval))

    procs = []
    for tag, stem, _ in ARMS:
        f = open(os.path.join(OUT, '%s.train.log' % tag), 'w', encoding='utf-8')
        cmd = [PY, 'train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % stem,
               '--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', '0', '--cudnn_benchmark', '--use_amp',
               '--epochs', str(a.epochs), '--extra_tag', tag, '--max_ckpt_save_num', '1', '--logger_iter_interval', '300',
               '--val_split', 'test', '--val_interval', '10', '--val_every', '3', '--val_match', 'greedy', '--skip_test_eval',
               '--set', 'DATA_CONFIG.SAMPLED_INTERVAL.train', str(a.interval)]
        procs.append((tag, subprocess.Popen(cmd, cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT), f))
        log('%s 臂训练已启动（%s）' % (tag, stem))
    for tag, p, f in procs:
        rc = p.wait()
        f.close()
        log('%s 臂训练结束 rc=%d' % (tag, rc))

    for tag, stem, ecfg in ARMS:
        ckpt = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm', stem, tag, 'ckpt', 'best.pth')
        if not os.path.exists(ckpt):
            log('%s 臂没有 best.pth，跳过评测' % tag)
            continue
        for src in ('pred', 'known'):
            name = 'Z_%s_%s' % (tag, src)
            with open(os.path.join(OUT, name + '.log'), 'w', encoding='utf-8') as f:
                rc = subprocess.call([PY, 'eval_camnorm.py', '--cfg', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % ecfg,
                                      '--ckpt', ckpt, '--tag', name, '--split', 'test', '--workers', '0',
                                      '--json', os.path.join(OUT, 'json', name + '_test.json'),
                                      '--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', src],
                                     cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT)
            log('%s 评测 rc=%d' % (name, rc))
    # 关键对照（网络 vs 常数先验 vs 真值上界），逐臂打印
    with open(os.path.join(OUT, 'compare.log'), 'w', encoding='utf-8') as f:
        rc = subprocess.call([PY, 'annot_audit/compare_depth_prior.py'], cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT)
    log('对照脚本 rc=%d' % rc)
    for line in open(os.path.join(OUT, 'compare.log'), encoding='utf-8', errors='replace'):
        if '位置中位' in line or '判据' in line:
            log('  ' + line.rstrip())
    log('ALL DONE')


if __name__ == '__main__':
    main()
