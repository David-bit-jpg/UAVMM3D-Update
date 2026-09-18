# -*- coding: utf-8 -*-
"""第二批夜间实验（2026-09-18 下午）：关键点头加权重多训，和现有最好配方的逐轮退化曲线。

K30：kp2d 损失权重 4 -> 12、30 轮。16 轮那次仿真域内角点误差 9.3 px、隐含尺度 0.82（size2d 域内是 0.99），
     即欠拟合而不是这条路不行。**先看域内**：隐含尺度到 0.95 以上、角点误差进 2 px 才有资格谈真实域。
D6 ：现有最好配方（sim_full_r34_geo_tr）全量数据 6 轮、每轮都存，逐轮评真实域。
     因为最好成绩 0.464 m 出现在很早（相当于 2 个全量轮次），8 轮的 M 已退到 0.573 m，要把退化曲线画准。

    python run_night2_0918.py
"""
import datetime
import json
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'night2')
MODELS = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm')
COMMON = ['--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', '0', '--cudnn_benchmark',
          '--use_amp', '--max_ckpt_save_num', '40', '--logger_iter_interval', '500', '--val_split', 'test',
          '--val_interval', '10', '--val_every', '5', '--val_match', 'greedy', '--skip_test_eval']


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


def show(js, prefix):
    try:
        d = json.load(open(js, encoding='utf-8'))
        log('  %s 位置 %.3f m | 深度 %.3f | <0.2m %.1f%% | <0.5m %.1f%% | 角度 %.1f°(折 %.1f) | uv %.1f px' % (
            prefix, d['pos_median'], d['z_median'], 100 * d['ACC_pos_0.2'], 100 * d['ACC_pos_0.5'],
            d['ang_median'], d['ang_fold_median'], d['uv_median']))
    except Exception as e:                                     # noqa: BLE001
        log('  %s 读结果失败 %r' % (prefix, e))


def main():
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    log('==== 第二批开始 ====')
    jobs = [('K30', 'sim_full_r34_kp_w12', 30), ('D6', 'sim_full_r34_geo_tr', 6)]
    procs = []
    for tag, stem, ep in jobs:
        ckdir = os.path.join(MODELS, stem, tag, 'ckpt')
        if os.path.exists(os.path.join(ckdir, 'checkpoint_epoch_%d.pth' % ep)):
            log('%s 已训过，跳过' % tag)
            continue
        f = open(os.path.join(OUT, '%s.train.log' % tag), 'w', encoding='utf-8')
        cmd = [PY, 'train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % stem,
               '--epochs', str(ep), '--extra_tag', tag] + COMMON
        procs.append((tag, subprocess.Popen(cmd, cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT), f))
        log('%s 训练启动（%s，%d 轮）' % (tag, stem, ep))
    for tag, p, f in procs:
        rc = p.wait()
        f.close()
        log('%s 训练结束 rc=%d' % (tag, rc))

    # D6：逐轮评真实域，画退化曲线
    for ep in range(1, 7):
        ck = os.path.join(MODELS, 'sim_full_r34_geo_tr', 'D6', 'ckpt', 'checkpoint_epoch_%d.pth' % ep)
        if not os.path.exists(ck):
            continue
        for src in ('known', 'pred'):
            name = 'D6_ep%d_%s' % (ep, src)
            js = os.path.join(OUT, 'json', name + '.json')
            run([PY, 'eval_camnorm.py', '--cfg', 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml',
                 '--ckpt', ck, '--tag', name, '--split', 'test', '--workers', '0', '--json', js,
                 '--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', src, 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], name)
            show(js, '第 %d 轮 %-6s' % (ep, src))

    # K30：先看域内收敛，再看真实域
    for ep in (10, 20, 30):
        ck = os.path.join(MODELS, 'sim_full_r34_kp_w12', 'K30', 'ckpt', 'checkpoint_epoch_%d.pth' % ep)
        if not os.path.exists(ck):
            continue
        run([PY, 'annot_audit/kp_error.py', '--ckpt', ck], 'kperr_K30_ep%d' % ep)
        for ln in open(os.path.join(OUT, 'kperr_K30_ep%d.log' % ep), encoding='utf-8', errors='replace'):
            if '角点绝对误差' in ln:
                log('  K30 第 %2d 轮 %s' % (ep, ln.strip()))
        for src in ('known', 'pred'):
            name = 'K30_ep%d_%s' % (ep, src)
            js = os.path.join(OUT, 'json', name + '.json')
            run([PY, 'eval_camnorm.py', '--cfg', 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml', '--ckpt', ck,
                 '--tag', name, '--split', 'test', '--workers', '0', '--json', js,
                 '--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', src], name)
            show(js, 'K30 第 %d 轮 %-6s' % (ep, src))
    log('ALL DONE')


if __name__ == '__main__':
    main()
