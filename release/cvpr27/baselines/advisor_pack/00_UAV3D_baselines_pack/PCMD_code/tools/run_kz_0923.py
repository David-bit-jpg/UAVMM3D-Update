# -*- coding: utf-8 -*-
"""KZ 臂驾驶脚本第二版（2026-09-23 03:00）：09-22 夜的会话 01:08 卡死，把 run_kz_0922.py 和种子 2、3（第 20 轮）一起带走了。
本脚本从断点续跑，并把评测、收尾（含折 90 的总表、画廊）合在一起；用 PowerShell Start-Process 独立启动，不挂在会话上。

  1) 种子 2、3 并行续训（train.py 不给 --ckpt 时自动从 run 目录里最新的 checkpoint 续，OneCycle 步数按 --epochs 30 不变）
  2) 种子 4 从头训
  3) 评测（真实 test 4800 帧，位置 = 发布版 5 集成 + 翻转 + 背景 + 自标定，各行逐帧相同）：
     旋转 ← KZ 各单种子（裁窗 95）、KZ 5 种子平均（裁窗 95 / 整幅对照）、K30 两种子（裁窗 95）；各种子仿真域内诊断；论文口径
  4) 总表（summarize_dumps 含最优常数 / 折 90 / 折180<10° 列）→ KZ_SUMMARY.md
  5) 可视化：查看器缺省 dump 换成 KZ 5 种子平均；kz_view 裁窗+角点拼图；画廊（序列顺序 12 帧整幅+放大、按航向误差 12 帧、按位置误差 12 帧）+ 缩略总览
"""
import glob
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
os.chdir(TOOLS)
PY = sys.executable
ENV = dict(os.environ, PYTHONUTF8='1', PYTHONPATH='@@ROOT@@')
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'kz')
MODELS = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm')
REL = os.path.join(TOOLS, '..', 'output', 'release_20260920')
REL22 = os.path.join(TOOLS, '..', 'output', 'release_20260922')
VIEWER = os.path.join(TOOLS, '..', 'output', 'camnorm', 'viewer')
NIGHT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'night0922')
CFG = 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom.yaml'
REAL_KP = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_kp.yaml'
COMMON = ['--batch_size', '8', '--workers', '2', '--fix_random_seed', '--cudnn_benchmark', '--use_amp', '--epochs', '30',
          '--max_ckpt_save_num', '3', '--logger_iter_interval', '500', '--val_split', 'test', '--val_interval', '10',
          '--val_every', '100', '--val_match', 'greedy', '--skip_test_eval']
SUMMARY = os.path.join(OUT, 'KZ_SUMMARY.md')
GAL = os.path.join(OUT, 'gallery')
ZOOM = '95'


def say(s):
    line = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), s)
    print(line, flush=True)
    open(os.path.join(OUT, 'driver.log'), 'a', encoding='utf-8').write(line + '\n')


def train(seed):
    f = open(os.path.join(OUT, 'KZs%d.stdout' % seed), 'a', encoding='utf-8')
    return subprocess.Popen([PY, 'train.py', '--cfg_file', CFG] + COMMON + ['--seed', str(seed), '--extra_tag', 'KZs%d' % seed],
                            stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=TOOLS)


def sh(args, log_name):
    with open(os.path.join(OUT, log_name), 'w', encoding='utf-8') as f:
        r = subprocess.run([PY] + args, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=TOOLS)
    if r.returncode != 0:
        say('!! 失败 (%d): %s' % (r.returncode, ' '.join(args)[:220]))
    return r.returncode == 0


def append(header, text):
    open(SUMMARY, 'a', encoding='utf-8').write('\n## %s\n\n```\n%s\n```\n' % (header, text.strip()))


def log_text(log_name):
    return '\n'.join(l.rstrip() for l in open(os.path.join(OUT, log_name), encoding='utf-8', errors='replace')
                     if 'INFO' not in l and 'loading' not in l)


def ck(seed):
    return os.path.join(MODELS, 'sim_full_r34_kp_w12_zoom', 'KZs%d' % seed, 'ckpt', 'checkpoint_epoch_30.pth')


def dump(name, kps, zoom=True):
    out = os.path.join(OUT, 'real_ens5flip_%s.pkl' % name)
    if os.path.exists(out):
        return out
    args = ['annot_audit/dump_real_preds.py', '--kp2d-from'] + kps + ['--kp2d-cfg', REAL_KP, '--out', out]
    if zoom:
        args += ['--kp2d-zoom', ZOOM]
    sh(args, 'dump_%s.log' % name)
    return out


def main():
    say('第二版驾驶脚本启动：续训种子 2、3')
    procs = [train(s) for s in (2, 3) if not os.path.exists(ck(s))]
    for p in procs:
        p.wait()
    say('种子 2、3 结束（返回码 %s）；启动种子 4' % [p.returncode for p in procs])
    if not os.path.exists(ck(4)):
        p4 = train(4)
        p4.wait()
        say('种子 4 结束（返回码 %d）' % p4.returncode)
    kz = [ck(s) for s in range(5) if os.path.exists(ck(s))]
    say('可用权重 %d 个：%s' % (len(kz), [os.path.basename(os.path.dirname(os.path.dirname(c))) for c in kz]))

    # ---------------- 评测 ----------------
    k30 = [os.path.join(REL22, 'k30_s0.pth'), os.path.join(REL22, 'k30_s1.pth')]
    pairs = [('旧旋转头', os.path.join(VIEWER, 'real_preds_ens5flip_rothead.pkl')),
             ('K30两种子_整幅', os.path.join(NIGHT, 'real_ens5flip_kpK30avg.pkl')),
             ('K30两种子_裁窗95', dump('kpK30avg_zoom', k30))]
    for s, c in enumerate(kz):
        pairs.append(('KZ_s%d_裁窗95' % s, dump('kpKZs%d_zoom' % s, [c])))
    pairs.append(('KZ_%d种子平均_裁窗95' % len(kz), dump('kpKZavg_zoom', kz)))
    pairs.append(('KZ_%d种子平均_整幅' % len(kz), dump('kpKZavg_full', kz, zoom=False)))
    r = subprocess.run([PY, 'annot_audit/summarize_dumps.py'] + ['%s=%s' % (n, p) for n, p in pairs if os.path.exists(p)],
                       capture_output=True, text=True, env=ENV, cwd=TOOLS, encoding='utf-8', errors='replace')
    append('总表：真实 test 4800 帧，位置 = 发布版 5 集成 + 翻转 + 背景 + 自标定（各行逐帧相同）；旋转来源对比', r.stdout)
    say('总表已写入')
    for s, c in enumerate(kz):
        run = os.path.dirname(os.path.dirname(c))
        tag = os.path.basename(run)
        if sh(['annot_audit/kp_arm_diag.py', '--run', run, '--sim-cfg', CFG, '--real-cfg', REAL_KP, '--interval', '4', '--parts', 'rot'],
              'kp_diag_%s.log' % tag):
            append('%s 单独（仿真 test 按 VAL_ZOOMS 2.5 裁窗 / 真实整幅 —— 真实这行不是部署方式，只看角点）' % tag, log_text('kp_diag_%s.log' % tag))
    base = ['annot_audit/paper_metrics.py', '--ckpt', os.path.join(REL, 'ramp_s0.pth'), '--ensemble'] + \
           [os.path.join(REL, 'ramp_s%d.pth' % i) for i in range(1, 5)] + ['--flip', '--self-calib']
    if sh(base + ['--tag', 'KZ %d种子平均(裁窗95)' % len(kz), '--kp2d-from'] + kz + ['--kp2d-cfg', REAL_KP, '--kp2d-zoom', ZOOM], 'paper_kz.log'):
        append('论文口径（10°10cm / ADD）：旋转 ← KZ %d 种子平均（裁窗 95）' % len(kz), log_text('paper_kz.log'))
    say('评测完成')

    # ---------------- 可视化 ----------------
    best = os.path.join(OUT, 'real_ens5flip_kpKZavg_zoom.pkl')
    if os.path.exists(best):
        cur = os.path.join(VIEWER, 'real_preds_ens5flip.pkl')
        if os.path.exists(cur) and not os.path.exists(os.path.join(VIEWER, 'real_preds_ens5flip_k30full.pkl')):
            shutil.move(cur, os.path.join(VIEWER, 'real_preds_ens5flip_k30full.pkl'))
        shutil.copy(best, cur)
        say('查看器缺省 dump 已换成 KZ 种子平均（K30 整幅版 = real_preds_ens5flip_k30full.pkl）')
        sh(['annot_audit/kz_view.py', '--kp2d-from'] + kz + ['--kp2d-zoom', ZOOM, '--n', '16', '--interval', '250',
            '--out', os.path.join(OUT, 'kz_view.jpg')], 'kz_view.log')
        os.makedirs(GAL, exist_ok=True)
        for tag, extra in (('seq', ['--sort', 'frame', '--export', '12', '--export-zoom']),
                           ('yaw', ['--sort', 'yaw', '--export', '12']),
                           ('pos', ['--sort', 'pos', '--export', '12'])):
            sh(['annot_audit/view_real_preds.py', '--panel-width', '900', '--export-dir', os.path.join(GAL, tag)] + extra, 'gallery_%s.log' % tag)
            say('画廊 %s：%d 张' % (tag, len(glob.glob(os.path.join(GAL, tag, '*.jpg')))))
        ims = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(GAL, 'seq', '*.jpg'))) if not p.endswith('_zoom.jpg')]
        ims = [cv2.resize(i, (i.shape[1] // 2, i.shape[0] // 2), interpolation=cv2.INTER_AREA) for i in ims if i is not None]
        if ims:
            while len(ims) % 2:
                ims.append(np.zeros_like(ims[0]))
            cv2.imwrite(os.path.join(GAL, 'gallery_sheet.jpg'), np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, len(ims), 2)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    say('全部完成')


if __name__ == '__main__':
    main()
