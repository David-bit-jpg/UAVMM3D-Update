# -*- coding: utf-8 -*-
"""第四批（2026-09-18 晚）：固化配方 + 两个没在新管线里试过的域泛化开关。

背景：现有最好配方（sim_full_r34_geo_tr = size2d 只用宽度几何解深度 + 目标级随机化 + 裁窗对齐）
在全量数据上第 4~5 轮达到 0.498/0.522 m（给尺寸），之后退化；关键点路线已判负（真实域角点误差 30 px 不动）；
推理期多尺度平均无效（0.415 -> 0.420）。剩下没试过的是两个域泛化开关，都只改一行配置：
    MixStyle   训练时混合批内样本的通道统计量，堵死「靠风格判尺度」
    FREEZE_BN  锁住骨干 ImageNet BN 的 running stats，不让它被仿真拉走
同时用两个新种子重跑基线，确认 0.5 m 这个数不是单种子运气。

    python run_night4_0918.py
"""
import datetime
import json
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'night4')
MODELS = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm')
EVAL = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'
# tag, 配置, 种子
JOBS = [('S1', 'sim_full_r34_geo_tr', 1), ('MIX', 'sim_full_r34_geo_tr_mix', 0),
        ('S2', 'sim_full_r34_geo_tr', 2), ('FBN', 'sim_full_r34_geo_tr_fbn', 0)]
EPOCHS = 5
EVAL_EPOCHS = (3, 4, 5)


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
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    log('==== 第四批开始（%d 轮 x %d 条）====' % (EPOCHS, len(JOBS)))
    for pair in (JOBS[:2], JOBS[2:]):
        procs = []
        for tag, stem, seed in pair:
            if os.path.exists(os.path.join(MODELS, stem, tag, 'ckpt', 'checkpoint_epoch_%d.pth' % EPOCHS)):
                log('%s 已训过，跳过' % tag)
                continue
            f = open(os.path.join(OUT, '%s.train.log' % tag), 'w', encoding='utf-8')
            cmd = [PY, 'train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % stem,
                   '--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', str(seed),
                   '--cudnn_benchmark', '--use_amp', '--epochs', str(EPOCHS), '--extra_tag', tag,
                   '--max_ckpt_save_num', '10', '--logger_iter_interval', '500', '--val_split', 'test',
                   '--val_interval', '10', '--val_every', '5', '--val_match', 'greedy', '--skip_test_eval']
            procs.append((tag, subprocess.Popen(cmd, cwd=TOOLS, env=env(), stdout=f, stderr=subprocess.STDOUT), f))
            log('%s 训练启动（%s，种子 %d）' % (tag, stem, seed))
        for tag, p, f in procs:
            rc = p.wait()
            f.close()
            log('%s 训练结束 rc=%d' % (tag, rc))

    for tag, stem, seed in JOBS:
        for ep in EVAL_EPOCHS:
            ck = os.path.join(MODELS, stem, tag, 'ckpt', 'checkpoint_epoch_%d.pth' % ep)
            if not os.path.exists(ck):
                continue
            for src in ('known', 'pred'):
                name = '%s_ep%d_%s' % (tag, ep, src)
                js = os.path.join(OUT, 'json', name + '.json')
                run([PY, 'eval_camnorm.py', '--cfg', EVAL, '--ckpt', ck, '--tag', name, '--split', 'test',
                     '--workers', '0', '--json', js, '--set',
                     'MODEL.POST_PROCESSING.SIZE_SOURCE', src, 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width'], name)
                try:
                    d = json.load(open(js, encoding='utf-8'))
                    log('  %-4s 第 %d 轮 %-6s 位置 %.3f m | 深度 %.3f | <0.2m %.1f%% | <0.5m %.1f%% | 角度 %.1f°(折 %.1f) | uv %.1f px' % (
                        tag, ep, src, d['pos_median'], d['z_median'], 100 * d['ACC_pos_0.2'], 100 * d['ACC_pos_0.5'],
                        d['ang_median'], d['ang_fold_median'], d['uv_median']))
                except Exception as e:                       # noqa: BLE001
                    log('  %s 读结果失败 %r' % (name, e))
    log('ALL DONE')


if __name__ == '__main__':
    main()
