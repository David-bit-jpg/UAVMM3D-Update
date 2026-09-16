# -*- coding: utf-8 -*-
"""camnorm 全量实验调度（用户 2026-09-14 定：虚拟深度 / 训练设计全改 / 1-5-10-25-50-100% 全档 x 3 种子）。

    cd E:/Open3DUAVDet/tools && python run_camnorm_sweep.py            # 全部（可重复运行，已完成的跳过）
    python run_camnorm_sweep.py --dry                                   # 只列出作业

作业：
  S1            仿真预训练（sim_indoor8_mz.yaml：原分辨率多焦距裁窗），选轮 = 仿真 test 抽帧（整幅/长焦轮流，不看真实数据）
                （S0 = 只整幅缩放的版本，虚拟深度覆盖不到 MAV6D，已作废，见 output/camnorm/diag_20260914/S0_fullframe）
  Z_S1          零样本：S1 选出的权重直接测 MAV6D val / test
  C_p{b}_s{k}   MAV6D 从零，预算 b%，种子 k
  T_p{b}_s{k}   MAV6D 迁移（S0 权重整体载入，不跳层、不换算），预算 b%，种子 k
每个 MAV6D 训练：--val_split val 选轮存 best.pth -> eval_camnorm.py 在 test 上只测一次（含原版 ADS）。

约束（这台机器）：同时最多 2 个训练进程（>=3 个带 spawn worker 的进程会锁死）；评测跟在各自训练后面、占同一个槽。
"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
MODELS = os.path.join(ROOT, 'output', 'models', 'uavdet_3d', 'camnorm')
OUT = os.path.join(ROOT, 'output', 'camnorm')
LOG = os.path.join(OUT, 'logs')
JS = os.path.join(OUT, 'json')

SIM_CFG = 'cfgs/models/uavdet_3d/camnorm/sim_indoor8_mz.yaml'
SIM_STEM = 'sim_indoor8_mz'
SIM_READY = 'E:/mmcache/indoor8jpg/READY'          # 原分辨率缓存 + 可见度分数都建好后才写
MAV_CFG = 'cfgs/models/uavdet_3d/camnorm/mav6d.yaml'
# (预算 %, SAMPLED_INTERVAL, TRAIN_REPEAT, 轮数, 每几轮验证)。<=50% 档每轮约 800 iteration；
# 总迭代 1/5/10% 各 8k、25% 10.4k、50% 14.4k、100% 19.3k（5% 档 20 轮 = 1.6k iteration 时验证分数仍在陡升，原计划偏少）
BUDGETS = [(1, 100, 50, 10, 1), (5, 20, 10, 10, 1), (10, 10, 5, 10, 1), (25, 4, 2, 13, 1), (50, 2, 1, 18, 1), (100, 1, 1, 12, 1)]
SEEDS = [0, 1, 2]
PRETRAIN_EPOCHS = 24
WORKERS = 2
BATCH = 8


def say(*a):
    msg = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), ' '.join(str(x) for x in a))
    print(msg, flush=True)
    with open(os.path.join(LOG, 'sweep.log'), 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


CACHES = {'mav6d': ['E:/mmcache/mav6d_cn/train/rgb.npy', 'E:/mmcache/mav6d_cn/val/rgb.npy',
                    'E:/mmcache/mav6d_cn/test/rgb.npy'],
          'sim': ['E:/mmcache/indoor8jpg/train/rgb_jpg.bin', 'E:/mmcache/indoor8jpg/test/rgb_jpg.bin']}


def warm(group):
    """顺序读一遍缓存文件进系统页缓存。机械盘随机读 memmap 每样本约 48 ms（实测），读进内存后不到 10 ms；
    已经在页缓存里时只是内存拷贝，几秒钟。"""
    t0 = time.time()
    n = 0
    for f in CACHES[group]:
        if not os.path.exists(f):
            continue
        with open(f, 'rb', buffering=0) as fh:
            while True:
                b = fh.read(64 << 20)
                if not b:
                    break
                n += len(b)
    say('预读 %s 缓存 %.1f GB，%.0f s' % (group, n / 1e9, time.time() - t0))


def best_ckpt(cfg_stem, tag):
    p = os.path.join(MODELS, cfg_stem, tag, 'ckpt', 'best.pth')
    return p if os.path.exists(p) else None


def train_done(cfg_stem, tag, epochs):
    h = os.path.join(MODELS, cfg_stem, tag, 'val_history.json')
    if not os.path.exists(h) or not best_ckpt(cfg_stem, tag):
        return False
    try:
        hist = json.load(open(h, encoding='utf-8'))
    except Exception:
        return False
    return any(int(x.get('epoch', -1)) >= epochs for x in hist)


class Job:
    def __init__(self, name, steps, deps=(), done=None, wait_files=()):
        self.name, self.steps, self.deps, self.done = name, steps, list(deps), done
        self.wait_files = list(wait_files)
        self.proc, self.step_i, self.log = None, 0, None

    def is_done(self):
        return self.done() if self.done else False


def train_cmd(cfg, tag, epochs, seed, val_split, val_interval, val_every, val_match, interval=None, pretrained=None,
              repeat=None):
    cmd = [PY, 'train.py', '--cfg_file', cfg, '--batch_size', str(BATCH), '--workers', str(WORKERS),
           '--fix_random_seed', '--seed', str(seed), '--cudnn_benchmark', '--use_amp', '--epochs', str(epochs),
           '--extra_tag', tag, '--max_ckpt_save_num', '1', '--logger_iter_interval', '200',
           '--val_split', val_split, '--val_interval', str(val_interval), '--val_every', str(val_every),
           '--val_match', val_match, '--skip_test_eval']
    if pretrained:
        cmd += ['--pretrained_model', pretrained]
    if interval is not None:
        cmd += ['--set', 'DATA_CONFIG.SAMPLED_INTERVAL.train', str(interval), 'DATA_CONFIG.TRAIN_REPEAT', str(repeat or 1)]
    return cmd


def eval_cmd(ckpt, tag, split='test', ads=True, cfg=MAV_CFG):
    cmd = [PY, 'eval_camnorm.py', '--cfg', cfg, '--ckpt', ckpt, '--tag', tag, '--split', split,
           '--workers', str(WORKERS), '--json', os.path.join(JS, '%s_%s.json' % (tag, split))]
    return cmd + (['--ads'] if ads else [])


def build_jobs():
    jobs = []
    s0_ckpt = os.path.join(MODELS, SIM_STEM, 'S1', 'ckpt', 'best.pth')
    jobs.append(Job('S1', [lambda: train_cmd(SIM_CFG, 'S1', PRETRAIN_EPOCHS, 0, 'test', 5, 2, 'greedy')],
                    done=lambda: train_done(SIM_STEM, 'S1', PRETRAIN_EPOCHS), wait_files=[SIM_READY]))
    jobs.append(Job('Z_S1', [lambda: eval_cmd(s0_ckpt, 'Z_S1', 'val', ads=False),
                             lambda: eval_cmd(s0_ckpt, 'Z_S1', 'test', ads=True)],
                    deps=['S1'], done=lambda: os.path.exists(os.path.join(JS, 'Z_S1_test.json'))))
    for pct, iv, rep, ep, ve in BUDGETS:
        for seed in SEEDS:
            for arm in ('C', 'T'):
                tag = '%s_p%03d_s%d' % (arm, pct, seed)
                pre = s0_ckpt if arm == 'T' else None
                steps = [
                    (lambda tag=tag, ep=ep, seed=seed, iv=iv, rep=rep, ve=ve, pre=pre:
                     train_cmd(MAV_CFG, tag, ep, seed, 'val', 3, ve, 'top1', interval=iv, pretrained=pre, repeat=rep)),
                    (lambda tag=tag: eval_cmd(os.path.join(MODELS, 'mav6d', tag, 'ckpt', 'best.pth'), tag, 'test')),
                ]
                jobs.append(Job(tag, steps, deps=(['S1'] if arm == 'T' else []),
                                done=(lambda tag=tag: os.path.exists(os.path.join(JS, '%s_test.json' % tag)))))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--slots', type=int, default=2)
    ap.add_argument('--only', default=None, help='逗号分隔的作业名前缀')
    args = ap.parse_args()
    os.makedirs(LOG, exist_ok=True)
    os.makedirs(JS, exist_ok=True)
    os.chdir(TOOLS)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=ROOT)

    jobs = build_jobs()
    if args.only:
        pre = args.only.split(',')
        jobs = [j for j in jobs if any(j.name.startswith(p) for p in pre)]
    names = {j.name: j for j in jobs}
    if args.dry:
        for j in jobs:
            print('%-14s done=%s deps=%s' % (j.name, j.is_done(), j.deps))
            for s in j.steps:
                print('      ', ' '.join(s()))
        return 0

    pending = [j for j in jobs if not j.is_done()]
    say('作业 %d 个，未完成 %d 个，槽位 %d' % (len(jobs), len(pending), args.slots))
    warm('mav6d')
    warm('sim')
    running = []
    failed = set()
    while pending or running:
        # 回收
        for j in list(running):
            rc = j.proc.poll()
            if rc is None:
                continue
            j.log.close()
            if rc != 0:
                say('FAIL %s 第 %d 步 rc=%d（日志 %s）' % (j.name, j.step_i + 1, rc, os.path.join(LOG, j.name + '.log')))
                failed.add(j.name)
                running.remove(j)
                continue
            j.step_i += 1
            if j.step_i < len(j.steps):
                j.log = open(os.path.join(LOG, j.name + '.log'), 'a', encoding='utf-8')
                cmd = j.steps[j.step_i]()
                j.log.write('\n$ %s\n' % ' '.join(cmd))
                j.log.flush()
                j.proc = subprocess.Popen(cmd, stdout=j.log, stderr=subprocess.STDOUT, env=env, cwd=TOOLS)
                say('STEP %s %d/%d' % (j.name, j.step_i + 1, len(j.steps)))
            else:
                say('DONE %s' % j.name)
                running.remove(j)
        # 派发
        for j in list(pending):
            if len(running) >= args.slots:
                break
            if any(d in failed for d in j.deps):
                say('SKIP %s（依赖失败）' % j.name)
                failed.add(j.name)
                pending.remove(j)
                continue
            if any(d in names and not names[d].is_done() for d in j.deps):
                continue
            if any(not os.path.exists(f) for f in j.wait_files):
                continue
            if j.is_done():
                pending.remove(j)
                continue
            j.step_i = 0
            # 训练已完成（best.pth + 最后一轮验证记录）就直接从评测步开始
            if len(j.steps) == 2 and j.name[:2] in ('C_', 'T_'):
                pct = int(j.name[3:6])
                ep = [b[3] for b in BUDGETS if b[0] == pct][0]
                if train_done('mav6d', j.name, ep):
                    j.step_i = 1
            warm('sim' if j.name == 'S1' else 'mav6d')
            j.log = open(os.path.join(LOG, j.name + '.log'), 'a', encoding='utf-8')
            cmd = j.steps[j.step_i]()
            j.log.write('\n$ %s\n' % ' '.join(cmd))
            j.log.flush()
            j.proc = subprocess.Popen(cmd, stdout=j.log, stderr=subprocess.STDOUT, env=env, cwd=TOOLS)
            running.append(j)
            pending.remove(j)
            say('START %s（第 %d/%d 步）pid=%d' % (j.name, j.step_i + 1, len(j.steps), j.proc.pid))
        time.sleep(20)
    say('全部结束，失败 %d 个: %s' % (len(failed), sorted(failed)))
    subprocess.call([PY, 'report_camnorm.py'], env=env, cwd=TOOLS)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
