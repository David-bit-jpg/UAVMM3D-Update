# -*- coding: utf-8 -*-
"""全量新仿真数据一条龙（2026-09-18）：选帧 -> 建缓存 -> 统计量 -> 自检 -> 训练 Q3 -> MAV6D 评测（预测尺寸 / 已知尺寸）。

数据：D:/data_collect 四场景主批次 + 仰拍 + _near + _near2，<=8 m 整帧，按序列 8:2。
每一步有产物就跳过，可断点续跑；自检不过直接停，不训练。脱离会话启动：
    PowerShell Start-Process python run_full_v2.py
"""
import datetime
import json
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(TOOLS, '..', 'output', 'camnorm', 'full_v2')
CACHE = 'E:/mmcache/full_v2'
SUBSET = 'cfgs/subsets/full_v2'
DS = 'cfgs/dataset_configs/uavdet_3d/'
MD = 'cfgs/models/uavdet_3d/camnorm/'
TAG = 'Q3'
TRAIN_CFG = MD + 'sim_full_r34_ratio.yaml'
EVAL_CFG = MD + 'mav6d_r34_ratio_full.yaml'


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


def tail(name, pat):
    p = os.path.join(OUT, name + '.log')
    return [l.strip() for l in open(p, encoding='utf-8', errors='replace') if pat in l][-3:]


def main():
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)
    log('==== 开始 ====')
    # 1. 选帧
    if not os.path.exists(os.path.join(TOOLS, SUBSET, 'near_test.txt')):
        if run([PY, 'build_near_subset.py', 'scan', '--root', 'D:/data_collect', '--cache', 'full_v2_scan.pkl'], 'subset_scan'):
            return
        if run([PY, 'build_near_subset.py', 'build', '--root', 'D:/data_collect', '--cache', 'full_v2_scan.pkl',
                '--max-dist', '8', '--mode', 'all', '--metric', 'z', '--out-dir', SUBSET], 'subset_build'):
            return
    # 2. 缓存
    for sp in ('train', 'test'):
        if not os.path.exists(os.path.join(CACHE, sp, 'index.pkl')):
            if run([PY, 'build_mm_cache.py', '--list', '%s/near_%s.txt' % (SUBSET, sp), '--split', sp, '--root', 'D:/data_collect',
                    '--out', CACHE, '--every', '3', '--max-label-range', '40', '--workers', '8', '--rgb-only',
                    '--intrinsic', 'auto', '--store', 'jpeg'], 'cache_' + sp):
                return
            log('  ' + ' | '.join(tail('cache_' + sp, '完成')))
    if not os.path.exists(os.path.join(CACHE, 'test', 'vis_score.npy')):
        if run([PY, 'mm_vis_score.py', '--cache', CACHE, '--splits', 'train', 'test'], 'vis_score'):
            return
    # 3. 统计量（只来自训练集本身）
    if run([PY, 'calc_cache_norm.py', CACHE, '--write', DS + 'camnorm_sim_full.yaml'], 'norm_stats'):
        return
    for target in (DS + 'camnorm_sim_full_ratio.yaml', DS + 'camnorm_mav6d_ratio_full.yaml'):
        if run([PY, 'calc_target_stats.py', '--cfg', TRAIN_CFG, '--write', target], 'target_stats_' + os.path.basename(target)[:-5]):
            return
    # 4. 自检：任何一项不过都不训练
    if run([PY, 'verify_camnorm.py', '--real', '--sim-cache', CACHE], 'verify_camnorm'):
        log('verify_camnorm 不通过，停止')
        return
    if run([PY, 'audit_domain_consistency.py', '--sim', CACHE, '--real', 'E:/mmcache/mav6d_cn'], 'audit_domain'):
        log('audit_domain_consistency 报错，停止')
        return
    # 5. 训练
    ckpt = os.path.join(TOOLS, '..', 'output', 'models', 'uavdet_3d', 'camnorm', 'sim_full_r34_ratio', TAG, 'ckpt', 'best.pth')
    if not os.path.exists(os.path.join(OUT, 'train_done_' + TAG)):
        rc = run([PY, 'train.py', '--cfg_file', TRAIN_CFG, '--batch_size', '8', '--workers', '2', '--fix_random_seed', '--seed', '0',
                  '--cudnn_benchmark', '--use_amp', '--epochs', '16', '--extra_tag', TAG, '--max_ckpt_save_num', '1',
                  '--logger_iter_interval', '500', '--val_split', 'test', '--val_interval', '5', '--val_every', '2',
                  '--val_match', 'greedy', '--skip_test_eval'], TAG + '.train')
        if rc:
            return
        open(os.path.join(OUT, 'train_done_' + TAG), 'w').close()
    # 6. 真实评测：预测尺寸 / 已知尺寸
    for src in ('pred', 'known'):
        tag = 'Z_%s_%s' % (TAG, src)
        run([PY, 'eval_camnorm.py', '--cfg', EVAL_CFG, '--ckpt', ckpt, '--tag', tag, '--split', 'test', '--workers', '0',
             '--json', os.path.join(OUT, 'json', tag + '_test.json'), '--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', src], tag)
        try:
            d = json.load(open(os.path.join(OUT, 'json', tag + '_test.json'), encoding='utf-8'))
            log('  %s: 位置中位 %.3f m | <0.2 m %.3f | 角度中位 %.1f° | uv %.1f px' % (
                tag, d['pos_median'], d['ACC_pos_0.2'], d['ang_median'], d['uv_median']))
        except Exception as e:   # noqa: BLE001
            log('  %s 读结果失败: %r' % (tag, e))
    log('ALL DONE')


if __name__ == '__main__':
    main()
