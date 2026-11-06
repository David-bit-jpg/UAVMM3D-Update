# -*- coding: utf-8 -*-
"""11-02 原型约束自训练：三组 × 5 种子，只换最终检测器 ST2P（同初始化 ramp_s{k}、同 3 轮），下游网络不动只重推理。
pf  = 只筛伪标签（置信度+原型相似度 综合分保留前 80%，index_pseudo_st2proto）
pl  = 只加原型 loss（伪标签不变，权重 0.1，30 个原型，温度 0.1）
pfl = 两者都加
先跑三组种子 0，再跑种子 1–4，然后逐组推理，最后和部署版、重训噪声对照比较。"""
import os, subprocess, sys, time
ROOT = '@@ROOT@@'; TOOLS = ROOT + '/tools'; G = ROOT + '/output/camnorm/geo1004'; O = G + '/proto'
PY = 'python'
MD = ROOT + '/output/models/uavdet_3d/camnorm'
SIMCFG = 'cfgs/models/uavdet_3d/camnorm/sim_full_r34_geo_tr_mix_sym8_mblur_ramp.yaml'
ARMS = {'pf': ('st2proto_mav6d_r34_geo_full_tr', False, 'PF'),
        'pl': ('st2plus_mav6d_r34_geo_full_tr', True, 'PL'),
        'pfl': ('st2proto_mav6d_r34_geo_full_tr', True, 'PFL'),
        # 11-03 老师方案「跨模态共识原型库」：伪标签不筛（同 ST2P），库来自仿真上 RGB/IR/LiDAR 三方 3D IoU 共识，L2 到最近原型
        'cpb': ('st2plus_mav6d_r34_geo_full_tr', 'cpb', 'CPB'),
        'cpbs': ('st2plus_mav6d_r34_geo_full_tr', 'cpbs', 'CPBS'),
        # 11-04 用户定 3 种子：伪标签过滤上限（真值滤掉 3D 完全不重叠）/ 不用真值的合成分过滤；部署版、噪声对照取同 3 种子重推理
        'orc': ('st2orc_mav6d_r34_geo_full_tr', False, 'ORC'),
        'cmb': ('st2cmb_mav6d_r34_geo_full_tr', False, 'CMB'),
        'dep3': ('st2plus_mav6d_r34_geo_full_tr', False, 'ST2P'),
        'nd3': ('st2plus_mav6d_r34_geo_full_tr', False, 'ND'),
        # 11-05 分项监督：伪距离换真值（上限）/ 真值错帧只关距离（上限）/ 合成分可疑帧只关距离（方法）
        'zgt': ('st2zgt_mav6d_r34_geo_full_tr', False, 'ZGT'), 'dmo': ('st2dmo_mav6d_r34_geo_full_tr', False, 'DMO'),
        'dmc': ('st2dmc_mav6d_r34_geo_full_tr', False, 'DMC'),
        # 11-05 弱化自训练数据增强（关光度+噪声）：无原型 / 原型 loss / 筛选+原型 loss
        'wa': ('st2pluswa_mav6d_r34_geo_full_tr', False, 'WA'), 'wapl': ('st2pluswa_mav6d_r34_geo_full_tr', True, 'WAPL'),
        'wapfl': ('st2protowa_mav6d_r34_geo_full_tr', True, 'WAPFL'),
        # 11-05 原型 loss 改进：距离门控（伪距离 ≥3.5 m 才算，仿真 <3.5 m 只占 14%）/ 干净仿真图建原型 / 两者
        'plg': ('st2plus_mav6d_r34_geo_full_tr', True, 'PLG'), 'plc': ('st2plus_mav6d_r34_geo_full_tr', True, 'PLC'),
        'plgc': ('st2plus_mav6d_r34_geo_full_tr', True, 'PLGC')}
ARMS['pl3'] = ('st2plus_mav6d_r34_geo_full_tr', True, 'PL')   # 已有 pl 的种子 0–2，3 种子集成作公平对照
PROTO_EXTRA = {'plg': ['--proto_min_dis', '3.5'], 'plc': ['--proto_clean'], 'plgc': ['--proto_min_dis', '3.5', '--proto_clean']}
ARMS['inst'] = ('st2inst_mav6d_r34_geo_full_tr', False, 'INST'); ARMS['wainst'] = ('st2instwa_mav6d_r34_geo_full_tr', False, 'WAINST')   # 11-05 实例原型修正伪标签（正常 / 弱增强）
ARMS['inst5'] = ARMS['inst']; ARMS['wainst5'] = ARMS['wainst']; ARMS['wa5'] = ARMS['wa']   # 11-05 补到 5 种子复核（种子 0–2 复用）
NSEED = {'orc': 3, 'cmb': 3, 'dep3': 3, 'nd3': 3, 'zgt': 3, 'dmo': 3, 'dmc': 3, 'wa': 3, 'wapl': 3, 'wapfl': 3, 'plg': 3, 'plc': 3, 'plgc': 3, 'pl3': 3, 'inst': 3, 'wainst': 3, 'inst5': 5, 'wainst5': 5, 'wa5': 5}
CPBD = O + '/cpb'
ENV = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=ROOT, TORCH_HOME='@@TORCH_HOME@@', PYTHONDONTWRITEBYTECODE='1')
REAL = 'cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml'
SETS = ['--set', 'MODEL.POST_PROCESSING.SIZE_SOURCE', 'pred', 'MODEL.POST_PROCESSING.SIZE2D_MODE', 'width']


def ck(name, k):
    c, _, t = ARMS[name]
    return '%s/%s/%s_s%d/ckpt/checkpoint_epoch_3.pth' % (MD, c, t, k)


def say(m):
    line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), m)
    print(line, flush=True)
    open(O + '/driver.log', 'a', encoding='utf-8').write(line + '\n')


def gpu_wait(need_free_mb=9000):
    while True:
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
            used, tot = (int(v) for v in out.strip().split(','))
            if tot - used >= need_free_mb:
                return
        except Exception:
            return
        time.sleep(60)


def run(args, log, cwd=G, env=None, check=None):
    if check and os.path.exists(check):
        say('跳过（已有）%s' % check)
        return
    gpu_wait()
    say('开始 %s → %s' % (' '.join(os.path.basename(a) for a in args[:3]), log))
    with open(O + '/' + log, 'a', encoding='utf-8') as f:
        r = subprocess.call([PY] + args, cwd=cwd, env=dict(ENV, **(env or {})), stdout=f, stderr=subprocess.STDOUT)
    if r:
        say('!! 失败 %s 退出码 %d' % (log, r))
        sys.exit(2)


def train(name, k):
    c, proto, t = ARMS[name]
    run(['train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % c, '--batch_size', '8', '--workers', '2',
         '--fix_random_seed', '--cudnn_benchmark', '--use_amp', '--epochs', '3', '--max_ckpt_save_num', '1', '--logger_iter_interval', '200',
         '--skip_test_eval', '--seed', str(k), '--extra_tag', '%s_s%d' % (t, k),
         '--pretrained_model', ROOT + '/output/release_20260920/ramp_s%d.pth' % k]
        + (['--proto_cfg', SIMCFG, '--proto_weight', '0.1', '--proto_k', '30', '--proto_temp', '0.1'] + PROTO_EXTRA.get(name, []) if proto is True else [])
        + (['--proto_bank', CPBD + '/bank_s%d.pt' % k, '--proto_dist', 'l2', '--proto_weight', '0.1'] if proto in ('cpb', 'cpbs') else [])
        + (['--src_cfg', SIMCFG, '--src_weight', '1.0'] if proto == 'cpbs' else []),
        '%s_train_s%d.log' % (name, k), cwd=TOOLS, check=ck(name, k))


def infer(name):
    CK = [ck(name, k) for k in range(NSEED.get(name, 5))]
    os.makedirs(G + '/%s_ens' % name, exist_ok=True)
    for sp in ('test', 'val'):
        out = G + '/%s_ens/ens_%s.pkl' % (name, sp)
        run([G + '/eval_dump.py', '--cfg', REAL, '--ckpt', CK[0], '--ensemble'] + CK[1:] + ['--flip-tta', '--tag', 'ens_%s_%s' % (name, sp), '--json', out.replace('.pkl', '.json'),
             '--split', sp, '--interval', '1', '--workers', '0', '--match', 'greedy', '--self-calib', 'val', '--temporal-bg', 'causal100'] + SETS,
            '%s_ensdump.log' % name, cwd=TOOLS, env={'DUMP_PKL': out}, check=out)
    for sp in ('val', 'test'):
        run([O + '/dump_orient_p.py', sp, name], '%s_orient_%s.log' % (name, sp),
            check='%s/output/camnorm/night1003/repl/%s_%s_kzt7q/kz_members_%s_%s_kzt7q.npz' % (ROOT, name, sp, name, sp))
    for sp in ('val', 'test'):
        run([G + '/lidar/kpsize/kps_shape_chain.py', 'shape', sp], '%s_kps_%s.log' % (name, sp), env={'UAV_POSTAG': name, 'UAV_SHAPESFX': '_' + name},
            check=G + '/lidar/kpsize/shape_%s_%s.pkl' % (sp, name))
    run([G + '/ablation/ablation.py', 'unc', 'V2Rp', name], '%s_unc.log' % name, env={'UAV_ENSDIR': name + '_ens'}, check=G + '/finalads/unc_val_%s.pkl' % name)
    say('%s 推理全部完成' % name)


def main():
    order = ['pl', 'pf', 'pfl']
    for k in range(5):
        for name in order:
            train(name, k)
        say('种子 %d 三组训练完' % k)
    for name in order:
        infer(name)
    run([O + '/compare_proto.py'], 'compare.log', env={'CUDA_VISIBLE_DEVICES': ''})
    say('全部结束')


if __name__ == '__main__':
    main()
# 11-06 无教师消融：伪标签第二段换 V2Qp（无红外剪影蒸馏）；ntq = 定稿配方（强增强、无原型）/ ntwainst = 弱增强 + 实例原型
ARMS['ntq'] = ('st2plusq_mav6d_r34_geo_full_tr', False, 'NTQ'); ARMS['ntwainst'] = ('st2instwaq_mav6d_r34_geo_full_tr', False, 'NTWAINST')
