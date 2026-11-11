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
INIT = {}   # 11-07 组的起点权重（缺省 ramp_s{k}）
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
    run(['train.py', '--cfg_file', 'cfgs/models/uavdet_3d/camnorm/%s.yaml' % c, '--batch_size', '8', '--workers', os.environ.get('GP_WORKERS', '2'),
         '--fix_random_seed', '--cudnn_benchmark', '--use_amp', '--epochs', '3', '--max_ckpt_save_num', '1', '--logger_iter_interval', '200',
         '--skip_test_eval', '--seed', str(k), '--extra_tag', '%s_s%d' % (t, k),
         '--pretrained_model', INIT.get(name, ROOT + '/output/release_20260920/ramp_s%d.pth') % k]
        + (['--proto_cfg', SIMCFG, '--proto_weight', '0.1', '--proto_k', '30', '--proto_temp', '0.1'] + PROTO_EXTRA.get(name, []) if proto is True else [])
        + (['--proto_bank', CPBD + '/bank_s%d.pt' % k, '--proto_dist', 'l2', '--proto_weight', '0.1'] if proto in ('cpb', 'cpbs') else [])
        + (['--src_cfg', SIMCFG, '--src_weight', '1.0'] if proto == 'cpbs' else [])
        + (gp_args(proto, k) if isinstance(proto, str) and proto.startswith('gp:') else []),
        '%s_train_s%d.log' % (name, k), cwd=TOOLS, check=ck(name, k))


NOTBG = set()   # 11-07 晚：这些组推理时不做热图时序背景减除（严格单帧口径）


def infer(name):
    CK = [ck(name, k) for k in range(NSEED.get(name, 5))]
    os.makedirs(G + '/%s_ens' % name, exist_ok=True)
    for sp in ('test', 'val'):
        out = G + '/%s_ens/ens_%s.pkl' % (name, sp)
        run([G + '/eval_dump.py', '--cfg', REAL, '--ckpt', CK[0], '--ensemble'] + CK[1:] + ['--flip-tta', '--tag', 'ens_%s_%s' % (name, sp), '--json', out.replace('.pkl', '.json'),
             '--split', sp, '--interval', '1', '--workers', '0', '--match', 'greedy', '--self-calib', 'val'] + ([] if name in NOTBG else ['--temporal-bg', 'causal100']) + SETS,
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
# 11-06 对照：伪标签只做时间中位平滑（±5 / ±10 帧，不用剪影）+ 弱增强，3 种子
ARMS['wamed5'] = ('st2med5wa_mav6d_r34_geo_full_tr', False, 'WAMED5'); ARMS['wamed10'] = ('st2med10wa_mav6d_r34_geo_full_tr', False, 'WAMED10')
NSEED.update({'wamed5': 3, 'wamed10': 3, 'ntq': 3, 'ntwainst': 3})
ARMS['watp'] = ('st2tpwa_mav6d_r34_geo_full_tr', False, 'WATP')   # 11-07 多模态教师原型（单帧尺度匹配修伪距离）+ 弱增强
NSEED.update({'watp': 3})
ARMS['wapaste'] = ('st2pastewa_mav6d_r34_geo_full_tr', False, 'WAPASTE')   # 11-07 红外教师抠图混合（目标域原型定距离/位置）+ 弱增强
NSEED.update({'wapaste': 3})

# 11-07 红外/LiDAR 特点引导的 RGB 原型（gproto，train.py --gp_bank）：proto 字段 = 'gp:<库名>:<权重>:<把握阈值>:<w|now>'；库 = geo1004/gproto/bank_<库名>_s{k}.pt
def gp_args(proto, k):
    _, bank, w, thr, mode = proto.split(':')
    return ['--gp_bank', G + '/gproto/bank_%s_s%d.pt' % (bank, k), '--gp_weight', w, '--gp_temp', '0.1', '--gp_thr', thr] + (['--gp_now'] if mode == 'now' else [])
ARMS['gpk'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:kmeans:0.1:0:now', 'GPK')      # A 常规原型对照：pooling + 普通聚类，统一权重
ARMS['gpg'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:guided:0.1:0.3:w', 'GPG')      # B 完整：红外⇄LiDAR 互校分组 + 可信度加权 + 把握筛选
ARMS['gpgn'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:guided:0.1:0:now', 'GPGN')    # C = B 去加权与筛选
NSEED.update({'gpk': 3, 'gpg': 3, 'gpgn': 3})
ARMS['gpo'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:guidedord:0.1:0.3:w', 'GPO')    # B + 二维有序软目标（距离档×尺寸档）
ARMS['gpr'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:random:0.1:0:now', 'GPR')       # 对照：随机分组
ARMS['gpt'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:gtbin:0.1:0:now', 'GPT')        # 上限：仿真真值分组 + 有序
NSEED.update({'gpo': 3, 'gpr': 3, 'gpt': 3})
# 11-07 第 2 轮（修正版：去公共分量、温度 0.05、把握阈值 0.4，见 gproto/gp_bank4.py）
ARMS['gpg2'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:guidedc:0.1:0.4:w', 'GPG2')       # B 完整（修正）
ARMS['gpgn2'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:guidedc:0.1:0:now', 'GPGN2')     # C 去加权筛选（修正）
ARMS['gpo2'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:guidedordc:0.1:0.4:w', 'GPO2')    # B + 二维有序（修正）
ARMS['gpr2'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:randomc:0.1:0:now', 'GPR2')       # 对照：随机分组（修正）
ARMS['gpt2'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:gtbinc:0.1:0:now', 'GPT2')        # 上限：仿真真值分组（修正）
NSEED.update({'gpg2': 3, 'gpgn2': 3, 'gpo2': 3, 'gpr2': 3, 'gpt2': 3})
# 11-07 第 3 轮：教前/教后按簇加权原型（gproto/gp_bank_pp.py）
ARMS['gpp'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:pp:0.1:0.4:w', 'GPP')      # 学生从教前（ramp）起步
ARMS['gpq'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:ppq:0.1:0.4:w', 'GPQ')     # 学生从教后（KDPOST）起步
ARMS['wq0'] = ('st2pluswa_mav6d_r34_geo_full_tr', False, 'WQ0')                   # 对照：从教后起步、只弱增强、无原型
INIT['gpq'] = INIT['wq0'] = MD + '/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_kdpost/KDPOST_s%d/ckpt/checkpoint_epoch_1.pth'
NSEED.update({'gpp': 3, 'gpq': 3, 'wq0': 3})
# 11-07 第 4 轮（探索性，门槛未过：wa 后真实 test 相关只剩 0.04–0.07）：尺度歧义线有序原型（gproto/gp_bank_line.py）
ARMS['gpl'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:line:0.1:0:now', 'GPL')
NSEED.update({'gpl': 3})
# 11-07 第 5 轮：伪标签定组的条件原型（gproto/gp_bank_cond.py）——真实目标去哪个组由红外修正过的伪标签决定，不再自己给自己定组
ARMS['gpce'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:conde:0.1:0:w', 'GPCE')   # 误差档（老师判 RGB 错多少）+ 伪标签相对起点的修正量定档
ARMS['gpca'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:conda:0.1:0:w', 'GPCA')   # 绝对档（老师修正的距离×尺寸）+ 伪标签距离/尺寸定档
NSEED.update({'gpce': 3, 'gpca': 3})
# 11-07 第 6 轮：老师子空间锚定（gproto/gp_bank_anchor.py）——自训练只把真实目标在 4 个老师方向上的投影钉在起点（防止扔掉老师知识），无原型分配项
ARMS['gpan'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:anc:0:0:w', 'GPAN')
NSEED.update({'gpan': 3})
# 11-07 第 7 轮（文献启发：SSA 小子空间对齐 / 关系约束代替绝对拉拢，gproto/gp_bank_r7.py）
ARMS['gpcs'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:csub:0.1:0:w', 'GPCS')   # 绝对档条件原型，只在老师子空间（20 维）里对齐
ARMS['gprl'] = ('st2pluswa_mav6d_r34_geo_full_tr', 'gp:rel:0:0:w', 'GPRL')      # 不拉原型：红外读出方向成对次序 + LiDAR 尺寸读出对齐
NSEED.update({'gpcs': 3, 'gprl': 3})

# 11-07 晚：严格单帧口径——只弱增强自训练 wa 的同一组权重（标签 WA），推理不做热图时序背景减除（causal100 要求同段前 100 帧）
ARMS['wanb'] = ('st2pluswa_mav6d_r34_geo_full_tr', False, 'WA'); NSEED['wanb'] = 3; NOTBG.add('wanb')
# 11-07 夜：严格单帧伪标签（gproto/make_strict_pseudo.py SEL=lb）+ 弱增强自训练；推理也不做热图时序背景（与 wanb 比）
ARMS['wsf'] = ('st2sfwa_mav6d_r34_geo_full_tr', False, 'WSF'); NSEED['wsf'] = 3; NOTBG.add('wsf')
ARMS['wsfnb'] = ('st2sfnbwa_mav6d_r34_geo_full_tr', False, 'WSFNB'); NSEED['wsfnb'] = 3; NOTBG.add('wsfnb')   # 全严格单帧伪标签 + 无背景推理
ARMS['wfp'] = ('st2fpwa_mav6d_r34_geo_full_tr', False, 'WFP'); NSEED['wfp'] = 3; NOTBG.add('wfp')   # 11-07 夜：不动点伪距离（全严格单帧，第二段与不动点各半混合 + 弹性/分歧筛选）+ 无背景推理
ARMS['wr2'] = ('st2r2wa_mav6d_r34_geo_full_tr', False, 'WR2'); NSEED['wr2'] = 3; NOTBG.add('wr2')   # 11-07 夜：第二轮全严格伪标签（起点 wsfnb + 第二段 + 弹性筛 + 尺度锚定），与 wfp（一轮不动点）对照
ARMS['wfpT'] = ('st2fpTwa_mav6d_r34_geo_full_tr', False, 'WFPT'); NSEED['wfpT'] = 3; NOTBG.add('wfpT')   # 11-07 夜（归因）：不动点伪标签 + 各机型整体尺度平移回 ST2+（颜色原型分类），拆"逐帧更准"与"整体偏移"
ARMS['wr2T'] = ('st2r2Twa_mav6d_r34_geo_full_tr', False, 'WR2T'); NSEED['wr2T'] = 3; NOTBG.add('wr2T')   # 11-07 夜（归因）：第二轮伪标签 + 各机型整体尺度平移回 ST2+

# 11-10 无标签版（gproto/nolabel/PREREG.md）：距离修正系数 1 / 0.7 / 0.75 / 0.8，只按置信度筛帧，弱增强
for _t in ('nl100', 'nl070', 'nl075', 'nl080', 'nl050'):
    ARMS[_t] = ('st2%swa_mav6d_r34_geo_full_tr' % _t, False, _t.upper())
    NSEED[_t] = 3; NOTBG.add(_t)
