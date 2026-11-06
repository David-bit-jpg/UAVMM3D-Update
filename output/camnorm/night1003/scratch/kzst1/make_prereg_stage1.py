# -*- coding: utf-8 -*-
"""Writes prereg_stage1.json (+ .sha256) for KZ-ST v2 STAGE 1. Run ONCE, before the GPU jobs and before any ground truth
of this experiment is read. Refuses to overwrite. Every stage script checks this hash before it runs."""
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kzst_common as C   # noqa: E402

OUT = C.PREREG


def h(p):
    return C.sha256(p) if os.path.exists(p) else None


def build():
    ref_g = json.load(open(C.TEST_REF_GATED, encoding='utf-8'))
    ref_l = json.load(open(C.TEST_REF_LABELFREE, encoding='utf-8'))
    bounds = C.gate_bounds(ref_g, ref_l)
    return {
        'id': 'KZST-v2-STAGE1-20261003',
        'title': 'KZ-ST v2 第一段：train 成员 dump + 无标签闸门 + AdaBN 臂 + A0\'（纵横比校正解码）；第二段（LOO 训练）不在本登记内',
        'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'written_before_any_ground_truth_of_this_experiment_is_read': True,
        'source_protocol': {'file': 'scratchpad/idea_kzst.json', 'sha256': h(os.path.join(C.SCRATCH, 'idea_kzst.json')),
                            'judges': '两位裁判均判 run_if_time，只跑第一段（闸门、AdaBN、A0\'）；第二段等闸门通过再另行编码与登记'},
        'disclosures_before_registration': [
            '本次准备过程中没有读取任何 test 真值，也没有读取 train 真值。',
            '曾在终端看到 ST1 的 output/camnorm/st/logs/dump_train.log 末尾的真值汇总行（train 检出 84.4%、位置中位 0.319 m、折算角度中位 34.7°）；这三项早已写在 idea_kzst.json 的评审意见里，不进入任何阈值或选择。',
            '作者此前的 author_kzst/label_integrity_audit.py 读过 train 真值航向轴的分布汇总（idea 已如实登记），本段不再扩大此类读取。',
            '无标签地看过 test 上的角点跨度比（selftest t1，仅冒烟）：phantom4 全部层 (0.630, 0.698)、agree 层 (0.720, 0.730)；mavic2 全部层 (0.586, 0.504)、agree 层 (0.705, 0.505)。'
            '协议里的预期值 phantom4 (0.72,0.74)、mavic2 (0.73,0.52) 来自 agree 层；协议的规则本身不限分层。本登记照规则原文（g0 全部帧）冻结 rho，agree 层 rho 只作报告臂（见 A0p_agree_report_only），不参与 A0* 或任何判定。',
            '无标签地在 test 上做过 freeze 冒烟（selftest t5/t7：用 test 数据冒充 train、用 6 个多模态学生里的 5 个冒充 BN），只核对代码与耗时，没有读真值；'
            '其中用 test 自身 g0 帧算出的冒烟 rho 为 phantom4 (0.634, 0.702)、mavic2 (0.618, 0.515)，此时 A0p 相对 A0 航向改动 >5° 或倾斜改动 >2° 的帧占 4.2%（无标签计数）。真正的 rho 只来自 train dump。',
        ],
        'code': {'kz_fuse_offline.py': {'sha256': h(os.path.join(HERE, 'kz_fuse_offline.py')), 'equals_e1sa_copy': h(os.path.join(HERE, 'kz_fuse_offline.py')) == h(os.path.join(C.E1, 'kz_fuse_offline.py'))},
                 'dump_kz_members.py': {'sha256': h(os.path.join(HERE, 'dump_kz_members.py')), 'equals_e1sa_copy': h(os.path.join(HERE, 'dump_kz_members.py')) == h(os.path.join(C.E1, 'dump_kz_members.py'))},
                 'stage_scripts': ['dump_kz_members_kzst.py (S1 GPU)', 'kzst_stage1_freeze.py (S2 CPU, 无标签)', 'adabn_kz.py (S3 GPU)',
                                   'sim_diag_adabn.py (S7 GPU)', 'kzst_stage1_eval.py freeze (CPU, 无标签) / evaluate (CPU, 读 test 真值)',
                                   'kzst_common.py（全部定义）'],
                 'note': '各阶段输出 JSON 里记录当次脚本 sha256；本登记冻结的是定义与阈值'},
        'inputs_sha256': {
            'test_ref_gated.json': h(C.TEST_REF_GATED), 'test_ref_labelfree.json': h(C.TEST_REF_LABELFREE),
            'e1sa/kz_members_e1sa.npz': h(C.TEST_NPZ), 'e1sa/real_ens5flip_kpKZavg_zoom_e1sa.pkl': h(C.TEST_DUMP),
            'e1sa/frozen_e1sa.npz': h(C.TEST_FROZEN), 'e1sa/results_e1sa.json': h(C.TEST_RESULTS_E5),
            'ST1 train dump real_train_preds.pkl': h(C.ST1_TRAIN_DUMP), 'ST1 index_pseudo_st1.pkl': h(C.ST1_PSEUDO_INDEX),
            **{os.path.basename(p): h(p) for p in C.RELEASE_KZ}},

        'S1_train_member_dump': {
            'command': 'dump_kz_members_kzst.py --split train --tag train5（--interval 1）',
            'pipeline': '与 e1sa 的部署 dump 完全相同：release_20260920 ramp_s0..s4 五集成 + 水平翻转 TTA + causal100 在线热图时序背景 + val 上无标签自标定 lwh（conf>=0.7, interval 2）；KZ 放大裁窗 target_w2d 95、z 夹 [1,4]、search 2，从去畸变原图裁窗；5 个 RGB KZ 成员 release_20260923/kz_s0..s4.pth，部署值为 5 成员 kp2d 图逐格均值（逐位不变）；记录钩子为 e1sa/dump_kz_members.make_hook 的原样拷贝',
            'split': 'MAV6D train 划分全部 12,866 帧（61 序列：phantom4 31 / 5414 帧，mavic2 30 / 7452 帧），SAMPLED_INTERVAL 1',
            'fallback_interval_2': '仅当 GPU 太忙时用 --interval 2（6,433 帧）：时序背景按隔帧序列计算（与部署不完全一致，须在报告里声明）；闸门跳变按原始 2 帧间隔 = 排序间隔 1；g0 邻域仍按 ±3 个排序位（相当于 ±6 原始帧）；ST1 的 g0 集合对照与 (iii) 位置对照只在 interval 1 时做',
            'leak_barrier': 'split != test 时屏蔽 dump_real_preds 打印的真值汇总行（DUMP 行与检出率），dump 完成后 pkl 内 gt / matched 置 None、info.det_rate 置 None；所有下游脚本只经 load_dump(with_gt=False) 读取',
        },
        'replication_checks_stop_if_fail': {
            'top_at_hook_cell': '顶检测（argmax conf）不在钩子格的帧 <=0.5%（这些帧视为 no_det，各臂不改）',
            'pnp_failures': '部署 PnP 失败 <=0.5% 帧；离线 C0 失败数 <= 部署失败数 + 0.1%',
            'i_pnp_replica': '对 dump 记录的部署向量做 PnP，>=99.9% 帧与 dump 旋转测地差 <1e-3°（只在部署 PnP 成功的帧上比）',
            'i_offline_C0': '离线 C0（5 成员公共格）>=99.5% 帧 <0.05°',
            'v_cell': '离线 float64 重选的公共格与记录一致 >=99.9%',
            'iii_vs_ST1': 'interval 1 时，新 train dump 的 pred[:, :6]、conf 与 ST1 的 real_train_preds.pkl 逐帧完全相同',
            'g0_vs_ST1': 'interval 1 时，g0 集合应与 index_pseudo_st1.pkl 的接受帧集合相同（只比键，记录不阻断）',
        },

        'gate': {
            'decoder': '部署解码器（不做 rho 校正），与 test 参照同口径',
            'statistics': {
                'gated_layer': 'E5 五成员分层（kz_fuse_offline.plan_satm，公共读出格，KEEP_FOLD 45°，FLIP_RAW 90°）为 agree 或 flip_only 的帧',
                'gated_share': '该机型全部 dump 帧中门控层的占比（分母含 no_det，与 test_ref_labelfree 的 agree+flip_only 同口径）',
                'jump': '每条序列按帧号数字排序（文件名数字），取排序间隔 2（= 原始 2 帧）的帧对，两帧都在门控层时计 f180(hd(R_SATM[a]) - hd(R_SATM[b]))，hd = atan2(R[0,0], R[2,0])；该机型全部帧对取中位数（author_kzst/test_ref_gated.py 原样）',
                'pairwise': '门控层每帧 5 个成员两两 fold(rel_yaw) 的中位数，再对该机型门控帧取中位数',
            },
            'reference_values_test': {c: {'gated_share': bounds[c]['test_gated_share'], 'jump_gap2_median': bounds[c]['test_jump_gap2_median'],
                                          'pairwise_fold_median': bounds[c]['test_pairwise_fold_median']} for c in C.CLASSES},
            'pass_rule_per_class': 'train gated_share >= test - 0.10 且 jump <= 1.3 x test 且 pairwise <= 1.3 x test（含等号；未取整比较）',
            'bounds': {c: {'share_min': bounds[c]['share_min'], 'jump_max': bounds[c]['jump_max'], 'pairwise_max': bounds[c]['pairwise_max']} for c in C.CLASSES},
            'disposition': {'phantom4 过且 mavic2 过': 'train_both', 'phantom4 过 mavic2 不过': 'train_phantom4_only（mavic2 护栏改为遗忘检查）',
                            'phantom4 不过': 'no_LOO（第二段不做；AdaBN 与 A0\' 照常给出）'},
            'minimum_sample_sizes': {'rule': '对计划参训的入选集：总数 >= 800，序列数 >= 20，参训机型每类 >= 300；任一不满足 -> no_LOO_sample_size（同样中止第二段）',
                                     'values': C.MIN_SAMPLE},
            'selection': {'g0': 'tools/make_pseudo_index.py 第 1-2 步原样：顶检测 conf>=0.5；同序列按帧序 ±3 个排序位内其它通过 conf 的帧为邻居，邻居 <2 丢弃；与邻居位置中位数相距 >0.3 m 丢弃；另要求顶检测在钩子格',
                          'g1': 'D_rho 解码下的五成员 E5 分层为 agree 或 flip_only',
                          'cap': '每序列至多 80 帧，按时间均匀：排序位 round(linspace(0, n-1, 80))',
                          'classes': '只取处置允许参训的机型'},
            'labelfree_reports': '按机型/序列的帧数、两种解码器下的分层占比（含 trim、both）、老师（D_rho SA-TM）给出的视角分布在取帧前后的对比（侧视 <30°、沿光轴 >60°）、入选帧上 SA-TM 向量的刚体残差、D_rho 下的闸门统计（只报告）',
        },
        'rho': {
            'rule': '在 train 的 g0 帧（顶检测在钩子格）上，对 5 个成员各自（公共读出格、部署解码器）：自己 8 个角点在原图像素中的横向跨度与纵向跨度，分别除以「同一盒子 PROTO8*lwh（自标定常数）、放在位置网络中心 t、用该成员自己的 PnP 旋转」重投影后的跨度；按机型对全部 (帧, 成员) 取中位数 -> rho_c = (rho_u, rho_v)。PnP 失败或投影退化的成员跳过',
            'report_only_variant': '同样规则只取 agree 层得到 rho_agree，只用于报告臂 A0p_agree_report_only',
            'D_rho_decoder': '成员 16 维向量 x = kp/z（float64）的 u 分量除以 rho_u、v 分量除以 rho_v，再转 float32 走与部署相同的 PnP（同一盒子）；SA-TM 的成员 PnP、分层、对齐、融合全部在 D_rho 解码后的向量上做；rho=None 时与部署逐位相同（已单测）',
        },
        'stage2_inputs_frozen_now_not_used_tonight': {
            'LOO_teacher': '入选帧上，对 k=0..4：其余 4 个成员在 D_rho 下、读出格取这 4 个成员 hm 均值的窗内 argmax，做 SA-TM 得 R_T(k)；quotient 标志 = 该 4 成员 SA-TM 的保留集中有成员需要 Pi180 对齐',
            'note': '第二段的训练、目标、损失、判据见 idea_kzst.json；只有闸门通过且样本量满足时才另行编码，届时补登记',
        },

        'AdaBN_arm': {
            'members': 'release_20260923/kz_s0..s4.pth，各自独立重估',
            'reestimation_frames': 'train 划分（断言不含任何 test 序列）：g0 且顶检测在钩子格，每序列至多 80 帧按时间均匀，两机型全部序列；从不用 test、不用任何标签或伪标签',
            'crops': 'train 成员 dump 记录的部署裁窗 (x0, y0, w, h, s)，用 kp_zoom.RawSource 的去畸变与钩子同样的 warpAffine / 归一化重新裁出；先用原始权重回放，须复现记录的成员窗口（GPU 容差 kp 2e-3、hm 2e-4；train、test 各抽 24 帧，读出格不同的帧至多 max(1, n//50) 个），否则停止',
            'recipe': '全部 44 个 BatchNorm2d：reset_running_stats，momentum=None（逐批累计平均），只有 BN 层 train 模式，其余 eval，no_grad；批大小 16，顺序 = numpy default_rng(0) 的随机排列，5 个成员看同样的批；不更新任何权重（断言非 BN 缓冲张量与原权重逐位相同）',
            'test_inputs': 'e1sa 的 test 成员 dump 记录的 4800 个部署裁窗重新裁出，5 个 AdaBN 成员逐帧（batch 1）前向；读出格 = 5 成员 hm 均值在 5x5 窗内 argmax；部署向量 = kp 均值 / z；位置、检出、裁窗逐帧与 e1sa 相同（断言）',
            'arm_definition': 'BN = 5 个 AdaBN 成员、部署解码器（不做 rho）、SA-TM（与 A0 同一套规则）；某帧 PnP 失败时回退到 e1sa dump 中的部署旋转（计数报告）',
        },
        'A0_prime': {
            'definition': 'A0\' = e1sa test dump 的 5 个原始 KZ 成员，用 train 冻结的 rho_c（按该帧机型）经 D_rho 解码后做 SA-TM；CPU 计算，不训练',
            'A0': 'A0 = 部署的 5 成员 SA-TM（E5 冻结：12.60°），本段离线重算并断言与 frozen_e1sa 逐位相同',
            'A0_star': 'A0* = A0 与 A0\' 中全部帧航向折 180 中位数较低者（平局取 A0）',
            'A0p_vs_A0_report': 'Δ = A0\' − A0，序列聚类 bootstrap（B 10000，种子 20261003）95% 区间与单侧 p，分机型、逐序列；Δ<0 且上界 <0 只记为「不训练的解码改进」，交给第 2 条（形状自洽解码）一并判定，本段不据此部署',
        },
        'evaluation': {
            'metrics': 'E5 的 KZ_SUMMARY 口径（kz_fuse_offline.per_frame_metrics / summary_numbers / ads_indoor）：航向折 180 中位（summarize_dumps.stats 复核）、<10° / <20°、倾斜中位、不折叠 10°10cm、ADD(0.5d)、ADS indoor；分机型；greedy 1 m 匹配沿用 dump',
            'reproduction_asserts': 'A0 的航向/倾斜与 results_e1sa.json 的 SATM 相等（<1e-9）；5 个原始成员自峰值单模型 16.21/18.11/18.31/19.29/16.75（两位小数）；A0 的 ADS indoor 复现 50.88（±0.01）',
            'P1_BN': {
                'delta': 'Δ = median_he(BN) − median_he(A0*)，全部匹配帧',
                'criteria_all_required': ['Δ <= −1.0°',
                                          '两级 bootstrap（B 10000）95% 上界 < 0：每次先有放回抽 5 个种子编号，BN 与 A0* 的原始成员用同一组编号（重复计入）各自做 SA-TM（重抽集合的读出格 = 所抽成员 hm 均值的窗内 argmax；恰为 0..4 时用记录的部署格），再对 10 个序列有放回聚类抽样，统计量 = 两者航向中位差',
                                          '10 个序列中 >= 7 个 Δ_s < 0（严格）',
                                          '配对种子：>= 4/5 个 k 满足 BN_k 自峰值航向中位 < kz_s{k} 自峰值航向中位（同为部署解码器）',
                                          'Holm（alpha 0.05）与 P1（LOO）一起校正；第一段 LOO 不跑，其 p 记为 1，所以 P1-BN 的单侧 p 须 <= 0.025'],
                'p_one_sided': '(1 + #{两级 bootstrap 统计量 >= 0}) / (B + 1)',
            },
            'guardrails_BN_all_required_for_deployment': {
                'tilt': '倾斜中位 <= 8.52 + 0.3 = 8.82°', 'ads_indoor': '>= 50.88 − 0.3 = 50.58',
                'positions': '位置 / 检出 / 裁窗逐帧与 e1sa 相同（断言）', 'mavic2': 'mavic2 融合航向 <= 27.56 + 1.0 = 28.56°',
                'flip_rate': '未折叠翻转率（未折叠航向误差 >90° 的匹配帧占比）相对 A0* 上升 <= 3 个百分点',
                'view_strata': '真值视角分层（侧视 <30°、斜视、沿光轴 >60°；角度 = f180(真值 hd − 90)；critic_kzst/view_regime.py 原样）每层 Δ（相对 A0*）<= +1.0°',
                'gt_atypical': '真值非典型层（|真值 hd − 本序列真值 hd 的轴向 medoid| 折 180 >20°）：报告 n 与 Δ；n >= 200 时升为护栏，Δ <= +1.0°',
                'sim_forgetting': 'sim_diag_adabn.py：kp_arm_diag.run(sim_full_r34_kp_w12_zoom.yaml, parts rot, interval 4) 仿真 test 航向折 180 中位，每个种子 AdaBN_k <= 基线_k + 1.5°；基线 = 同次重测的原始 kz_s{k}（--with-orig）否则日志值 7.3/7.4/7.1/8.0/7.8；未跑则部署决定为 pending',
            },
            'deployment': 'P1-BN 全部判据 + 全部护栏 -> deploy_BN；仿真护栏未跑而其余全过 -> pending_sim_guardrail；否则不部署。A0\' 不在本段部署（见 A0_prime）',
            'report_only': ['A0p_agree_report_only（agree 层 rho）', 'BN_rho_report_only（BN 成员 + D_rho）', '按机型、按场景 01/02、按 E5 分层、按真值视角层的 Δ',
                            '序列内去均值（轴向均值）后预测航向与真值航向的皮尔逊相关', '各臂未折叠翻转率', 'BN 成员两两分歧 / KZ 成员两两分歧（<0.5 记为塌缩）',
                            'BN 与 KZ 成员的角点跨度比（无标签）', 'BN 统计量相对仿真统计量的偏移'],
        },
        'failure_interpretation': {
            'phantom4_gate_fail': '老师在 train 上太噪（与 ST1 的 31.6° 一致），用这位老师在 MAV6D train 上做旋转自训练不可行，本线关闭；BN 与 A0\' 结论照常给出',
            'BN_pass': 'AdaBN 单独就有真实域收益；若将来 LOO 不优于 BN，部署 BN',
            'BN_fail': '只换 BN 统计量不足以改善 KZ 的真实航向',
            'A0p_better': 'mavic2 的问题主要在解码时纵横比不一致，与训练无关，交给第 2 条处理',
        },
        'not_done': ['不训练任何权重（AdaBN 只换 BN 统计量）', '不改数据', '不在真实标签上选变体或超参', '不碰 @@ROOT@@ / @@MMCACHE@@ / @@MAV6D@@ 的任何文件（只读）'],
    }


def main():
    if os.path.exists(OUT):
        print('prereg_stage1.json 已存在，拒绝覆盖：', OUT)
        sys.exit(1)
    P = build()
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(P, f, ensure_ascii=False, indent=1)
    hh = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
    with open(OUT + '.sha256', 'w', encoding='utf-8') as f:
        f.write('%s  prereg_stage1.json\n' % hh)
    print('写出', OUT)
    print('sha256', hh)
    print('gate bounds', json.dumps(P['gate']['bounds']))


if __name__ == '__main__':
    main()
