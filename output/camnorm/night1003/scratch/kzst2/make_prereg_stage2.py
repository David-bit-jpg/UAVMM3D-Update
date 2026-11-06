# -*- coding: utf-8 -*-
"""Writes @@ROOT@@/output/camnorm/night1003/kzst2/prereg_stage2.json (+ .sha256) BEFORE any stage-2 result exists
and before any stage-2 ground truth is read. Refuses to overwrite once any stage-2 output (pseudo labels / training /
test inference / evaluation) exists unless --force-before-results (only allowed while no training / eval output exists)."""
import sys

sys.dont_write_bytecode = True
import argparse      # noqa: E402
import glob          # noqa: E402
import hashlib       # noqa: E402
import json          # noqa: E402
import os            # noqa: E402
import time          # noqa: E402

import kzst2_common as S   # noqa: E402


def h(p):
    return S.sha256(p) if os.path.exists(p) else None


def build():
    s1 = S.KZST1
    gv = json.load(open(S.STAGE1['gate'], encoding='utf-8'))
    r1 = json.load(open(S.STAGE1['results'], encoding='utf-8'))
    code = {n: h(os.path.join(S.HERE, n)) for n in ('kzst2_common.py', 'kzst2_build.py', 'kzst2_torch.py', 'kzst2_train.py', 'kzst2_test_infer.py',
                                                     'kzst2_sim_diag.py', 'kzst2_eval.py', 'selftest_kzst2.py', 'make_prereg_stage2.py')}
    code.update({'kzst1/' + n: h(os.path.join(s1, n)) for n in ('kzst_common.py', 'kz_fuse_offline.py', 'adabn_kz.py', 'kzst_stage1_eval.py',
                                                                   'selftest_stage1.py', 'sim_diag_adabn.py')})
    P = {
        'id': 'KZST-v2-STAGE2-20261003',
        'title': 'KZ-ST v2 第二段：LOO 自训练 KZ（真实无标签 train 裁窗、解码器原像目标、只在 flip_only 帧用 180° 商损失），'
                 '主判据基线 A0** = min(A0*, BN)',
        'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'written_before_any_stage2_result_exists': True,
        'written_before_any_stage2_ground_truth_is_read': True,
        'source_protocol': {'file': 'scratchpad/idea_kzst.json', 'sha256': h(os.path.join(S.SCRATCH, 'idea_kzst.json')),
                            'version': 'KZ-ST v2（revised）第二段 = LOO 自训练'},
        'stage1': {'prereg_stage1_sha256': S.STAGE1_SHA['prereg_stage1.json'], 'gate_verdict_sha256': h(S.STAGE1['gate']),
                   'final_disposition': gv['final_disposition'], 'gate_pass': {c: gv['gate'][c]['pass'] for c in S.CLASSES},
                   'results_stage1_sha256': h(S.STAGE1['results']), 'RESULTS_stage1_md_sha256': h(os.path.join(s1, 'RESULTS_stage1.md')),
                   'A0_head': r1['table']['A0']['all']['head'], 'A0p_head': r1['table']['A0p']['all']['head'], 'A0_star': r1['A0_star']['arm'],
                   'BN_head': r1['table']['BN']['all']['head'], 'P1_BN_pass': r1['P1_BN']['pass'], 'P1_BN_p_one_sided': r1['P1_BN']['p_one_sided'],
                   'decision_BN': r1['decision_BN']},
        'coordinator_amendment_before_any_stage2_result': [
            '第一段 BN 臂（只在真实 train 裁窗上重估 BN 统计量）融合航向 9.06°，过了 P1 但仿真遗忘护栏失败（不部署）。第二段的自训练在真实裁窗上训练，'
            '必然同时把 BN 统计量换成真实域的，所以 H1 的主基线改为 A0** = min(A0*, BN)（全部帧融合航向中位较低者，平局取 A0*），按第一段结果 A0** = BN（9.06°）：'
            'LOO 学生必须以 1° 的余量胜过 AdaBN，否则任何增益都只是 BN 适应。对 A0*（=A0，12.60°）的比较保留为次要报告线。',
            '仿真遗忘：照第一段 sim_diag_adabn.py 的同一测量报告学生的数字，并在此写明处理方式（见 sim_forgetting）。',
            '学生起点按协议为原始 kz_s{k}；训练只用真实裁窗，所以训练中 BN 的 running stats 只来自真实裁窗（train 模式、momentum 0.1）。'],
        'disclosures_before_registration': [
            '写本登记前读过 RESULTS_stage1.md / results_stage1.json（A0 12.60°、A0\' 12.62°、A0* = A0、BN 9.06°、P1-BN 通过、仿真护栏失败、不部署 BN）。'
            '本段的全部阈值照抄协议，或来自协调者的修订（主基线 A0**、仿真遗忘的处理），没有一项是看了第二段结果后定的——第二段此时没有任何结果。',
            '没有读取任何 train 真值；没有读取第二段的任何 test 真值。只在合成夹具（伪造真值）上跑过 evaluate；在真实 test 输入上只跑过无标签的 freeze'
            '（用 6 个多模态 KZ 学生中的 5 个冒充 LOO，只为核对代码与耗时）。',
            '冒烟（CPU、全部无标签）：40 帧伪标签 + 裁窗（LOO 老师与第一段冻结值逐位相同，往返 PnP 误差 <1e-5°），在 20 张训练裁窗上画过伪角点，'
            '5 个学生各训 5 iter，test 推理 8 帧（原始 KZ 回放复现记录窗口），仿真 BN 重估 32 帧（测量用 mock）。',
            '训练用的伪标签全集（3021 帧）在登记之后才生成（kzst2_build.py，无标签）。'],
        'code_sha256_at_registration': code,
        'inputs_sha256': {
            'frozen_train_stage1.npz': S.STAGE1_SHA['frozen_train_stage1.npz'], 'kz_members_train5.npz': S.STAGE1_SHA['kz_members_train5.npz'],
            'real_ens5flip_kpKZavg_zoom_train5.pkl (gt scrubbed)': S.STAGE1_SHA['real_ens5flip_kpKZavg_zoom_train5.pkl'],
            'frozen_test_stage1.npz': h(S.STAGE1['frozen_test']), 'adabn/kz_members_adabn_test.npz': h(S.STAGE1['bn_npz']),
            'adabn/sim_diag_adabn.json': h(S.STAGE1['bn_sim']),
            'e1sa/kz_members_e1sa.npz': h(S.TEST_NPZ), 'e1sa/real_ens5flip_kpKZavg_zoom_e1sa.pkl': h(S.TEST_DUMP),
            'e1sa/frozen_e1sa.npz': h(S.TEST_FROZEN), 'e1sa/results_e1sa.json': h(S.TEST_RESULTS_E5),
            **{'kz_s%d.pth' % k: h(S.RELEASE_KZ[k]) for k in range(5)}},
        'rho': {'rule': '第一段规则原文版（g0 全部帧、逐成员跨度比的中位数），不是 agree 层的只报告版',
                'values': gv['rho'], 'not_used': {'rho_agree_report_only': gv['rho_agree_report_only']},
                'D_rho': '成员 16 维向量 x = kp/z 的 u 分量除以 rho_u、v 分量除以 rho_v，再走部署 PnP（盒子 = 自标定常数 lwh）；kzst1/kzst_common.RhoCtx'},
        'training_frames': {
            'definition': '第一段冻结的入选帧 sel_idx（frozen_train_stage1.npz）：g0（ST1 规则）∧ 顶检测在钩子格 ∧ D_rho 下五成员 E5 分层为 agree 或 flip_only'
                          ' ∧ 处置 train_both 的两机型，每序列至多 80 帧按时间均匀',
            'n': 3021, 'per_class': gv['sample_size']['per_class'], 'n_seq': gv['sample_size']['n_seq'], 'split': 'MAV6D train（断言不含任何 test 序列）',
            'per_student_validity': '帧 i 进入学生 k 的训练集当且仅当：(a) 学生 k 的 LOO 老师（其余 4 成员 SA-TM）PnP 没有回退（否则不存在 LOO SA-TM，'
                                    '回退链的最后一级还含有 k 自己）；(b) 伪框中心投影在裁窗内；(c) 目标往返（裁窗内参 PnP、部署解码器几何 PnP）都 <0.01°。'
                                    '各学生的有效帧数写进 pseudo_stage2.json'},
        'teacher_and_targets': {
            'LOO_teacher': '学生 k 的伪旋转 R_T(k) = 其余 4 个 KZ 成员在 D_rho 下、读出格取这 4 个成员 hm 均值的窗内 argmax，做 SA-TM（KFO.fuse_subset）；'
                           '重新计算并断言与 frozen_train_stage1.npz 的 loo_RT / loo_quotient / loo_stratum 逐位相同',
            'quotient_flag': '第一段冻结的定义：该 4 成员 SA-TM 的保留集里有成员需要 Pi180 对齐（flip_only 型分裂）-> 用 L_Q；否则用普通 L1',
            'target': "g'(k) = diag(rho_c) · g(R_T(k))；g(R) = 刚体框 (R, t = 位置网络中心, s = 部署 PnP 盒子 0.3489×0.3487×0.1996) 8 个角点"
                      '在【部署裁窗】相机里的投影相对投影中心的偏移 / 64（KZ kp2d 约定；proj(t) 与解码器中心 uv 相同，断言到 1e-6 px）。'
                      '在整幅输入单位里正好是 z·diag(rho)·x16，x16 经 D_rho + PnP 回到 R_T（往返 <0.01°）',
            'pair_permutation': 'P = [4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11]（D = diag(-1,-1,1) 作用于 PROTO8 的 P180 换到通道），与 diag(rho) 和水平翻转的重新编号都可交换',
            'kp2d_map': '数据集写 32 通道 [g\', P g\']（商损失帧）或 [g\', g\']（其他帧），只在中心格（KP_NEIGHBOR 0）；'
                        'hm / center_res / center_dis / dim / rot 目标由仓库 DataPreProcessor（center_point_encoder_mav6d）在同一伪框上生成',
            'loss': "kp2d 项 = Σ_格 min(|p − g'_a|₁, |p − g'_b|₁) / (16·格数)；两半相同时等于原 L1（16 通道路径一字不动，CenterHeadKZST 运行时注册，不改仓库）",
            'loss_weights': S.LOSS_WEIGHTS},
        'crops': {
            'rule': '训练裁窗 = train 成员 dump 记录的部署裁窗 (x0,y0,w,h,s)，从 1920×1080 原图经 kp_zoom.RawSource 去畸变后按钩子语句 warpAffine 重切（kzst1/adabn_kz.recut），'
                    '512×288，uint8 BGR 无损存为 kzst2/data/crops_train_u8.npy；归一化用 MAV6D NORM（与部署钩子相同）；各向同性（sx == sy 断言）',
            'student_noise': 'hflip 0.5（图像翻转 + hflip_K + R -> M R S，目标随之镜像与重新编号）+ 光度 + 噪声 0.01（仓库 MMCache_Det_Dataset._augment 原样）',
            'VIEW_AUG': '不使用（见 ambiguities_resolved a）'},
        'recipe': dict(S.TRAIN, aug=S.AUG, loss_weights=S.LOSS_WEIGHTS,
                       iterations='3 轮 × ceil(N_k / 8)（drop_last False；N_k = 学生 k 的有效帧数，约 3 × 378 = 1134），上限 2500（断言）',
                       base_config='cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom.yaml（KZ 的训练配置）+ 上述覆盖；AUG 整体替换并断言与本登记逐键相同',
                       no_validation='不传 --val_split，不做任何测试集评测；固定取最后一个 checkpoint（checkpoint_epoch_3.pth）'),
        'test_inference': 'test 4800 帧：位置网络、检出、时序背景、z、裁窗全部取自 e1sa 部署 dump（逐帧断言相同），KZ 输入裁窗只依赖这些，故从原图按钩子语句重切，'
                          '5 个学生逐帧（batch 1）前向，记录 5×5 窗口，读出格 = 5 学生 hm 均值窗内 argmax；先用原始 kz_s{k} 回放 24 帧须复现记录窗口（GPU 容差 kp 2e-3、hm 2e-4）',
        'arms': {
            'LOO': '5 个 LOO 学生经 D_rho 解码、SA-TM 融合（主臂）',
            'A0': '部署 KZ 5 种子 SA-TM（12.60°，断言与 frozen_e1sa 逐位相同）', "A0'": 'A0 的成员经 D_rho 解码后 SA-TM',
            'A0*': "A0 与 A0' 中全部帧航向中位较低者（第一段：A0）",
            'BN': '第一段 AdaBN 臂（部署解码器，SA-TM；断言与第一段 frozen_test 逐位相同）',
            'A0**': 'A0* 与 BN 中全部帧航向中位较低者（平局取 A0*）；按第一段结果为 BN（9.06°）',
            'report_only': ['LOO_dep（学生 + 部署解码器）', 'BN_rho（AdaBN + D_rho）']},
        'H1_primary_P1': {
            'statement': 'LOO 的 SA-TM 融合航向（折 180 中位，test 全部匹配帧）≤ A0** − 1.0°',
            'criteria_all_required': [
                'Δ = median_he(LOO) − median_he(A0**) ≤ −1.0°',
                '两级 bootstrap（B 10000，种子 20261004+1）95% 上界 < 0：每次先有放回抽 5 个种子编号，LOO 与 A0** 的成员用同一组编号（LOO_k 与 BN_k 都源自 kz_s{k}）'
                '各自做 SA-TM（读出格 = 所抽成员 hm 均值的窗内 argmax，恰为 0..4 时用记录格），再对 10 个序列有放回聚类抽样，统计量 = 两者航向中位差',
                '10 个序列中 ≥ 7 个 Δ_s < 0（严格）',
                '配对种子：≥ 4/5 个 k 满足 LOO_k 自峰值航向中位 < A0** 第 k 成员自峰值航向中位，两边都用同一解码器 D_rho（A0** = BN 时比 BN_rho 的单成员）',
                'Holm（α 0.05）族 = {P1-LOO（本条）, P1-BN（第一段 p = %.4f）}；单侧 p = (1 + #{bootstrap 统计量 ≥ 0}) / (B + 1)' % r1['P1_BN']['p_one_sided']]},
        'secondary_report_line': 'LOO 对 A0*（协议原 P1）：同样的 Δ、两级 bootstrap（种子 20261004+2）、序列数、配对种子（对 kz_s{k}，同为 D_rho），只报告「按原判据会否通过」，不参与决定',
        'guardrails_all_required_for_deployment': {
            'tilt': '倾斜中位 ≤ 8.52 + 0.3 = 8.82°', 'ads_indoor': '≥ 50.88 − 0.3 = 50.58',
            'positions': '位置 / 检出 / 裁窗逐帧与 e1sa 相同（断言）', 'mavic2': 'mavic2 融合航向 ≤ 27.56 + 1.0 = 28.56°（两机型都训）',
            'flip_rate': '未折叠翻转率（未折叠航向误差 > 90° 的匹配帧占比）相对 A0* 上升 ≤ 3 pp',
            'view_strata': '真值视角分层（侧视 <30°、斜视、沿光轴 >60°；f180(真值 hd − 90)；第一段定义原样）每层 Δ（相对 A0*）≤ +1.0°；另报告相对 A0** 的值',
            'gt_atypical': '真值非典型层（|真值 hd − 本序列轴向 medoid| 折 180 > 20°）：n ≥ 200 时 Δ（相对 A0*）≤ +1.0°（第一段 n = 1003，故为护栏）',
            'sim_forgetting': 'H_simBN(LOO_k) ≤ H_simBN(kz_s{k}) + 1.5° 对每个 k（见 sim_forgetting）'},
        'sim_forgetting': {
            'measurement': 'annot_audit/kp_arm_diag.run(sim_full_r34_kp_w12_zoom.yaml, ckpt, parts rot, interval 4)：仿真 test 航向折 180 中位（第一段 sim_diag_adabn.py 同一测量）',
            'A_own_stage1_procedure': '学生 checkpoint 原样（BN running stats 是真实裁窗统计量）。照第一段报告：对日志基线 7.3/7.4/7.1/8.0/7.8 的差、第一段规则（≤ 基线 + 1.5）的通过与否、'
                                      '以及对同种子 AdaBN_k（第一段 10.61/11.28/11.38/10.88/12.25）的差。只报告，不单独否决部署。',
            'B_simBN_decision': '把模型的 BN running stats 在仿真 TRAIN 帧上重估（第一段 AdaBN 配方：reset、momentum None、只 BN 为 train 模式、no_grad、batch 16、'
                                'default_rng(0) 顺序；帧 = 仿真 train 划分、KZ 评测视图 VAL_ZOOMS 2.5、MIN_RGB_VIS 5 过滤后每 n 帧取 1，n = N // 4000 = 18101 // 4000 = 4，共 4526 帧），'
                                '再做同一测量；对 LOO_k 和原始 kz_s{k} 都这样做（同批次）。护栏：每个 k 都 H_simBN(LOO_k) ≤ H_simBN(kz_s{k}) + 1.5°。',
            'reasoning': '只换 BN 统计量造成的仿真退化是可逆的：换回该域的统计量即恢复（AdaBN_k 的权重与 kz_s{k} 逐位相同，B 下两者完全相同）。'
                         '训练过的权重则不可逆。B 去掉可逆部分，只留下权重改变带来的遗忘，这才是协议这条护栏要防的「忘了仿真里学到的朝向函数」。'
                         'A 与 B 的差即 BN 统计量部分。这一处理是对协议原文（学生原样 ≤ kz_s{k} + 1.5）的修订，依据是第一段结果与协调者指示，登记于任何第二段结果之前。',
            'not_run': '若仿真诊断未跑，部署决定为 pending_sim_guardrail'},
        'decision': {
            'deploy_LOO': 'H1 主判据全部满足 ∧ 全部护栏（含 B）满足',
            'pending_sim_guardrail': '其余全过、仿真诊断未跑',
            'do_not_deploy_LOO': '其他情况',
            'BN': '第一段的 BN 决定（do_not_deploy_BN）本段不改；只在报告里注明：按 B 的处理，BN 的仿真护栏按构造通过（权重未变），是否据此重议由用户 / 协调者决定',
            'A0p': "A0' 只是解码参照，不在本段部署"},
        'report_only': ['LOO 对 A0* 的完整次要报告线', '按机型 / 场景 01、02 / E5 分层 / 真值视角层的 Δ（相对 A0* 与 A0**）',
                        '序列内去轴向均值后预测航向与真值航向的皮尔逊相关（各臂）', '各臂未折叠翻转率',
                        'LOO 学生两两分歧 / KZ 成员两两分歧（同为 D_rho；< 0.5 记为塌缩）', '学生角点跨度比（部署解码器）与 rho 目标之比（≈1 表示学到了目标尺度）',
                        'LOO_dep（部署解码器）与 BN_rho（D_rho）两条只报告臂；LOO − BN_rho（同解码器）的单级 bootstrap',
                        '每种子自峰值表（kz、kz_rho、bn、bn_rho、loo_rho、loo_dep）', '训练损失分项（前 / 后 50 iter）、各学生帧数与商损失帧数',
                        '老师标签的视角分布在取帧前后的对比（第一段无标签报告）', '仿真诊断 A 全部数字'],
        'failure_interpretation': {
            'primary_pass': '伪标签在 BN 适应之外带来了 ≥1° 的额外信息（护栏全过则部署 LOO）',
            'primary_fail_secondary_pass': '对 A0 的增益主要来自真实域 BN 统计量，伪标签没有在 BN 之上确立额外增益；不部署 LOO',
            'LOO_worse_than_BN': '伪标签训练抵消了 BN 适应的收益（老师噪声 / 确认偏差），这位老师在 train 上不能教旋转',
            'single_seed_better_fusion_not': '集成知识被蒸进各成员（看两两分歧比），不部署',
            'simBN_fail': '学生的权重不可逆地忘了仿真里的朝向函数',
            'own_fail_simBN_pass': '仿真退化只是可逆的 BN 统计量部分（与 AdaBN 同性质）',
            'view_or_atypical_fail': '学生向 train 视角分布 / 序列典型姿态偏移'},
        'ambiguities_resolved': [
            'a 裁窗：协议写「VIEW_AUG 与 KZ 相同（t_range [1.15,1.75]）」、学生用随机裁窗；第二段任务明确要求「与部署完全相同的裁窗（同窗口 / 同放大倍数，从原图裁）」。'
            '按任务要求用记录的部署窗口，不加随机裁窗；学生侧的扰动只剩 hflip / 光度 / 噪声。训练输入分布因此与 test 推理输入完全同分布。',
            'b 衍生数据：协议原计划把原图去畸变后以 JPEG q95 写到 @@MMCACHE@@/mav6d_raw_kzst；本段不许写 @@MMCACHE@@，改为把部署裁窗本身无损存到 kzst2/data（协议风险 7 随之消失）。',
            'c 「每帧约 3 遍」：取 3 轮 × ceil(N_k/8)（drop_last False），与 ceil(3N/8) 至多差 2 次迭代；上限 2500 断言。',
            'd workers：协议写 2；任务要求 0 或 1（Windows 并发锁死）；取 0，主进程 numpy 随机数由 set_random_seed(k) 决定。',
            'e LOO 老师回退：4 成员 SA-TM 的 PnP 失败时 fuse_subset 会依次回退到 4 成员 C0、5 成员 C0（后者含学生自己）；这样的 (帧, 学生) 对不训练。',
            'f 商损失标志用第一段冻结定义（LOO 老师保留集中有成员需 Pi180 对齐），与该 4 成员子集的分层（含 trim_only / both）无关；其余帧普通 L1。',
            "g 目标中心：g(R) 相对伪框中心 t 的投影（= 解码器中心 uv，断言 1e-6 px）；hm / center_res 目标在同一投影中心。",
            'h LOO 臂用 D_rho 解码（目标是 D_rho 的原像）；部署解码器版本只报告。',
            'i 配对种子两边都用 D_rho（协议原文「同一解码器 D_rho」）：主判据比 BN_rho 单成员，次要线比 kz_rho 单成员。',
            'j Holm 族 = {P1-LOO 主判据, P1-BN}，P1-BN 的 p 取第一段（0.0032）；BN 的部署决定不在本段重议。',
            'k 视角层、翻转率、非典型层护栏仍相对 A0*（协议原文，协调者要求其余照旧）；相对 A0** 的值只报告。',
            'l 仿真遗忘的处理见 sim_forgetting（协调者修订）。',
            'm test 推理用「部署裁窗重切 + 学生前向」代替整条 dump_real_preds：KZ 的输入只依赖已记录的位置网络输出，二者等价；用原始 KZ 回放复现记录窗口作硬性检查（第一段 AdaBN 臂同法）。',
            'n 第二段 bootstrap 用独立种子 20261004（第一段 20261003）。',
            'o 学生从原始 kz_s{k} 起步（协议），不是从 AdaBN 权重；训练后不另做 BN 重估，直接取最后一个 checkpoint。']}
    return P


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--force-before-results', action='store_true')
    a = ap.parse_args(argv)
    os.makedirs(S.OUT, exist_ok=True)
    produced = [p for p in [S.PSEUDO_JSON] + glob.glob(os.path.join(S.TRAIN_DIR, 'LOO_s*', 'DONE.json')) +
                [os.path.join(S.TEST_DIR, 'kz_members_loo_test.json'), os.path.join(S.SIM_DIR, 'sim_diag_loo.json'),
                 os.path.join(S.EVAL_DIR, 'frozen_test_stage2.json'), os.path.join(S.EVAL_DIR, 'results_stage2.json')] if os.path.exists(p)]
    if os.path.exists(S.PREREG) and not a.force_before_results:
        raise SystemExit('%s exists -- the registration is frozen' % S.PREREG)
    if produced:
        raise SystemExit('stage-2 outputs already exist (%s) -- refusing to (re)register' % produced)
    P = build()
    with open(S.PREREG, 'w', encoding='utf-8') as f:
        json.dump(P, f, ensure_ascii=False, indent=1)
    hh = hashlib.sha256(open(S.PREREG, 'rb').read()).hexdigest()
    open(S.PREREG + '.sha256', 'w', encoding='utf-8').write('%s  %s\n' % (hh, os.path.basename(S.PREREG)))
    print('[prereg] wrote %s sha256 %s' % (S.PREREG, hh))
    return hh


if __name__ == '__main__':
    main()
