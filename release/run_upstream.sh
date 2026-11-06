#!/usr/bin/env bash
# 上游网络（定稿方法第 1–3 步 + 部署用的测距/朝向/高度网络）：从头重训。种子数保持研究时的设置（多为 5 个）。
# 改编自交接包 UAV3D_pack 的 scripts/ours_train.sh 第 1–9 步（同样的命令和参数）；去掉了"挪走自带结果"和出对比表的部分。
# 【未在本仓库里端到端跑过】—— 每条命令都和原始训练日志一致，但整条脚本没有从头连续跑过一次。
# 用法：bash release/run_upstream.sh            # 从第 1 步开始，已完成的步骤自动跳过
#       bash release/run_upstream.sh 5          # 从第 5 步开始
#       GPUS="0 1" bash release/run_upstream.sh
# 跑完后接着运行 bash release/run_final_3seed.sh（实例原型 + 弱增强自训练 3 种子 + 推理出表）。
set -e
source "$(dirname "$0")/env.sh"
START=${1:-1}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
G_ARR=($GPUS); NG=${#G_ARR[@]}
LOG=$RUNS/upstream/logs; mkdir -p "$LOG"
cd "$R/tools"

step() { [ "$1" -ge "$START" ] && [ ! -f "$RUNS/upstream/step$1.done" ]; }
done_() { touch "$RUNS/upstream/step$1.done"; echo "[$(date '+%F %T')] 第 $1 步完成：$2"; }
pmap() {  # pmap <日志名前缀> <命令模板，{} 换成种子> 种子...  —— 按卡并行
  local name=$1 tmpl=$2; shift 2; local i=0
  for it in "$@"; do
    CUDA_VISIBLE_DEVICES=${G_ARR[$((i % NG))]} bash -c "cd $R/tools && ${tmpl//\{\}/$it}" > "$LOG/${name}_$it.log" 2>&1 &
    i=$((i + 1)); (( i % NG == 0 )) && wait
  done
  wait
}
need() { for f in "$@"; do [ -s "$f" ] || { echo "缺少产出 $f，上一步失败了，看 $LOG"; exit 1; }; done; }
TR="$PY train.py --batch_size 8 --workers 4 --fix_random_seed --cudnn_benchmark --use_amp --val_interval 10 --val_every 100 --val_match greedy --skip_test_eval"
SEEDS="0 1 2 3 4"

# 1. 仿真检测器（距离 = 焦距 × 横向跨度 / 预测 2D 宽度）：RAMP 5 个种子 + 红外定位版 P0 5 个种子（第二段测距用它在仿真上抽裁窗）
if step 1; then
  pmap ramp "$TR --cfg_file cfgs/models/uavdet_3d/camnorm/sim_full_r34_geo_tr_mix_sym8_mblur_ramp.yaml --epochs 5 --val_split test --max_ckpt_save_num 10 --seed {} --extra_tag RAMPs{}" $SEEDS
  mkdir -p $R/output/release_20260920
  for k in $SEEDS; do $PY $R/release/strip_ckpt.py $M/sim_full_r34_geo_tr_mix_sym8_mblur_ramp/RAMPs$k/ckpt/checkpoint_epoch_5.pth $R/output/release_20260920/ramp_s$k.pth; done
  pmap p0 "$TR --cfg_file cfgs/models/uavdet_3d/camnorm/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_irloc.yaml --epochs 5 --val_split test --val_interval 100 --max_ckpt_save_num 1 --seed {} --extra_tag P0_s{}" $SEEDS
  for k in $SEEDS; do need $M/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_irloc/P0_s$k/ckpt/checkpoint_epoch_5.pth; done
  done_ 1 "仿真检测器"
fi

RAMPS=$(for k in 1 2 3 4; do printf "%s " $R/output/release_20260920/ramp_s$k.pth; done)
# 2. 第一轮自训练：仿真检测器给真实训练图打伪标签 → 筛选 → 只用伪标签微调 5 个种子
if step 2; then
  $PY annot_audit/dump_real_preds.py --split train --ckpt $R/output/release_20260920/ramp_s0.pth --ensemble $RAMPS --out ../output/camnorm/st/real_train_preds_rothead.pkl > $LOG/st2_dump.log 2>&1
  $PY make_pseudo_index.py --dump ../output/camnorm/st/real_train_preds_rothead.pkl --tag st2 > $LOG/st2_index.log 2>&1
  pmap st2 "$TR --workers 2 --cfg_file cfgs/models/uavdet_3d/camnorm/st2_mav6d_r34_geo_full_tr.yaml --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --val_split val --seed {} --extra_tag ST2_s{} --pretrained_model $R/output/release_20260920/ramp_s{}.pth" $SEEDS
  mkdir -p $R/output/release_20260924
  for k in $SEEDS; do $PY $R/release/strip_ckpt.py $M/st2_mav6d_r34_geo_full_tr/ST2_s$k/ckpt/checkpoint_epoch_3.pth $R/output/release_20260924/st2_s$k.pth; done
  ST2S=$(for k in 1 2 3 4; do printf "%s " $R/output/release_20260924/st2_s$k.pth; done)
  $PY annot_audit/dump_real_preds.py --split train --ckpt $R/output/release_20260924/st2_s0.pth --ensemble $ST2S --out ../output/camnorm/st/real_train_preds_st2.pkl > $LOG/st2_dump2.log 2>&1
  mkdir -p $G/final
  DUMP_PKL=$G/final/ens_val.pkl $PY $G/eval_dump.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml --ckpt $R/output/release_20260924/st2_s0.pth --ensemble $ST2S \
    --flip-tta --tag final_ens_val --json $G/final/ens_val.json --split val --interval 1 --workers 0 --match greedy --self-calib val --temporal-bg causal100 \
    --set MODEL.POST_PROCESSING.SIZE_SOURCE pred MODEL.POST_PROCESSING.SIZE2D_MODE width > $LOG/st2_ensval.log 2>&1
  need $R/output/camnorm/st/real_train_preds_st2.pkl $G/final/ens_val.pkl
  done_ 2 "第一轮自训练"
fi

# 3. 第二段测距网络的训练数据：仿真裁窗 + 红外剪影（仿真红外图阈值化），真实背景库，真实无标签裁窗
if step 3; then
  pmap softkd_extract "cd $G && $PY softkd_extract.py {}" $SEEDS
  (cd $G && $PY xpaste_bank.py) > $LOG/xpaste_bank.log 2>&1
  mkdir -p $G/irloc_train_recs
  pmap p0_train "DUMP_PKL=$G/irloc_train_recs/train_P0_s{}.pkl $PY $G/eval_dump.py --ckpt $M/sim_full_r34_geo_tr_mix_sym8_mblur_ramp_irloc/P0_s{}/ckpt/checkpoint_epoch_5.pth --tag irtrain_train_P0_s{} --json $G/irloc_train_recs/train_P0_s{}.json --cfg cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml --split train --interval 1 --workers 0 --match greedy --self-calib val --temporal-bg causal100 --set MODEL.POST_PROCESSING.SIZE_SOURCE pred MODEL.POST_PROCESSING.SIZE2D_MODE width" $SEEDS
  pmap selftrain_extract "cd $G && $PY selftrain_extract.py {}" $SEEDS
  need $G/xpaste/bank_bg.npy $G/softkd/train_s4_rgb.npy $G/selftrain/real_s4_rgb.npy
  done_ 3 "测距网络训练数据"
fi

# 4. 第二段测距网络 种子 0–4：V2P（红外剪影监督）→ V2Pp（真实域适配）。V2Q/V2Qp = 不用红外剪影的对照（只用于"测距弱"对比行）
if step 4; then
  for a in V2P V2Q; do pmap $a "cd $G && $PY secondstage_v2.py $a {}" $SEEDS; done
  for a in V2Pp V2Qp; do pmap $a "cd $G && $PY secondstage_v2.py $a {}" $SEEDS; done
  need $G/softkd/V2Pp_s4.pt
  done_ 4 "测距网络（种子 0–4）"
fi

ST2PS=$(for k in 1 2 3 4; do printf "%s " $M/st2plus_mav6d_r34_geo_full_tr/ST2P_s$k/ckpt/checkpoint_epoch_3.pth; done)
# 5. 第二轮自训练：伪标签距离用 V2Pp（种子 0–4）修正 → index_pseudo_st2plus.pkl（实例原型就在它上面再修）→ 训 5 个种子 → 5 集成出 val/test
#    （st2plus_ens 的检测框还用来给实例原型定裁窗位置，见 run_final_3seed.sh）
if step 5; then
  (cd $G && $PY make_st2plus_dump.py) > $LOG/st2plus_dump.log 2>&1
  $PY make_pseudo_index.py --dump ../output/camnorm/st/real_train_preds_st2plus.pkl --tag st2plus > $LOG/st2plus_index.log 2>&1
  pmap st2p "$TR --workers 2 --cfg_file cfgs/models/uavdet_3d/camnorm/st2plus_mav6d_r34_geo_full_tr.yaml --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 --val_split val --seed {} --extra_tag ST2P_s{} --pretrained_model $R/output/release_20260920/ramp_s{}.pth" $SEEDS
  mkdir -p $G/st2plus_ens
  for sp in test val; do
    DUMP_PKL=$G/st2plus_ens/ens_$sp.pkl $PY $G/eval_dump.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml --ckpt $M/st2plus_mav6d_r34_geo_full_tr/ST2P_s0/ckpt/checkpoint_epoch_3.pth --ensemble $ST2PS \
      --flip-tta --tag ens_ens_$sp.pkl --json $G/st2plus_ens/ens_$sp.json --split $sp --interval 1 --workers 0 --match greedy --self-calib val --temporal-bg causal100 \
      --set MODEL.POST_PROCESSING.SIZE_SOURCE pred MODEL.POST_PROCESSING.SIZE2D_MODE width > $LOG/st2plus_ens_$sp.log 2>&1
  done
  need $G/st2plus_ens/ens_test.pkl $G/st2plus_ens/ens_val.pkl $CN/train/index_pseudo_st2plus.pkl
  done_ 5 "第二轮自训练"
fi

# 6. 部署用测距网络 种子 5–9：V2P/V2Q → V2Pp/V2Qp；再用第二轮自训练的伪距离做 V2Rp（推理时用它）
if step 6; then
  S59="5 6 7 8 9"
  for a in V2P V2Q; do pmap $a "cd $G && $PY secondstage_v2.py $a {}" $S59; done
  for a in V2Pp V2Qp; do pmap $a "cd $G && $PY secondstage_v2.py $a {}" $S59; done
  $PY annot_audit/dump_real_preds.py --split train --ckpt $M/st2plus_mav6d_r34_geo_full_tr/ST2P_s0/ckpt/checkpoint_epoch_3.pth --ensemble $ST2PS --out ../output/camnorm/st/real_train_preds_st2p.pkl > $LOG/st2pp_dump.log 2>&1
  (cd $G && $PY make_pseudo_dump.py $R/output/camnorm/st/real_train_preds_st2p.pkl $R/output/camnorm/st/real_train_preds_st2pp.pkl 1.0 $G/st2plus_ens/ens_val.pkl) > $LOG/st2pp_pseudo.log 2>&1
  $PY make_pseudo_index.py --dump ../output/camnorm/st/real_train_preds_st2pp.pkl --tag st2pp > $LOG/st2pp_index.log 2>&1
  (cd $G && $PY ssst_targets.py) > $LOG/ssst_targets.log 2>&1
  pmap V2Rp "cd $G && $PY secondstage_v2.py V2Rp {}" $S59
  need $G/softkd/V2Rp_s9.pt
  done_ 6 "部署用测距网络（种子 5–9）"
fi

# 7. 朝向网络（不用红外/LiDAR 的版本）：KZ 5 种子 → AdaBN → 伪朝向自训练。定稿方法的朝向用第 8 步；这一步保留是因为原链路按这个顺序生成中间文件
if step 7; then
  pmap kz "$TR --workers 2 --cfg_file cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom.yaml --epochs 30 --max_ckpt_save_num 3 --logger_iter_interval 500 --val_split test --seed {} --extra_tag KZs{}" $SEEDS
  mkdir -p $R/output/release_20260923
  for k in $SEEDS; do $PY $R/release/strip_ckpt.py $M/sim_full_r34_kp_w12_zoom/KZs$k/ckpt/checkpoint_epoch_30.pth $R/output/release_20260923/kz_s$k.pth; done
  $PY $S/kzst1/dump_kz_members_kzst.py --split train --tag train5 > $LOG/kz_train5.log 2>&1
  # AdaBN：只重估 BN 统计量。原脚本会拿"原始权重的记录输出"做回放核对，重训后的权重自然对不上，所以跳过这项核对
  $PY -c "import sys; sys.path.insert(0, '$S/kzst1'); import adabn_kz as AB; AB.verify_replay = lambda *a, **k: {'pass': True, 'skipped': 'retrained'}; AB.main(['--force'])" > $LOG/kz_adabn.log 2>&1
  $PY $G/finalads/dump_members_st2p.py train adabn > $LOG/kz_dump_train.log 2>&1
  $PY $G/kzst4/kzst4_build.py > $LOG/kzst4_build.log 2>&1
  pmap kzst4 "$PY $G/kzst4/kzst4_train.py --student {}" $SEEDS
  for sp in val test; do $PY $G/kzst4/kzst4_eval.py dump $sp > $LOG/kzst4_dump_$sp.log 2>&1; done
  done_ 7 "朝向网络（不蒸馏版）"
fi

# 8. 朝向网络（定稿）：裁窗关键点 + 红外/LiDAR 辅助答案 KZM 5 种子 → 导出/AdaBN → 伪朝向自训练（kzst6 → kzst7 q 臂 = 定稿 5 成员）
if step 8; then
  pmap kzm "$PY $G/lidar/kzlid/train_mm.py --cfg_file cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_irlid.yaml --batch_size 8 --workers 2 --fix_random_seed --cudnn_benchmark --use_amp --epochs 30 --max_ckpt_save_num 3 --logger_iter_interval 500 --val_split test --val_interval 10 --val_every 100 --val_match greedy --skip_test_eval --seed {} --extra_tag KZMs{}" $SEEDS
  $PY $G/lidar/irlid/kzm_post.py export > $LOG/kzm_export.log 2>&1
  $PY $G/lidar/irlid/kzm_post.py adabn > $LOG/kzm_adabn.log 2>&1
  $PY $G/lidar/irlid/kzm_post.py dump train > $LOG/kzm_dump_train.log 2>&1
  $PY $G/kzst6/kzst6_build.py --tag st2p_train_kzma > $LOG/kzst6_build.log 2>&1
  pmap kzst6 "$PY $G/kzst6/kzst6_train.py --student {}" $SEEDS
  for sp in val test; do $PY $G/kzst6/kzst6_eval.py dump $sp > $LOG/kzst6_dump_$sp.log 2>&1; done
  $PY $G/kzst7/kzst7_build.py > $LOG/kzst7_build.log 2>&1
  pmap kzst7q "$PY $G/kzst7/kzst7_train.py --arm q --student {}" $SEEDS
  # 第二轮自训练检测器框上的朝向（st2p_<split>_kzt7q）：统一评测帧表 cmpeval/<split>_meta.npz 由它生成
  for sp in val test; do $PY $G/kzst7/kzst7_eval.py dump $sp q > $LOG/kzst7_dump_$sp.log 2>&1; done
  for k in $SEEDS; do need $G/kzst7/train_q/LOO_s$k/ckpt/checkpoint_epoch_3.pth; done
  need $REPL/st2p_test_kzt7q/kz_members_st2p_test_kzt7q.npz
  done_ 8 "朝向网络（定稿 5 成员）"
fi

# 9. LiDAR 关键点定高度：在 KZM 上加关键点头（目标来自仿真 LiDAR 模板）→ AdaBN（推理时读出高宽比，高度 = 比值 × 长宽均值）
if step 9; then
  [ -s $G/lidar/kpsize/templates.npy ] || (cd $G/lidar/kpsize && $PY lidar_templates.py) > $LOG/lidar_templates.log 2>&1   # 仿真 LiDAR → 每机型关键点模板
  pmap kps "$PY $G/lidar/kpsize/train_mmk.py --cfg_file cfgs/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_irlid_kps.yaml --batch_size 8 --workers 2 --fix_random_seed --cudnn_benchmark --use_amp --epochs 8 --max_ckpt_save_num 2 --logger_iter_interval 500 --val_split test --val_interval 10 --val_every 100 --val_match greedy --skip_test_eval --seed {} --extra_tag KPSs{} --pretrained_model $M/sim_full_r34_kp_w12_zoom_irlid/KZMs{}/ckpt/checkpoint_epoch_30.pth" 0 1
  for s in 0 1; do $PY $G/lidar/kpsize/kps_eval.py adabn $s > $LOG/kps_adabn_$s.log 2>&1; done
  done_ 9 "LiDAR 关键点定高度"
fi
echo "上游全部完成。下一步：bash release/run_final_3seed.sh"
