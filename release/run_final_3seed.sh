#!/usr/bin/env bash
# 定稿方法的最后几步：实例原型修正伪标签距离 → 弱增强自训练检测器（种子 0、1、2）→ 3 集成推理（朝向/高度/测距）→ 推理时实例原型（等后 5 帧）→ 出表。
# 前提：release/run_upstream.sh 已跑完（或已有同名的上游产物）。
# 逻辑与研究时的驱动 output/camnorm/geo1004/proto/run_proto.py（train()/infer()，组名 wainst）、run_inst.py、inst/online_fuse.py 完全一致，
# 只是把 Python 驱动展开成了 shell 命令。【未端到端跑过】：各条命令在研究时分别跑过，这个脚本本身没有从头连续跑过。
# 用法：bash release/run_final_3seed.sh            # 已完成的步骤自动跳过
#       WITH_BASELINE=1 bash release/run_final_3seed.sh   # 另外算"原型之前"的对照行（第二轮自训练检测器种子 0–2，同样 3 集成）
#       GPUS="0 1 2" bash release/run_final_3seed.sh
set -e
source "$(dirname "$0")/env.sh"
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
G_ARR=($GPUS); NG=${#G_ARR[@]}
LOG=$RUNS/final/logs; mkdir -p "$LOG"
SEEDS="0 1 2"                       # 定稿：自训练检测器 3 个种子
REAL=cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml
SETS="--set MODEL.POST_PROCESSING.SIZE_SOURCE pred MODEL.POST_PROCESSING.SIZE2D_MODE width"
need() { for f in "$@"; do [ -s "$f" ] || { echo "缺少 $f（上游没跑完？看 $LOG）"; exit 1; }; done; }
say() { echo "[$(date '+%F %T')] $*"; }

# ---------- 0. 上游产物检查 ----------
need $CN/train/index_pseudo_st2plus.pkl $G/st2plus_ens/ens_val.pkl $G/st2plus_ens/ens_test.pkl
for k in $SEEDS; do need $R/output/release_20260920/ramp_s$k.pth; done
for k in 5 6 7 8 9; do need $G/softkd/V2Rp_s$k.pt; done
for k in 0 1 2 3 4; do need $G/kzst7/train_q/LOO_s$k/ckpt/checkpoint_epoch_3.pth; done
for sp in val test; do   # 统一评测帧表（真值只用于打分）
  [ -s $G/cmpeval/${sp}_meta.npz ] || (cd $G/cmpeval && CUDA_VISIBLE_DEVICES= $PY build_eval_crops.py $sp) > $LOG/eval_crops_$sp.log 2>&1
done

# ---------- 1. 实例原型修正训练伪标签的距离（只用 CPU） ----------
# 每个飞行片段：相机固定 → 时间中位数当背景，|帧 − 背景| = 无人机剪影；第 t 帧剪影当原型，与 t+1/2/3/5/10 帧做连续尺度匹配 → 帧间尺度比；
# 每片段最小二乘（伪标签距离 σ 5% + 尺度比 σ 1.2%，匹配质量 ≥ 0.8）→ 沿视线缩放伪标签位置 → index_pseudo_st2inst.pkl
if [ ! -s $CN/train/index_pseudo_st2inst.pkl ]; then
  say "实例原型：训练集剪影帧对（4 个进程）"
  for w in 0 1 2 3; do (cd $INST && CUDA_VISIBLE_DEVICES= $PY inst2_train.py $w 4) > $LOG/inst_pairs_w$w.log 2>&1 & done; wait
  (cd $INST && CUDA_VISIBLE_DEVICES= $PY make_inst_index.py) > $LOG/make_inst_index.log 2>&1
  need $CN/train/index_pseudo_st2inst.pkl
  tail -1 $LOG/make_inst_index.log
fi

# ---------- 2. 弱增强自训练：只留水平翻转（关光度/噪声），伪标签 = 实例原型修过的版本；从仿真检测器 ramp_s{k} 出发 3 轮 ----------
ck() { echo "$M/$1/$2_s$3/ckpt/checkpoint_epoch_3.pth"; }   # ck <配置名> <标签> <种子>
i=0
for k in $SEEDS; do
  c=$(ck st2instwa_mav6d_r34_geo_full_tr WAINST $k)
  [ -s "$c" ] && continue
  say "训练 弱增强+实例原型 种子 $k"
  CUDA_VISIBLE_DEVICES=${G_ARR[$((i % NG))]} bash -c "cd $R/tools && $PY train.py --cfg_file cfgs/models/uavdet_3d/camnorm/st2instwa_mav6d_r34_geo_full_tr.yaml \
    --batch_size 8 --workers 2 --fix_random_seed --cudnn_benchmark --use_amp --epochs 3 --max_ckpt_save_num 1 --logger_iter_interval 200 \
    --skip_test_eval --seed $k --extra_tag WAINST_s$k --pretrained_model $R/output/release_20260920/ramp_s$k.pth" > $LOG/wainst_train_s$k.log 2>&1 &
  i=$((i + 1)); (( i % NG == 0 )) && wait
done
wait
for k in $SEEDS; do need $(ck st2instwa_mav6d_r34_geo_full_tr WAINST $k); done

# ---------- 3. 推理：3 集成 + 翻转 + 时序热力图背景 + 自标定尺寸；朝向 5 成员；LiDAR 关键点定高度；测距 V2Rp ----------
infer() {  # infer <组名> <配置名> <标签>  —— 同 run_proto.infer()
  local n=$1 cfg=$2 tag=$3; local CK=() ; for k in $SEEDS; do CK+=($(ck $cfg $tag $k)); done
  mkdir -p $G/${n}_ens
  for sp in test val; do
    out=$G/${n}_ens/ens_$sp.pkl; [ -s $out ] && continue
    (cd $R/tools && DUMP_PKL=$out $PY $G/eval_dump.py --cfg $REAL --ckpt ${CK[0]} --ensemble ${CK[@]:1} --flip-tta --tag ens_${n}_$sp --json ${out%.pkl}.json \
      --split $sp --interval 1 --workers 0 --match greedy --self-calib val --temporal-bg causal100 $SETS) > $LOG/${n}_ensdump_$sp.log 2>&1
  done
  for sp in val test; do
    [ -s $REPL/${n}_${sp}_kzt7q/kz_members_${n}_${sp}_kzt7q.npz ] || (cd $G && $PY $O/dump_orient_p.py $sp $n) > $LOG/${n}_orient_$sp.log 2>&1
  done
  for sp in val test; do
    [ -s $G/lidar/kpsize/shape_${sp}_$n.pkl ] || (cd $G && UAV_POSTAG=$n UAV_SHAPESFX=_$n $PY $G/lidar/kpsize/kps_shape_chain.py shape $sp) > $LOG/${n}_kps_$sp.log 2>&1
  done
  [ -s $G/finalads/unc_val_$n.pkl ] || (cd $G && UAV_ENSDIR=${n}_ens $PY $G/ablation/ablation.py unc V2Rp $n) > $LOG/${n}_unc.log 2>&1
  need $G/finalads/unc_test_$n.pkl $G/finalads/unc_val_$n.pkl
  say "$n 推理完成"
}
infer wainst st2instwa_mav6d_r34_geo_full_tr WAINST
SPEC='wainst:弱增强+实例原型(3种子)'
if [ "${WITH_BASELINE:-0}" = 1 ]; then
  for k in $SEEDS; do need $(ck st2plus_mav6d_r34_geo_full_tr ST2P $k); done
  infer dep3 st2plus_mav6d_r34_geo_full_tr ST2P
  SPEC='dep3:原型之前的定稿模型(3种子),'$SPEC
fi

# ---------- 4. 验证/测试集的实例剪影帧对（裁窗位置来自第二轮自训练 5 集成的检测框，不看真值；只用 CPU） ----------
[ -s $INST/pairs2_test.pkl ] && [ -s $INST/pairs2_val.pkl ] || (cd $INST && CUDA_VISIBLE_DEVICES= $PY inst2.py val test) > $LOG/inst_pairs_valtest.log 2>&1

# ---------- 5. 推理时实例原型：第 t 帧用 [t−2000, t+5] 帧窗口（等后 5 帧）做同样的最小二乘，沿视线缩放最终位置 → 出表 ----------
(cd $INST && CUDA_VISIBLE_DEVICES= ONLINE_W=2000 ONLINE_LAG=5 CMP_SPEC="$SPEC" CMP_OUT=final3 $PY online_fuse.py) | tee $RUNS/final/online_final3.log
CUDA_VISIBLE_DEVICES= $PY $R/release/print_table.py $INST/online_final3.pkl | tee $RUNS/final/table_final3.md
say "完成：$RUNS/final/table_final3.md"
