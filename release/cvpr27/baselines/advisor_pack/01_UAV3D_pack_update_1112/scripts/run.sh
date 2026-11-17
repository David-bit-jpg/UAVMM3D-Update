#!/usr/bin/env bash
# 跑一个任务：bash scripts/run.sh <方法> <种子> <源域> <步骤> <目标域>
#   方法：  gupnet | monocd | monodgp | monodetr | monocop
#   源域：  in  = 室内 UE5 仿真 sim_train
#           out = 室外 CARLA 仿真 sim_outdoor_train
#           mix = sim_train + sim_outdoor_train + sim_outdoor_town10_kept（Town10 按画质筛过的帧；三个标签文件直接拼，不复制图像）
#   步骤：  sim = 只用源域训练 → 同一个模型预测两个目标域的测试集（目标域写 both）
#           st  = 用源域模型给目标域训练图打伪标签（单帧、分数前 80%）→ 源域 + 伪标签重训 → 预测该目标域测试集（路线 2：无标签自训练）
#           kd  = 伪标签换成我们 PCMD 的教师伪标签（红外 / LiDAR 知识）→ 源域 + 伪标签重训 → 预测（路线 3）
#   目标域：mav  = MAV6D（无标签训练图 real_train，测试 real_test），st / kd 的源域固定为 in
#           anti = AntiUAV3D 80 m 内（无标签训练图 outdoor_train，测试 outdoor_test），st / kd 的源域固定为 out
#           both = 只给 sim 用
#           antifu = AntiUAV3D 焦距统一版（11-12 新增，可选；先运行一次 python scripts/focal_unify.py）：
#                    sim 步骤 = 用已训好的 sim 模型预测 outdoor_test_fu（不重训）→ runs/<方法>_<源域>_simfu_s<种子>/
#                    st / kd 步骤 = 同 anti，但目标域训练图和测试图都换成焦距统一后的 → runs/<方法>_out_<st|kd>fu_anti_s<种子>/
#                    预测文件仍叫 pred_outdoor_test.pkl，用原测试真值打分（只改像素和内参，相机坐标系不变）
# 已完成的步骤自动跳过；断了重跑同一命令即可接着来。显存不够会自动把批大小减半重试（学习率按比例缩）。
set -e
source "$(dirname "$0")/env.sh"
m=$1; s=$2; src=$3; step=$4; tgt=$5
[ -n "$tgt" ] || { echo "用法：bash scripts/run.sh <方法> <种子> <in|out|mix> <sim|st|kd> <mav|anti|both>"; exit 1; }
H=$CF; mkdir -p "$RUNS"
case $src in
  in)  SRC_LABELS="$DATA/sim_train/labels.pkl" ;;
  out) SRC_LABELS="$DATA/sim_outdoor_train/labels.pkl" ;;
  mix) SRC_LABELS="$DATA/sim_train/labels.pkl $DATA/sim_outdoor_train/labels.pkl $DATA/sim_outdoor_town10_kept/labels.pkl" ;;
  *) echo "源域只能是 in / out / mix"; exit 1 ;;
esac
case $step in
  sim) [ $tgt = both ] || [ $tgt = antifu ] || { echo "sim 步骤的目标域写 both（一个模型测两个目标域）或 antifu"; exit 1; } ;;
  st|kd)
    case $tgt in
      mav)  [ $src = in ]  || { echo "MAV6D 的 st / kd 从 in 源域模型出发"; exit 1; }; TEST=real_test;    TGT=real_train ;;
      anti) [ $src = out ] || { echo "AntiUAV3D 的 st / kd 从 out 源域模型出发"; exit 1; }; TEST=outdoor_test; TGT=outdoor_train ;;
      antifu) [ $src = out ] || { echo "AntiUAV3D 的 st / kd 从 out 源域模型出发"; exit 1; }; TEST=outdoor_test_fu; TGT=outdoor_train_fu ;;
      *) echo "st / kd 的目标域只能是 mav / anti / antifu"; exit 1 ;;
    esac ;;
  *) echo "步骤只能是 sim / st / kd"; exit 1 ;;
esac
# 官方批大小；轮数按"和官方 KITTI 训练看到的样本数相同"（以 sim_train 为准）设定，换源域时轮数不变（见各方法 NOTES.md；GUPNet 每轮固定 3712 个样本）
case $m in
  gupnet)   EP=140; BS=${BATCH:-32}; BOPT=--batch-size ;;
  monocd)   EP=19;  BS=${BATCH:-8};  BOPT=--batch ;;
  monodgp)  EP=48;  BS=${BATCH:-8};  BOPT=--batch-size ;;
  monodetr) EP=37;  BS=${BATCH:-16}; BOPT=--batch-size ;;
  monocop)  EP=48;  BS=${BATCH:-16}; BOPT=--batch-size ;;
  *) echo "不认识的方法 $m"; exit 1 ;;
esac
final_ckpt() { case $m in gupnet) echo "$1/checkpoints/checkpoint_final.pth" ;; monocd) echo "$1/model_final.pth" ;; *) echo "$1/model_epoch_$EP.pth" ;; esac; }
resume_flag() {
  case $m in
    gupnet) c=$(ls -v "$1"/checkpoints/checkpoint_epoch_*.pth 2>/dev/null | tail -1); [ -n "$c" ] && echo "--resume $c" ;;
    monocd) [ -f "$1/model_checkpoint.pth" ] && echo "--resume" ;;
    *)      [ -f "$1/checkpoint.pth" ] && echo "--resume" ;;
  esac
}
train() {  # $1 = 输出目录，其余 = 标签文件
  out=$1; shift
  ck=$(final_ckpt "$out")
  [ -f "$ck" ] && { echo "已训练好：$ck"; return; }
  mkdir -p "$out"; bs=$BS
  while true; do
    echo "[$(date '+%F %T')] 训练 $m 种子 $s 源域 $src 步骤 $step 目标域 $tgt（$EP 轮，批 $bs）→ $out" | tee -a "$out/train.log"
    set +e
    bash "$H/$m/train.sh" --labels "$@" --out "$out" --seed "$s" --epochs $EP $BOPT $bs $(resume_flag "$out") ${SMOKE:+--max-iters 5} ${WORKERS:+--workers $WORKERS} >> "$out/train.log" 2>&1
    set -e
    [ -f "$ck" ] && return
    if tail -200 "$out/train.log" | grep -qi "out of memory" && [ $bs -gt 1 ]; then bs=$((bs / 2)); echo "显存不够，批大小减到 $bs 重试" | tee -a "$out/train.log"; continue; fi
    [ -n "$SMOKE" ] && return
    echo "训练失败，看 $out/train.log"; exit 1
  done
}
predict() {  # $1 = 模型  $2 = 数据划分  $3 = 输出
  [ -f "$3" ] && { echo "已有预测：$3"; return; }
  [ -n "$SMOKE" ] && [ ! -f "$1" ] && { c=$(ls -t "$(dirname "$1")"/checkpoint.pth "$(dirname "$1")"/model_*.pth "$(dirname "$1")"/checkpoints/*.pth 2>/dev/null | head -1); set -- "$c" "$2" "$3"; }   # 冒烟：用最后一个存档
  T=""; LIM=--limit; case $m in gupnet) T="--thresh 0"; LIM=--n ;; monocd) T="--thresh 0" ;; esac
  (cd "$H/$m" && $PY predict.py --ckpt "$1" --split "$2" --out "$3" $T ${SMOKE:+$LIM 20} ${WORKERS:+--workers $WORKERS})
}
src_model() {  # 源域模型：本包 runs/ 里训好的，或我们已训好随包附带的 pretrained/（只有 gupnet / monodgp 的 in 种子 0）
  c=$(final_ckpt "$RUNS/${m}_${src}_sim_s$s"); [ -f "$c" ] && { echo "$c"; return; }
  c=$(ls "$PKG/pretrained/${m}_${src}_sim_s$s"/checkpoints/checkpoint_final.pth "$PKG/pretrained/${m}_${src}_sim_s$s"/model_*.pth 2>/dev/null | head -1); [ -n "$c" ] && { echo "$c"; return; }
  [ -n "$SMOKE" ] && ls -t "$RUNS/${m}_${src}_sim_s$s"/checkpoint.pth "$RUNS/${m}_${src}_sim_s$s"/model_*.pth "$RUNS/${m}_${src}_sim_s$s"/checkpoints/*.pth 2>/dev/null | head -1
}
O0="$RUNS/${m}_${src}_sim_s$s"
case $step in
  sim)
    if [ $tgt = antifu ]; then   # 焦距统一版：不重训，等同一个 sim 模型训完后预测 outdoor_test_fu
      [ -f "$DATA/outdoor_test_fu/labels.pkl" ] || { echo "先运行一次：python scripts/focal_unify.py"; exit 1; }
      while [ -z "$(src_model)" ] && [ -z "$SMOKE" ]; do [ -f "$RUNS/queue/${m}_${src}_sim_both_s$s.fail" ] && { echo "源域模型训练失败，先修好它"; exit 1; }; echo "等 $m 种子 $s 源域 $src 的 sim 步骤训完…"; sleep 600; done
      OF="$RUNS/${m}_${src}_simfu_s$s"; mkdir -p "$OF"
      predict "$(src_model)" outdoor_test_fu "$OF/pred_outdoor_test.pkl"
    else
      train "$O0" $SRC_LABELS
      for TEST in real_test outdoor_test; do predict "$(final_ckpt "$O0")" $TEST "$O0/pred_${TEST}.pkl"; done
    fi ;;
  st|kd)
    O="$RUNS/${m}_${src}_${step}_${tgt}_s$s"
    if [ $tgt = antifu ]; then
      [ -f "$DATA/outdoor_train_fu/labels.pkl" ] || { echo "先运行一次：python scripts/focal_unify.py"; exit 1; }
      O="$RUNS/${m}_${src}_${step}fu_anti_s$s"
    fi
    mkdir -p "$O"
    if [ ! -f "$O/pseudo.pkl" ]; then
      if [ $step = st ]; then
        while [ -z "$(src_model)" ] && [ -z "$SMOKE" ]; do [ -f "$RUNS/queue/${m}_${src}_sim_both_s$s.fail" ] && { echo "源域模型训练失败，先修好它"; exit 1; }; echo "等 $m 种子 $s 源域 $src 的 sim 步骤训完…"; sleep 600; done
        predict "$(src_model)" $TGT "$O/pred_${TGT}_by_sim.pkl"
        $PY "$H/make_pseudo.py" "$O/pred_${TGT}_by_sim.pkl" "$O/pseudo.pkl" --labels "$DATA/$TGT/labels.pkl" ${SMOKE:+--smoke}
      else
        $PY "$H/build_pcmd_labels.py" "$DATA" "$O/pseudo.pkl" $TGT
      fi
    fi
    train "$O" $SRC_LABELS "$O/pseudo.pkl"
    predict "$(final_ckpt "$O")" $TEST "$O/pred_${TEST%_fu}.pkl" ;;   # 焦距统一版的预测也叫 pred_outdoor_test.pkl（用原测试真值打分）
esac
echo "[$(date '+%F %T')] 完成：$m 种子 $s 源域 $src 步骤 $step 目标域 $tgt"
