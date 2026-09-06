#!/usr/bin/env bash
# 夜间实验 v2（2026-09-07 00:55 接手 run_night_mtkd.sh）：
# 第一版蒸馏（特征逐元素 L2）把域内召回从 0.51 打到 0.22/0.24（kd_feat 训到最后仍 ~390，GRAD_NORM_CLIP 之下检测项拿不到梯度），
# 主驱动在阶段 3 刚开 MT_p01/MT_p05 时被停掉（只杀了驱动 bash，两个训练进程留着）。本脚本：
#   等 MT_p01/MT_p05 跑完 → 评测 → 用 FEAT_MODE=cos 的配置重训 S1_mix ∥ S1MT_mix → 其余迁移对 → 零样本 → 曲线/汇总 → ALL DONE
# （run_night_phase5.sh 仍在等 ALL DONE）。旧的 v1 蒸馏权重与日志改名 *_v1mse 留作证据。
set -u
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
MM=cfgs/models/uavdet_3d/mmcache
MAV=cfgs/models/uavdet_3d/mav6d/centerdet.yaml
MAVR=cfgs/models/uavdet_3d/mav6d/centerdet_retain.yaml
MIX=E:/data_collect/aug_paste_v1/mixed_cache
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/night_mtkd.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
prewarm() { for f in rgb ir depth tag; do cat "$1/$2/$f.npy" > /dev/null; done; say "页缓存预热完成: $1/$2"; }
n_train() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match '$1' } | Measure-Object).Count" 2>/dev/null | tr -d '\r '; }

sim() {  # tag cfg extra...
  local tag=$1 cfg=$2; shift 2
  local dir=$OUT/mmcache/$(basename $cfg .yaml)/$tag
  [ -n "$(last_ckpt $dir)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag"
  "$PY" train.py --cfg_file $cfg --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --epochs 12 --logger_iter_interval 100 --extra_tag "$tag" "$@" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
xfer() {  # tag src_ckpt epochs interval [cfg [set-extra...]]
  local tag=$1 src=$2 ep=$3 iv=$4; shift 4
  local cfg=$MAV
  if [ $# -gt 0 ]; then cfg=$1; shift; fi
  local dir=$OUT/mav6d/$(basename $cfg .yaml)/$tag
  [ -n "$(last_ckpt $dir)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag"
  "$PY" train.py --cfg_file $cfg --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
      --pretrained_model "$src" --pretrained_skip hm --pretrained_src_max_dis 40 \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" "$@" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
ev() {  # tag [cfg-dir]
  local tag=$1 cd=${2:-centerdet}
  [ -f "$JS/$tag.json" ] && return
  local ck; ck=$(last_ckpt $OUT/mav6d/$cd/$tag)
  [ -n "$ck" ] || { say "eval $tag: 无 ckpt（训练失败？看 $LOG/$tag.log）"; return; }
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差 ' $LOG/eval_$tag.log | head -1)"
}
zero() {  # tag ckpt
  [ -f "$JS/$1.json" ] && return
  "$PY" eval_on_mav6d.py --ckpt "$2" --tag "$1" --decode-max-dis 40 --json "$JS/$1.json" > "$LOG/eval_$1.log" 2>&1
  say "zero-shot $1: $(grep -E '位置误差 ' $LOG/eval_$1.log | head -1)"
}

M0_CK=$(last_ckpt $OUT/mmcache/student_rgb_mix/M0_mix)
T_CK=$(last_ckpt $OUT/mmcache/teacher_mix/T_mix)
MT_CK=$(last_ckpt $OUT/mmcache/student_mt_mix/MT_mix)
[ -n "$M0_CK" ] && [ -n "$T_CK" ] && [ -n "$MT_CK" ] || { say "v2: 缺 ckpt M0=$M0_CK T=$T_CK MT=$MT_CK"; exit 1; }
say "== v2 接手；等 MT_p01/MT_p05 跑完"
until { grep -q "DONE  MT_p01" "$LOG/night_mtkd.log" && grep -q "DONE  MT_p05" "$LOG/night_mtkd.log"; } || [ "$(n_train 'extra_tag MT_p0')" = "0" ]; do sleep 60; done
say "MT_p01/MT_p05 结束"
ev MT_p01; ev MT_p05

# ---------------- 阶段 2（重训，FEAT_MODE=cos） ----------------
prewarm "$MIX" train
sim S1_mix   $MM/student_kd_mix.yaml   --set MODEL.DISTILL.TEACHER_CKPT "$T_CK" &
sim S1MT_mix $MM/student_kdmt_mix.yaml --set MODEL.DISTILL.TEACHER_CKPT "$T_CK" &
wait
S1_CK=$(last_ckpt $OUT/mmcache/student_kd_mix/S1_mix)
S1MT_CK=$(last_ckpt $OUT/mmcache/student_kdmt_mix/S1MT_mix)
[ -n "$S1_CK" ] && [ -n "$S1MT_CK" ] || { say "阶段 2(v2) 失败：S1=$S1_CK S1MT=$S1MT_CK，停止"; exit 1; }
say "阶段 2(v2) 完成：S1=$S1_CK S1MT=$S1MT_CK"
for t in S1_mix S1MT_mix; do say "域内 $t: $(grep -A2 'recall@2m' $LOG/$t.log | tail -3 | tr -s ' \n' ' ')"; done

# ---------------- 阶段 3 ----------------
xfer RM0_p01 "$M0_CK" 80 100 $MAVR MODEL.DISTILL.TEACHER_CKPT "$M0_CK" & xfer RM0_p05 "$M0_CK" 40 20 $MAVR MODEL.DISTILL.TEACHER_CKPT "$M0_CK" & wait
ev RM0_p01 centerdet_retain; ev RM0_p05 centerdet_retain
xfer S1_p01 "$S1_CK" 80 100 & xfer S1_p05 "$S1_CK" 40 20 & wait
ev S1_p01; ev S1_p05
xfer MT_p10 "$MT_CK" 30 10 & xfer RM0_p10 "$M0_CK" 30 10 $MAVR MODEL.DISTILL.TEACHER_CKPT "$M0_CK" & wait
ev MT_p10; ev RM0_p10 centerdet_retain
xfer S1MT_p01 "$S1MT_CK" 80 100 & xfer S1MT_p05 "$S1MT_CK" 40 20 & wait
ev S1MT_p01; ev S1MT_p05
xfer S1_p10 "$S1_CK" 30 10 & xfer S1MT_p10 "$S1MT_CK" 30 10 & wait
ev S1_p10; ev S1MT_p10

# ---------------- 阶段 4 ----------------
zero ZMT_mix "$MT_CK"
zero ZS1_mix "$S1_CK"
zero ZS1MT_mix "$S1MT_CK"
ARMS="C:scratch (real only),B:sim near15 pretrain,M0:mix pretrain,MT:mix multi-task,S1:mix KD,S1MT:mix KD+MT,RM0:M0 + retain FT"
"$PY" plot_lowdata_curve.py --arms "$ARMS" --fracs p01,p05,p10 --out ../docs/results/lowdata_night.png > "$LOG/plot_night.log" 2>&1
"$PY" plot_lowdata_curve.py --arms "$ARMS" --fracs p01,p05,p10 --metric ang_median --out ../docs/results/lowdata_night_ang.png >> "$LOG/plot_night.log" 2>&1
(cd .. && "$PY" tools/summarize_bench.py --arms C,B,M0,MT,S1,S1MT,RM0 --fracs p01,p05,p10 --zero ZM0_mix,ZMT_mix,ZS1_mix,ZS1MT_mix --out output/bench_json/summary_night.md) 2>&1 | tee -a "$LOG/night_mtkd.log"
for t in S1MT_p05 MT_p05; do
  ck=$(last_ckpt $OUT/mav6d/centerdet/$t); [ -n "$ck" ] || continue
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$t" --vis 4 --vis-out ../output/vis_night_mav6d/$t > /dev/null 2>&1
done
say "ALL DONE"
