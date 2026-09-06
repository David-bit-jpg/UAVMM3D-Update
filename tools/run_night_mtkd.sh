#!/usr/bin/env bash
# 2026-09-06 夜间自主实验（用户：「你自己组织实验，我回来看看结果」）。方向 = 多模态蒸馏 + 多任务 + 迁移框架。
# 训练集 = mmcache_mix（3000 交叉贴增广 + 1436 原版白天帧，尺度对齐），与第四版 M0 完全同协议（12 轮、batch 8），
# 所以 M0 就是这些臂的「无蒸馏、无多任务」对照；C（从零）、B（near15 预训练）沿用旧结果。
#   阶段 1  T_mix 多模态教师(rgb+ir+depth+tag) ∥ MT_mix 多任务 RGB 学生（幻觉 ir/depth/tag）
#   阶段 2  S1_mix 蒸馏学生 ∥ S1MT_mix 蒸馏+多任务学生                      （教师 = T_mix）
#   阶段 3  MT / S1 / S1MT 各自迁 MAV6D 1%/5%/10%（朴素微调，同 B/C/M0 协议）
#           + RM0：M0 权重做「保留式微调」（centerdet_retain：冻结的 M0 当教师，特征保留项）
#           每对训练跑完先评测再开下一对（Windows 上 >=3 个 dataloader 进程会锁死）
#   阶段 4  零样本 + 曲线 + 汇总表
# 严格每次 2 个训练作业；有产物即跳过，可续跑。日志 output/bench_logs/night_mtkd.log
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
mkdir -p "$LOG" "$JS"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/night_mtkd.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
prewarm() { for f in rgb ir depth tag; do cat "$1/$2/$f.npy" > /dev/null; done; say "页缓存预热完成: $1/$2"; }

sim() {  # tag cfg extra...   源域 12 轮
  local tag=$1 cfg=$2; shift 2
  local dir=$OUT/mmcache/$(basename $cfg .yaml)/$tag
  [ -n "$(last_ckpt $dir)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag"
  "$PY" train.py --cfg_file $cfg --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --epochs 12 --logger_iter_interval 100 --extra_tag "$tag" "$@" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
xfer() {  # tag src_ckpt epochs interval [cfg [set-extra...]]   迁 MAV6D（与 run_paste_transfer.sh 同协议）
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
ev() {  # tag [cfg-dir]   评测（训练间隙串行）
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

[ -f "$MIX/READY" ] || { say "没有混合缓存 $MIX"; exit 1; }
M0_CK=$(last_ckpt $OUT/mmcache/student_rgb_mix/M0_mix)
[ -n "$M0_CK" ] || { say "没有 M0 ckpt"; exit 1; }
say "== 开始；M0=$M0_CK"
prewarm "$MIX" train

# ---------------- 阶段 1 ----------------
sim T_mix  $MM/teacher_mix.yaml &
sim MT_mix $MM/student_mt_mix.yaml &
wait
T_CK=$(last_ckpt $OUT/mmcache/teacher_mix/T_mix)
MT_CK=$(last_ckpt $OUT/mmcache/student_mt_mix/MT_mix)
[ -n "$T_CK" ] && [ -n "$MT_CK" ] || { say "阶段 1 失败：T=$T_CK MT=$MT_CK，停止"; exit 1; }
say "阶段 1 完成：T=$T_CK MT=$MT_CK"

# ---------------- 阶段 2 ----------------
sim S1_mix   $MM/student_kd_mix.yaml   --set MODEL.DISTILL.TEACHER_CKPT "$T_CK" &
sim S1MT_mix $MM/student_kdmt_mix.yaml --set MODEL.DISTILL.TEACHER_CKPT "$T_CK" &
wait
S1_CK=$(last_ckpt $OUT/mmcache/student_kd_mix/S1_mix)
S1MT_CK=$(last_ckpt $OUT/mmcache/student_kdmt_mix/S1MT_mix)
[ -n "$S1_CK" ] && [ -n "$S1MT_CK" ] || { say "阶段 2 失败：S1=$S1_CK S1MT=$S1MT_CK，停止"; exit 1; }
say "阶段 2 完成：S1=$S1_CK S1MT=$S1MT_CK"

# ---------------- 阶段 3：迁到 MAV6D ----------------
xfer MT_p01 "$MT_CK" 80 100 & xfer MT_p05 "$MT_CK" 40 20 & wait
ev MT_p01; ev MT_p05
xfer S1_p01 "$S1_CK" 80 100 & xfer S1_p05 "$S1_CK" 40 20 & wait
ev S1_p01; ev S1_p05
xfer S1MT_p01 "$S1MT_CK" 80 100 & xfer S1MT_p05 "$S1MT_CK" 40 20 & wait
ev S1MT_p01; ev S1MT_p05
xfer RM0_p01 "$M0_CK" 80 100 $MAVR MODEL.DISTILL.TEACHER_CKPT "$M0_CK" & xfer RM0_p05 "$M0_CK" 40 20 $MAVR MODEL.DISTILL.TEACHER_CKPT "$M0_CK" & wait
ev RM0_p01 centerdet_retain; ev RM0_p05 centerdet_retain
xfer MT_p10 "$MT_CK" 30 10 & xfer S1_p10 "$S1_CK" 30 10 & wait
ev MT_p10; ev S1_p10
xfer S1MT_p10 "$S1MT_CK" 30 10 & xfer RM0_p10 "$M0_CK" 30 10 $MAVR MODEL.DISTILL.TEACHER_CKPT "$M0_CK" & wait
ev S1MT_p10; ev RM0_p10 centerdet_retain

# ---------------- 阶段 4：零样本 + 曲线 + 汇总 ----------------
zero ZMT_mix "$MT_CK"
zero ZS1_mix "$S1_CK"
zero ZS1MT_mix "$S1MT_CK"
"$PY" plot_lowdata_curve.py --arms "C:scratch (real only),B:sim near15 pretrain,M0:mix pretrain,MT:mix multi-task,S1:mix KD,S1MT:mix KD+MT,RM0:M0 + retain FT" \
    --fracs p01,p05,p10 --out ../docs/results/lowdata_night.png > "$LOG/plot_night.log" 2>&1
"$PY" plot_lowdata_curve.py --arms "C:scratch (real only),B:sim near15 pretrain,M0:mix pretrain,MT:mix multi-task,S1:mix KD,S1MT:mix KD+MT,RM0:M0 + retain FT" \
    --fracs p01,p05,p10 --metric ang_median --out ../docs/results/lowdata_night_ang.png >> "$LOG/plot_night.log" 2>&1
(cd .. && "$PY" tools/summarize_bench.py --arms C,B,M0,MT,S1,S1MT,RM0 --fracs p01,p05,p10 --zero ZM0_mix,ZMT_mix,ZS1_mix,ZS1MT_mix --out output/bench_json/summary_night.md) 2>&1 | tee -a "$LOG/night_mtkd.log"
for t in S1MT_p05 MT_p05; do
  ck=$(last_ckpt $OUT/mav6d/centerdet/$t); [ -n "$ck" ] || continue
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$t" --vis 4 --vis-out ../output/vis_night_mav6d/$t > /dev/null 2>&1
done
say "ALL DONE"
