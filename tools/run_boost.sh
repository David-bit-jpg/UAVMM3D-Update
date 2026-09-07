#!/usr/bin/env bash
# 用户 2026-09-08：「一起来做吧，主要就是想要提高效果+迁移」——三件事一起上：
#   (1) MAV6D 微调加在线增广（翻转同步镜像姿态 / 随机尺度 / 光度），针对实测的旋转过拟合
#       （rot 训练损失 0.01 而测试角度中位 31°；15202 帧只来自 77 个序列）
#   (2) 迁移时把 rot 头也重置（原来只跳过 hm），仿真的姿态先验在低数据量下是负先验
#   (3) 源域预训练从 12 轮加到 30 轮（12 轮时 loss 仍在 1.1-1.7 震荡、域内召回只有 0.51-0.59）
# 严格每次 2 个训练作业；评测放在训练之间串行。
# 阶段 1（快反馈，用现有 12 轮权重）：5% 档 增广 / 增广+跳过 rot 各一条 -> 评测
# 阶段 2：30 轮预训练 T30 教师 ∥ M0_30 纯 RGB
# 阶段 3：S1MT_30（蒸馏+多任务，用 T30）
# 阶段 4：三个 30 轮权重迁 1%/5%/10%（增广 + 跳 hm rot）+ 从零对照（增广）-> 评测 -> ADS 全表
set -u
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
MM=cfgs/models/uavdet_3d/mmcache
MAV=cfgs/models/uavdet_3d/mav6d/centerdet.yaml
MIX=E:/data_collect/aug_paste_v1/mixed_cache
AUGSET="DATA_CONFIG.AUG.hflip 0.5 DATA_CONFIG.AUG.scale [0.85,1.2] DATA_CONFIG.AUG.photometric True DATA_CONFIG.AUG.noise 0.01"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/boost.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
prewarm() { for f in rgb ir depth tag; do cat "$1/$2/$f.npy" > /dev/null; done; say "页缓存预热完成"; }

# tag src_ckpt epochs interval skip...   （src_ckpt 为 - 表示从零）
xfer() {
  local tag=$1 src=$2 ep=$3 iv=$4; shift 4
  local dir=$OUT/mav6d/centerdet/$tag
  [ -n "$(last_ckpt $dir)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag"
  if [ "$src" = "-" ]; then
    "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
        --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
        --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" $AUGSET > "$LOG/$tag.log" 2>&1
  else
    "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
        --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
        --pretrained_model "$src" --pretrained_skip "$@" --pretrained_src_max_dis 40 \
        --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" $AUGSET > "$LOG/$tag.log" 2>&1
  fi
  say "DONE  $tag rc=$?"
}
ev() {
  local tag=$1
  [ -f "$JS/$tag.json" ] && return
  local ck; ck=$(last_ckpt $OUT/mav6d/centerdet/$tag)
  [ -n "$ck" ] || { say "eval $tag: 无 ckpt（看 $LOG/$tag.log）"; return; }
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差 |朝向 < 20' $LOG/eval_$tag.log | tr -s ' \n' ' ')"
}
sim() {  # tag cfg epochs extra...
  local tag=$1 cfg=$2 ep=$3; shift 3
  local dir=$OUT/mmcache/$(basename $cfg .yaml)/$tag
  [ -n "$(last_ckpt $dir)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag（$ep 轮）"
  "$PY" train.py --cfg_file $cfg --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --epochs "$ep" --logger_iter_interval 200 --extra_tag "$tag" "$@" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?  域内: $(grep -A2 'recall@2m' $LOG/$tag.log | tail -3 | tr -s ' \n' ' ')"
}

S1MT12=$(last_ckpt $OUT/mmcache/student_kdmt_mix/S1MT_mix)
[ -n "$S1MT12" ] || { say "缺 S1MT_mix 12 轮权重"; exit 1; }

# ---------------- 阶段 1：增广 / 跳 rot 的快反馈（5% 档）----------------
say "== 阶段 1：增广与跳过 rot 头的快反馈"
xfer A_scratch_p05  -         40 20 &
xfer A_S1MT_p05     "$S1MT12" 40 20 hm &
wait
ev A_scratch_p05; ev A_S1MT_p05
xfer A_S1MTr_p05    "$S1MT12" 40 20 hm rot &
wait
ev A_S1MTr_p05

# ---------------- 阶段 2：30 轮预训练 ----------------
say "== 阶段 2：30 轮源域预训练"
prewarm "$MIX" train
sim T30   $MM/teacher_mix.yaml     30 &
sim M0_30 $MM/student_rgb_mix.yaml 30 &
wait
T30=$(last_ckpt $OUT/mmcache/teacher_mix/T30)
M030=$(last_ckpt $OUT/mmcache/student_rgb_mix/M0_30)
[ -n "$T30" ] || { say "T30 失败，停止"; exit 1; }
say "T30=$T30  M0_30=$M030"

# ---------------- 阶段 3：蒸馏+多任务 30 轮 ----------------
sim S1MT_30 $MM/student_kdmt_mix.yaml 30 --set MODEL.DISTILL.TEACHER_CKPT "$T30" &
wait
S1MT30=$(last_ckpt $OUT/mmcache/student_kdmt_mix/S1MT_30)
[ -n "$S1MT30" ] || { say "S1MT_30 失败，停止"; exit 1; }
say "S1MT_30=$S1MT30"

# ---------------- 阶段 4：迁移 ----------------
say "== 阶段 4：30 轮权重迁 MAV6D（增广 + 跳 hm rot）"
xfer B_S1MT30_p05 "$S1MT30" 40 20 hm rot &
xfer B_M030_p05   "$M030"   40 20 hm rot &
wait
ev B_S1MT30_p05; ev B_M030_p05
xfer B_S1MT30_p01 "$S1MT30" 80 100 hm rot &
xfer A_scratch_p01 -        80 100 &
wait
ev B_S1MT30_p01; ev A_scratch_p01
xfer B_S1MT30_p10 "$S1MT30" 30 10 hm rot &
xfer A_scratch_p10 -        30 10 &
wait
ev B_S1MT30_p10; ev A_scratch_p10

say "== 汇总"
(cd .. && "$PY" tools/summarize_bench.py --arms A_scratch,B_S1MT30,Cn,S1MT --fracs p01,p05,p10 --out output/bench_json/summary_boost.md) 2>&1 | tee -a "$LOG/boost.log"
say "BOOST DONE"
