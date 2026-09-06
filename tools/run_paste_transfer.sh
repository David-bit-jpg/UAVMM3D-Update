#!/usr/bin/env bash
# 用户 2026-09-06：「生成的 3000 张近距离模拟 RGB + 筛选出的约 2000 张原版白天帧 -> 训 RGB 检测器 -> 迁到 MAV6D 试效果」。
# 等 run_paste_aug_batch.sh 打包出 READY 后自动接：合并缓存 -> 训 M0（纯 RGB，12 轮）-> MAV6D 1%/5%/10% 微调 -> 评测 + 零样本
# -> 低数据量曲线（与已有的 A=从零、B=near15 仿真预训练 同图）。严格每次 2 个训练作业；batch 8。
set -u
AUG=E:/data_collect/aug_paste_v1
SRC=E:/mmcache/paste_src
MIX=$AUG/mixed_cache
PY=/d/Miniconda3/envs/city/python.exe
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
cd /e/Open3DUAVDet/tools || exit 1
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
MM=cfgs/models/uavdet_3d/mmcache
MAV=cfgs/models/uavdet_3d/mav6d/centerdet.yaml
mkdir -p "$LOG" "$JS"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/paste_transfer.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
prewarm() { for f in rgb ir depth tag; do cat "$1/$2/$f.npy" > /dev/null; done; say "页缓存预热完成: $1/$2"; }

until [ -f "$AUG/cache/train/READY" ]; do sleep 120; done
say "增广缓存就绪"

# ---- 合并：3000 增广 + 原版白天可见帧（<=2000） ----
if [ ! -f "$MIX/READY" ]; then
  say "合并缓存"
  "$PY" merge_mmcaches.py --out "$MIX" --input "$AUG/cache:0:0:aug" --input "$SRC:5:2000:orig" --test-from orig --test-n 100 2>&1 | tail -4 | tee -a "$LOG/paste_transfer.log"
fi

# ---- 阶段 1：纯 RGB 学生 M0 在混合集上训练 ----
sim() {  # tag cfg extra...
  local tag=$1 cfg=$2; shift 2
  say "START $tag"
  "$PY" train.py --cfg_file $cfg --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --epochs 12 --logger_iter_interval 100 --extra_tag "$tag" "$@" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
prewarm "$MIX" train
M0_CK=$(last_ckpt $OUT/mmcache/student_rgb_mix/M0_mix)
if [ -z "$M0_CK" ]; then
  sim M0_mix $MM/student_rgb_mix.yaml
  M0_CK=$(last_ckpt $OUT/mmcache/student_rgb_mix/M0_mix)
fi
say "M0=$M0_CK"

# ---- 阶段 2：迁到 MAV6D（与 run_kd_pipeline.sh 同协议）----
xfer() {  # tag src_ckpt epochs interval
  local tag=$1 src=$2 ep=$3 iv=$4
  [ -n "$(last_ckpt $OUT/mav6d/centerdet/$tag)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag"
  "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
      --pretrained_model "$src" --pretrained_skip hm --pretrained_src_max_dis 40 \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
scratch() {  # tag epochs interval
  local tag=$1 ep=$2 iv=$3
  [ -n "$(last_ckpt $OUT/mav6d/centerdet/$tag)" ] && { say "SKIP $tag（已有）"; return; }
  say "START $tag"
  "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
xfer M0_p01 "$M0_CK" 80 100 &
xfer M0_p05 "$M0_CK" 40 20 &
wait
xfer M0_p10 "$M0_CK" 30 10 &
# 从零对照 = 已有的 C 臂（C_p01/p05/p10，real only，同协议）；缺了才补跑
[ -f "$JS/C_p01.json" ] || scratch C_p01 80 100 &
wait
[ -f "$JS/C_p05.json" ] || scratch C_p05 40 20 &
[ -f "$JS/C_p10.json" ] || scratch C_p10 30 10 &
wait

# ---- 阶段 3：评测 ----
for tag in M0_p01 M0_p05 M0_p10 C_p01 C_p05 C_p10; do
  [ -f "$JS/$tag.json" ] && continue
  ck=$(last_ckpt $OUT/mav6d/centerdet/$tag); [ -n "$ck" ] || continue
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差 ' $LOG/eval_$tag.log | head -1)"
done
"$PY" eval_on_mav6d.py --ckpt "$M0_CK" --tag ZM0_mix --decode-max-dis 40 --json "$JS/ZM0_mix.json" > "$LOG/eval_ZM0.log" 2>&1
say "zero-shot M0: $(grep -E '位置误差 ' $LOG/eval_ZM0.log | head -1)"
"$PY" plot_lowdata_curve.py --arms "C:scratch (real only),B:sim near15 pretrain,M0:mixed 3000 aug + 2000 sim" --fracs p01,p05,p10 --out ../docs/results/lowdata_mix.png > "$LOG/plot_mix.log" 2>&1
"$PY" eval_on_mav6d.py --ckpt "$(last_ckpt $OUT/mav6d/centerdet/M0_p05)" --tag M0_p05 --vis 4 --vis-out ../output/vis_mix_mav6d/M0_p05 > /dev/null 2>&1
say "ALL DONE"
