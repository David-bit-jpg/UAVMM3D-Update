#!/usr/bin/env bash
# 尺度对齐缓存（mm20c）+ 背景翻译缓存（mm20c_bgx）上的整条流水线。与 run_kd_pipeline.sh 同结构，多一条 bgx 臂。
# 【未经用户拍板不要启动】——用户 2026-09-06 说「先停，我再想想」。启动前确认 GPU 上没有别的训练。
#
# 阶段 1  教师(rgb+ir+depth+tag) ∥ 纯RGB学生 S0                      —— mm20c
# 阶段 2  蒸馏学生 S1mm(学生看原始RGB) ∥ 蒸馏学生 S1bgx(学生看翻译RGB，教师看原始)  —— mm20c / mm20c_bgx
# 阶段 3  三个 RGB 权重 + 从零对照 各自迁到 MAV6D：1% / 5% / 10%，朴素微调
# 阶段 4  统一评测 + 零样本 + 可视化
# 严格每次 2 个作业；batch 8（见 run_kd_pipeline.sh 头部说明）。
set -u
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1
export PYTHONPATH='E:/Open3DUAVDet'
LOG=/e/Open3DUAVDet/output/bench_logs
MM=cfgs/models/uavdet_3d/mmcache
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
JS=/e/Open3DUAVDet/output/bench_json
CACHE=/e/mmcache/mm20c
BGX=/e/mmcache/mm20c_bgx
mkdir -p "$LOG" "$JS"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/kd_crop_driver.log"; }
prewarm() { for f in rgb ir depth tag; do cat "$1/$2/$f.npy" > /dev/null; done; say "页缓存预热完成: $1/$2"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }

until [ -f "$CACHE/test/vis_score.npy" ]; do sleep 60; done     # 缓存 + 可见度分数就绪（run_bgx_translate.sh demo 阶段算的）
say "缓存就绪 ($CACHE)"

# ---------------- 阶段 1 ----------------
prewarm "$CACHE" train
sim() {  # tag cfg extra...
  local tag=$1 cfg=$2; shift 2
  say "START $tag"
  "$PY" train.py --cfg_file $cfg --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --epochs 12 --logger_iter_interval 200 --extra_tag "$tag" "$@" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
sim teacher_crop $MM/teacher_crop.yaml &
sim student_rgb_crop $MM/student_rgb_crop.yaml &
wait
T_CK=$(last_ckpt $OUT/mmcache/teacher_crop/teacher_crop)
S0_CK=$(last_ckpt $OUT/mmcache/student_rgb_crop/student_rgb_crop)
say "teacher=$T_CK  s0=$S0_CK"

# ---------------- 阶段 2 ----------------
until [ -f "$BGX/READY" ]; do say "等翻译缓存 $BGX/READY"; sleep 300; done
prewarm "$CACHE" train; prewarm "$BGX" train
sim student_kd_crop $MM/student_kd_crop.yaml --set MODEL.DISTILL.TEACHER_CKPT "$T_CK" &
sim student_kd_bgx  $MM/student_kd_bgx.yaml  --set MODEL.DISTILL.TEACHER_CKPT "$T_CK" &
wait
S1MM_CK=$(last_ckpt $OUT/mmcache/student_kd_crop/student_kd_crop)
S1BGX_CK=$(last_ckpt $OUT/mmcache/student_kd_bgx/student_kd_bgx)
say "s1mm=$S1MM_CK  s1bgx=$S1BGX_CK"

# ---------------- 阶段 3：迁到 MAV6D ----------------
MAV=cfgs/models/uavdet_3d/mav6d/centerdet.yaml
xfer() {  # tag src_ckpt epochs interval
  local tag=$1 src=$2 ep=$3 iv=$4
  say "START $tag"
  "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
      --pretrained_model "$src" --pretrained_skip hm --pretrained_src_max_dis 40 \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
scratch() {  # tag epochs interval
  local tag=$1 ep=$2 iv=$3
  say "START $tag"
  "$PY" train.py --cfg_file $MAV --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --logger_iter_interval 200 --epochs "$ep" --extra_tag "$tag" \
      --set DATA_CONFIG.DATA_PATH E:/MAV6D DATA_CONFIG.SAMPLED_INTERVAL.train "$iv" > "$LOG/$tag.log" 2>&1
  say "DONE  $tag rc=$?"
}
for frac in "p01 80 100" "p05 40 20" "p10 30 10"; do
  set -- $frac; f=$1; ep=$2; iv=$3
  xfer C0_$f "$S0_CK"    $ep $iv &
  xfer C1_$f "$S1MM_CK"  $ep $iv &
  wait
  xfer C4_$f "$S1BGX_CK" $ep $iv &
  scratch C3_$f $ep $iv &
  wait
done

# ---------------- 阶段 4：评测与可视化 ----------------
prewarm "$CACHE" test
for tag in C0_p01 C1_p01 C4_p01 C3_p01 C0_p05 C1_p05 C4_p05 C3_p05 C0_p10 C1_p10 C4_p10 C3_p10; do
  ck=$(last_ckpt $OUT/mav6d/centerdet/$tag); [ -n "$ck" ] || continue
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差 ' $LOG/eval_$tag.log | head -1)"
done
"$PY" eval_on_mav6d.py --ckpt "$S0_CK"    --tag ZC0_rgb  --decode-max-dis 40 --json "$JS/ZC0_rgb.json"  > "$LOG/eval_ZC0.log" 2>&1
"$PY" eval_on_mav6d.py --ckpt "$S1MM_CK"  --tag ZC1_kdmm --decode-max-dis 40 --json "$JS/ZC1_kdmm.json" > "$LOG/eval_ZC1.log" 2>&1
"$PY" eval_on_mav6d.py --ckpt "$S1BGX_CK" --tag ZC4_kdbgx --decode-max-dis 40 --json "$JS/ZC4_kdbgx.json" > "$LOG/eval_ZC4.log" 2>&1
"$PY" vis_mm_pred.py --cfg $MM/teacher_crop.yaml     --ckpt "$T_CK"     --num 6 --tag teacher --out ../output/vis_kd_crop > "$LOG/vis_crop.log" 2>&1
"$PY" vis_mm_pred.py --cfg $MM/student_rgb_crop.yaml --ckpt "$S0_CK"    --num 6 --tag s0_rgb  --out ../output/vis_kd_crop >> "$LOG/vis_crop.log" 2>&1
"$PY" vis_mm_pred.py --cfg $MM/student_kd_crop.yaml  --ckpt "$S1MM_CK"  --num 6 --tag s1_kdmm --out ../output/vis_kd_crop >> "$LOG/vis_crop.log" 2>&1
"$PY" vis_mm_pred.py --cfg $MM/student_kd_bgx.yaml   --ckpt "$S1BGX_CK" --num 6 --tag s1_kdbgx --out ../output/vis_kd_crop >> "$LOG/vis_crop.log" 2>&1
for tag in C0_p05 C1_p05 C4_p05 C3_p05; do
  ck=$(last_ckpt $OUT/mav6d/centerdet/$tag); [ -n "$ck" ] || continue
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --vis 4 --vis-out ../output/vis_kd_mav6d_crop/$tag > /dev/null 2>&1
done
say "ALL DONE"
