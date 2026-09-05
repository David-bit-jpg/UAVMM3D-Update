#!/usr/bin/env bash
# 多模态教师 -> 跨模态蒸馏 RGB 学生 -> 迁到 MAV6D 的整条流水线（等缓存建完后启动）。
#
# 阶段 1  教师(rgb+ir+depth+tag) ∥ 纯RGB学生 S0                 —— 都在仿真 mm20 缓存上
# 阶段 2  蒸馏学生 S1mm(教师=多模态) ∥ 自蒸馏对照 S1rgb(教师=S0)  —— 同一蒸馏方式
# 阶段 3  三个 RGB 权重各自迁到 MAV6D：1% / 5% / 10%，朴素微调（实测「保护预训练」是反效果）
# 阶段 4  统一评测 + 仿真域可视化 + MAV6D 可视化
#
# 严格每次 2 个作业（上次 3-4 个并发把 Windows 的 dataloader 锁死了）。
# batch 用 8 不用 16：两个 batch-16 作业把 24 GB 显存顶满后，Windows 会把 CUDA 显存溢出到
# 系统内存而不是报 OOM —— 实测 30 秒/迭代、磁盘空闲、GPU 100%，一轮要 12 小时。
set -u
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1
export PYTHONPATH='E:/Open3DUAVDet'
LOG=/e/Open3DUAVDet/output/bench_logs
MM=cfgs/models/uavdet_3d/mmcache
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
JS=/e/Open3DUAVDet/output/bench_json
mkdir -p "$LOG" "$JS"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/kd_driver.log"; }
# 机械盘上随机读 memmap 只有 ~90 IOPS（实测 1 s/it、磁盘队列 3.9）。训练前把该 split 的
# 四个 .npy 顺序读一遍（21.7 GB 约 3 分钟），进系统页缓存后随机读走内存。RAM 64 GB 装得下。
prewarm() { for f in rgb ir depth tag; do cat "/e/mmcache/mm20/$1/$f.npy" > /dev/null; done; say "页缓存预热完成: $1"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }

# 等缓存搬到 E 盘（build_mm_cache 先写 C:，搬完后我手动放 READY 标记）
until [ -f /e/mmcache/mm20/READY ]; do sleep 60; done
say "缓存就绪 (E:/mmcache/mm20)"

# ---------------- 阶段 1 ----------------
prewarm train
sim() {  # tag cfg extra...
  local tag=$1 cfg=$2; shift 2
  say "START $tag"
  "$PY" train.py --cfg_file $cfg --batch_size 8 --workers 2 --fix_random_seed --max_ckpt_save_num 2 \
      --epochs 12 --logger_iter_interval 200 --extra_tag "$tag" "$@" > "$LOG/$tag.log" 2>&1
  # ↑ 12 轮（RGB 可见度过滤后帧数减半以上，每轮更短）。此前 8 轮而非配置里的 16：两作业并行各 1.8 it/s、2637 it/轮，16 轮要 6.5 小时、整条线 ~19 小时；
  #   8 轮 ≈ 17 万样本，与 near15 那版 30 轮的样本量相当。
  say "DONE  $tag rc=$?"
}
sim teacher_mm $MM/teacher.yaml &
sim student_rgb $MM/student_rgb.yaml &
wait
T_CK=$(last_ckpt $OUT/mmcache/teacher/teacher_mm)
S0_CK=$(last_ckpt $OUT/mmcache/student_rgb/student_rgb)
say "teacher=$T_CK  s0=$S0_CK"

# ---------------- 阶段 2 ----------------
prewarm train
sim student_kd_mm  $MM/student_kd.yaml            --set MODEL.DISTILL.TEACHER_CKPT "$T_CK" &
sim student_kd_rgb $MM/student_kd_rgbteacher.yaml --set MODEL.DISTILL.TEACHER_CKPT "$S0_CK" &
wait
S1MM_CK=$(last_ckpt $OUT/mmcache/student_kd/student_kd_mm)
S1RGB_CK=$(last_ckpt $OUT/mmcache/student_kd_rgbteacher/student_kd_rgb)
say "s1mm=$S1MM_CK  s1rgb=$S1RGB_CK"

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
for frac in "p01 80 100" "p05 40 20" "p10 30 10"; do
  set -- $frac; f=$1; ep=$2; iv=$3
  xfer K0_$f "$S0_CK"   $ep $iv &
  xfer K1_$f "$S1MM_CK" $ep $iv &
  wait
  xfer K2_$f "$S1RGB_CK" $ep $iv &
  wait
done

# ---------------- 阶段 4：评测与可视化 ----------------
prewarm test
say "评测开始"
for tag in K0_p01 K1_p01 K2_p01 K0_p05 K1_p05 K2_p05 K0_p10 K1_p10 K2_p10; do
  ck=$(last_ckpt $OUT/mav6d/centerdet/$tag); [ -n "$ck" ] || continue
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/$tag.json" > "$LOG/eval_$tag.log" 2>&1
  say "eval $tag: $(grep -E '位置误差 ' $LOG/eval_$tag.log | head -1)"
done
# 零样本：三个 RGB 学生直接测 MAV6D（源域 MAX_DIS=40）
"$PY" eval_on_mav6d.py --ckpt "$S0_CK"   --tag Z0_rgb  --decode-max-dis 40 --json "$JS/Z0_rgb.json"  > "$LOG/eval_Z0.log" 2>&1
"$PY" eval_on_mav6d.py --ckpt "$S1MM_CK" --tag Z1_kdmm --decode-max-dis 40 --json "$JS/Z1_kdmm.json" > "$LOG/eval_Z1.log" 2>&1
"$PY" eval_on_mav6d.py --ckpt "$S1RGB_CK" --tag Z2_kdrgb --decode-max-dis 40 --json "$JS/Z2_kdrgb.json" > "$LOG/eval_Z2.log" 2>&1
# 仿真域可视化：教师 / S0 / S1mm 同一批帧
"$PY" vis_mm_pred.py --cfg $MM/teacher.yaml     --ckpt "$T_CK"    --num 6 --tag teacher --out ../output/vis_kd_sim > "$LOG/vis_sim.log" 2>&1
"$PY" vis_mm_pred.py --cfg $MM/student_rgb.yaml --ckpt "$S0_CK"   --num 6 --tag s0_rgb  --out ../output/vis_kd_sim >> "$LOG/vis_sim.log" 2>&1
"$PY" vis_mm_pred.py --cfg $MM/student_kd.yaml  --ckpt "$S1MM_CK" --num 6 --tag s1_kdmm --out ../output/vis_kd_sim >> "$LOG/vis_sim.log" 2>&1
# MAV6D 可视化：5% 档三臂
for tag in K0_p05 K1_p05 K2_p05; do
  ck=$(last_ckpt $OUT/mav6d/centerdet/$tag); [ -n "$ck" ] || continue
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --vis 4 --vis-out ../output/vis_kd_mav6d/$tag > /dev/null 2>&1
done
say "ALL DONE"
