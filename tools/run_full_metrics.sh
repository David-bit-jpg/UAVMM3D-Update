#!/usr/bin/env bash
# 用户 2026-09-08：「表里没体现性能好坏，要 3D 检测精度、姿态准确率」。
# 用补全后的 eval_on_mav6d.py（检出率 / 位置与朝向准确率 / 5cm5deg / ADD / 2D 重投影，分母 = 全部 4800 帧）
# 把主要权重重测一遍，结果写到 output/bench_json/full_<tag>.json。串行跑，一次一个进程。
# 注意：B/C 两条臂是 2026-09-05 之前训的，早于数据集加入按域归一化，必须 --no-norm 评测。
set -u
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
LOG=/e/Open3DUAVDet/output/bench_logs
JS=/e/Open3DUAVDet/output/bench_json
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/full_metrics.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }

run() {  # tag ckpt_dir [extra args...]
  local tag=$1 dir=$2; shift 2
  [ -f "$JS/full_$tag.json" ] && { say "SKIP $tag"; return; }
  local ck; ck=$(last_ckpt "$dir")
  [ -n "$ck" ] || { say "$tag 无 ckpt: $dir"; return; }
  "$PY" eval_on_mav6d.py --ckpt "$ck" --tag "$tag" --json "$JS/full_$tag.json" "$@" > "$LOG/full_$tag.log" 2>&1
  say "$tag: $(grep -E '位置 < 0.2|检出率' $LOG/full_$tag.log | tr -s ' \n' ' ')"
}

M=$OUT/mav6d/centerdet
R=$OUT/mav6d/centerdet_retain
S=$OUT/mmcache

# ---- 5% 预算（主对比）----
run Cn_p05        $M/Cn_p05
run C_p05         $M/C_p05        --no-norm
run B_p05         $M/B_p05        --no-norm
run M0_p05        $M/M0_p05
run MT_p05        $M/MT_p05
run S1_p05        $M/S1_p05
run S1MT_p05      $M/S1MT_p05
run RM0_p05       $R/RM0_p05
run ST_S1MT_p05   $M/ST_S1MT_p05
run ST_Cn_p05     $M/ST_Cn_p05
# ---- 10% ----
run C_p10         $M/C_p10        --no-norm
run M0_p10        $M/M0_p10
run MT_p10        $M/MT_p10
run S1_p10        $M/S1_p10
run S1MT_p10      $M/S1MT_p10
# ---- 1% ----
run C_p01         $M/C_p01        --no-norm
run M0_p01        $M/M0_p01
run S1_p01        $M/S1_p01
run S1MT_p01      $M/S1MT_p01
# ---- 零样本（只用仿真）----
run Z_M0          $S/student_rgb_mix/M0_mix       --decode-max-dis 40
run Z_MT          $S/student_mt_mix/MT_mix        --decode-max-dis 40
run Z_S1          $S/student_kd_mix/S1_mix        --decode-max-dis 40
run Z_S1MT        $S/student_kdmt_mix/S1MT_mix    --decode-max-dis 40
say "FULL METRICS DONE"
