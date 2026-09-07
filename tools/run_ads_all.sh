#!/usr/bin/env bash
# 用仓库自带的 LAA3D_ADS 指标把所有训练过的权重过一遍（tools/eval_ads_mav6d.py），
# 报告写到 output/ads_metrics/<tag>/{laa,indoor}/report.txt，汇总用 tools/print_ads_table.py。
# 串行，一次一个进程。B/C 两条臂早于按域归一化（2026-09-05 之前训的），必须 --no-norm。
set -u
PY=/d/Miniconda3/envs/city/python.exe
cd /e/Open3DUAVDet/tools || exit 1
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet
LOG=/e/Open3DUAVDet/output/bench_logs
OUT=/e/Open3DUAVDet/output/models/uavdet_3d
ADS=/e/Open3DUAVDet/output/ads_metrics
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/ads_all.log"; }
last_ckpt() { ls "$1"/ckpt/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }

run() {  # tag ckpt_dir [extra...]
  local tag=$1 dir=$2; shift 2
  [ -f "$ADS/$tag/indoor/report.txt" ] && { say "SKIP $tag"; return; }
  local ck; ck=$(last_ckpt "$dir")
  [ -n "$ck" ] || { say "$tag 无 ckpt: $dir"; return; }
  "$PY" eval_ads_mav6d.py --ckpt "$ck" --tag "$tag" "$@" > "$LOG/ads_$tag.log" 2>&1
  say "$tag: $(grep -E '最终 LAA3D_ADS' $LOG/ads_$tag.log | tr -s ' \n' ' ')"
}

M=$OUT/mav6d/centerdet
R=$OUT/mav6d/centerdet_retain
S=$OUT/mmcache

run Z_M0        $S/student_rgb_mix/M0_mix     --decode-max-dis 40
run Z_MT        $S/student_mt_mix/MT_mix      --decode-max-dis 40
run Z_S1        $S/student_kd_mix/S1_mix      --decode-max-dis 40
run Z_S1MT      $S/student_kdmt_mix/S1MT_mix  --decode-max-dis 40
run C_p01       $M/C_p01     --no-norm
run M0_p01      $M/M0_p01
run S1_p01      $M/S1_p01
run S1MT_p01    $M/S1MT_p01
run Cn_p05      $M/Cn_p05
run B_p05       $M/B_p05     --no-norm
run M0_p05      $M/M0_p05
run MT_p05      $M/MT_p05
run S1_p05      $M/S1_p05
run S1MT_p05    $M/S1MT_p05
run RM0_p05     $R/RM0_p05
run ST_S1MT_p05 $M/ST_S1MT_p05
run ST_Cn_p05   $M/ST_Cn_p05
run C_p10       $M/C_p10     --no-norm
run M0_p10      $M/M0_p10
run MT_p10      $M/MT_p10
run S1_p10      $M/S1_p10
run S1MT_p10    $M/S1MT_p10
say "ADS ALL DONE"
