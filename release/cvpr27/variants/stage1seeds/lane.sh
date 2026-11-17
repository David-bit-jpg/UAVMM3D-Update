#!/bin/bash
# [review1113 stage1seeds] three lanes = the whole plan (README.md "顺序"). NOT RUN by its author. usage: bash lane.sh <A|B|C>   (DRY=1: print the plan)
#   A, B = the two stage-1 TRAINING slots (each item via gpu_slot.sh heavy; at most one spawn-loader training per lane), then the E4 runs.
#   C    = shared stage-1 outputs + downstream chains of priority 1 (MAV6D PST + E1) and 2 (OutdoorUAV3D PST + E1), in that order; each step
#          waits for exactly the stage-1 networks it needs, so MAV6D E1 sd30 starts as soon as lane A has the indoor detectors + RGB nets.
# Every step is resumable / skipped when done; a failed step is logged and the lane moves on (its dependants then die with a clear message).
S1=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds; LANE=$1; DRY=${DRY:-0}
mkdir -p $S1/logs; LL=$S1/logs/lane_$LANE.log
say() { echo "$(date '+%m-%d %H:%M:%S') [lane $LANE] $1" | tee -a $LL >> $S1/logs/stage1seeds.log; }
TA="det:0 det:1 det:2 rgb:0 rgb:1 irT:0 kdir:0 kdir:1"      # indoor ~15.8 h / outdoor ~4.9 h per seed set (2 trainings at once)
TB="irlid:0 irlid:1 kps:0 lid:0 lid:1"                     # indoor ~17.9 h / outdoor ~4.9 h per seed set
case $LANE in
  A) STEPS=("train_stage1.sh in 30 $TA" "train_stage1.sh in 40 $TA" "train_stage1.sh out 30 $TA" "train_stage1.sh out 40 $TA"
            "run_group.sh E4 mav 30" "run_group.sh E4 antifs 30") ;;
  B) STEPS=("train_stage1.sh in 30 $TB" "train_stage1.sh in 40 $TB" "train_stage1.sh out 30 $TB" "train_stage1.sh out 40 $TB"
            "run_group.sh E4 mav 40" "run_group.sh E4 antifs 40") ;;
  C) STEPS=("prep_shared.sh mav 30" "run_group.sh E1 mav 30" "run_group.sh full mav 30"
            "prep_shared.sh mav 40" "run_group.sh E1 mav 40" "run_group.sh full mav 40"
            "prep_shared.sh antifs 30" "run_group.sh full antifs 30" "run_group.sh E1 antifs 30"
            "prep_shared.sh antifs 40" "run_group.sh full antifs 40" "run_group.sh E1 antifs 40") ;;
  *) echo "usage: lane.sh <A|B|C>"; exit 2 ;;
esac
if [ $DRY = 1 ]; then i=0; for s in "${STEPS[@]}"; do i=$((i + 1)); echo "  lane $LANE step $i: bash $S1/$s"; done; exit 0; fi
say "lane start (${#STEPS[@]} steps)"
for s in "${STEPS[@]}"; do
  say "step: $s"
  bash $S1/$s >> $S1/logs/lane_${LANE}_steps.out 2>&1 && say "ok: $s" || say "!! step returned non-zero: $s (lane continues; see logs)"
done
say "lane end"
