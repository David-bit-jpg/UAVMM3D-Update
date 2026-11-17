#!/bin/bash
# [review1113 seeds] seed spread for the PST / RGB-only headline numbers: 7 seed-shifted clean-chain runs, STRICTLY ONE AFTER ANOTHER.
# NOT RUN by its author (a scheduler launches it). Each run = run_seed_chain.sh <teach> <src> <tgt> <shift> (see its header / SEEDS_README.md).
#   1 MAV6D     none s10     2 MAV6D     full s20     3 MAV6D     none s20
#   4 AntiUAV3D full s10     5 AntiUAV3D none s10     6 AntiUAV3D full s20     7 AntiUAV3D none s20
#   (MAV6D full s10 = review1113/sup3/run_seed2.sh, already running separately; it is NOT repeated here.)
# Restartable: a run whose logs/run.log already has "ALL DONE:" is skipped; an interrupted run resumes inside run_seed_chain.sh (every step
#   skips once its output exists, trainings resume from their last checkpoint).
# A failed run is logged and the wrapper moves on to the next one (STOP_ON_FAIL=1 stops instead); re-launch the wrapper to retry failed runs.
# WAIT_FOR="<log> ..." : before the first run, wait until each listed log contains "ALL DONE" (e.g. the sup3 seed run, so that no two
#   chains overlap):  WAIT_FOR=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run/logs/run_seed2.log bash run_all_seeds.sh
# DRY=1 : config print for all 7 runs (builds the lib copies only), no GPU, no outputs.
# Estimated wall time (from the reported runs' logs, fixed stage-1 steps excluded, GPU alone): MAV6D full 3.5 h, MAV6D none 3.25 h,
#   AntiUAV3D full 2.9 h, AntiUAV3D none 2.8 h  ->  total ~21.5 h; ~25-28 h with gpu_slot queueing behind other users.
SD=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds
RUNS=("none in mav 10" "full in mav 20" "none in mav 20" "full out antifs 10" "none out antifs 10" "full out antifs 20" "none out antifs 20")
STOP_ON_FAIL=${STOP_ON_FAIL:-0}; DRY=${DRY:-0}
mkdir -p $SD/logs; WL=$SD/logs/run_all_seeds.log
say() { echo "$(date '+%m-%d %H:%M:%S') $1" | tee -a $WL; }
[ $DRY = 1 ] && WL=/dev/null
if [ $DRY != 1 ]; then for w in $WAIT_FOR; do n=0; until grep -q "ALL DONE" "$w" 2>/dev/null; do [ $n = 0 ] && say "waiting for ALL DONE in $w"; n=1; sleep 300; done; done; fi
say "run_all_seeds start (${#RUNS[@]} runs, sequential)"
fails=()
for r in "${RUNS[@]}"; do
  set -- $r; RUN=${3}_${1}_s$4; RL=$SD/$RUN/logs/run.log
  if [ $DRY = 1 ]; then echo "== $RUN  ($r)"; DRY=1 bash $SD/run_seed_chain.sh $r || { echo "   DRY FAILED: $RUN"; fails+=($RUN); }; continue; fi
  if grep -q "ALL DONE:" $RL 2>/dev/null; then say "skip $RUN: $(grep 'ALL DONE:' $RL | tail -1 | cut -c16-)"; continue; fi
  say "launch $RUN ($r)"
  if bash $SD/run_seed_chain.sh $r > $SD/logs/$RUN.stdout 2>&1 && grep -q "ALL DONE:" $RL; then say "finished $RUN: $(grep 'ALL DONE:' $RL | tail -1 | cut -c16-)"
  else say "!! $RUN FAILED (see $RL, $SD/logs/$RUN.stdout)"; fails+=($RUN); [ $STOP_ON_FAIL = 1 ] && break; fi
done
say "run_all_seeds end; failed: ${fails[*]:-none}"
[ ${#fails[@]} = 0 ]
