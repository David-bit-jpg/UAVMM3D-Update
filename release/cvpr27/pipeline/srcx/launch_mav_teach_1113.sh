#!/bin/bash
# [11-13] M3 teacher swaps on MAV6D (source = indoor sim): waits for in2mav (M4) to finish with "ALL DONE", then runs
# SX_TEACH = none only (11-13: ir / lid rows dropped by the user) (one seed each = the chain's default seeds). Method / constants unchanged (see STATUS.md "11-13 MAV6D target").
# A variant that ends with "!!" in its log is reported and the next one still runs. Usage: nohup bash launch_mav_teach_1113.sh > logs/launch_mav_teach_1113.log 2>&1 &
H=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx
say() { echo "$(date '+%m-%d %H:%M:%S') $1"; }
M4=$H/in2mav/logs/run_x.log
say "waiting for ALL DONE in $M4"
until grep -q "ALL DONE" $M4 2>/dev/null; do
  if grep -qE "^[0-9-]+ [0-9:]+ !! " $M4 2>/dev/null; then   # run_x.sh die() line = the chain stopped
    # M4 died: do not start the teacher swaps on a broken base (ir copies the full run's outputs)
    tail -3 $M4; say "in2mav failed -> M3 not started"; exit 2
  fi
  sleep 300
done
say "in2mav ALL DONE"
for t in none; do   # 11-13: user dropped the IR-only / LiDAR-only rows; only RGB-only (none) remains
  mkdir -p $H/in2mav_$t/logs
  say "start SX_TEACH=$t"
  ( cd $H && SX_TEACH=$t bash run_x.sh in mav > $H/in2mav_$t/logs/run_x.stdout 2>&1 )
  say "end SX_TEACH=$t: $(tail -1 $H/in2mav_$t/logs/run_x.log)"
done
say "M3 done"
