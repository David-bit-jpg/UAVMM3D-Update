#!/bin/bash
# [review1113 heads] follow-up line: the two seed-shift runs (shift 10 = paired with review1113/sup3/run, shift 20 = with seeds/mav_full_s20)
# of H1-H6 (+ the H1c deploy-only diagnostic), MAV6D, strictly one after another, started only after ALL single runs (H1-H7) have finished
# (.done / .failed / .skipped). Order per shift: cheap arms first (H4, H6, H1c, H5), then the full-chain arms (H2, H1, H3).
# Each run = run_arm.sh (gates, deadline guard LATEST_START, resume, done markers, its own pooled 2-3-pair bootstrap at the end).
# Writes heads/logs/seeds.done at the end (gate of heads_anti.sh). NOT RUN by its author.
HD=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads
mkdir -p $HD/logs; GL=$HD/logs/heads.log
say() { echo "$(date '+%m-%d %H:%M:%S') [seeds] $1" | tee -a $GL; }
SINGLES="H4/mav_s0 H5/mav_s0 H6/mav_s0 H1c/mav_s0 H2/mav_s0 H1/mav_s0 H3/mav_s0 H7/mav_s10 H7/mav_s20"
for d in $SINGLES; do n=0; until [ -f $HD/$d/.done ] || [ -f $HD/$d/.failed ] || [ -f $HD/$d/.skipped ]; do
  [ $n = 0 ] && say "waiting for single run $d"; n=1; sleep 600; done; done
say "all single runs finished; starting seed-shift runs"
for s in 10 20; do for a in H4 H6 H1c H5 H2 H1 H3; do
  bash $HD/run_arm.sh $a mav $s > $HD/logs/${a}_mav_s$s.stdout 2>&1 || say "!! $a mav s$s failed (see $HD/$a/mav_s$s/logs/run.log)"
done; done
say "seed-shift runs end"; touch $HD/logs/seeds.done
