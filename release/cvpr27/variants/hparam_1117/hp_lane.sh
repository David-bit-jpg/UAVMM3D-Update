#!/bin/bash
# [review1113 hparam_1117] the hyperparameter-sensitivity cells, strictly one after another (= at most one training of ours at a time).
# Started by local_runner_1113.sh (queue line "hparam_1117"). Single instance (lock dir). Every cell skips if its .done exists; a failed cell
# is logged and the lane moves on (re-launching the lane retries failed cells; their trainings resume).
# Deadline guard (results due machine 11-24 19:59): no student cell starts after 11-24 15:00, no lambda chain after 11-24 08:00.
HP=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117
LOCK=$HP/.lane.lock; mkdir $LOCK 2>/dev/null || { echo "hp_lane already running ($LOCK)"; exit 0; }
trap 'rmdir $LOCK' EXIT
GLOG=$HP/logs/hparam.log; mkdir -p $HP/logs
say() { echo "$(date '+%m-%d %H:%M:%S') [lane] $1" >> $GLOG; }
late() { [ $((10#$(date '+%m%d%H%M'))) -gt $((10#$1)) ]; }
say "lane start (PID $$)"
CELLS=(
  # MAV6D, the reported run (shift 0)
  "stu g05 mav_s0" "stu g20 mav_s0" "stu m06 mav_s0" "stu m24 mav_s0"
  "lam 050 mav 0" "lam 100 mav 0"
  # retraining-noise reference of the student protocol
  "stu rep mav_s0"
  # MAV6D repeats on the two other PST runs (shift 10, 20)
  "stu g05 mav_s10" "stu g20 mav_s10" "stu gall mav_s10" "stu m06 mav_s10" "stu m24 mav_s10"
  "stu g05 mav_s20" "stu g20 mav_s20" "stu gall mav_s20" "stu m06 mav_s20" "stu m24 mav_s20"
  # RealDrone3D (antifs), shift 0
  "stu g05 anti_s0" "stu g20 anti_s0" "stu gall anti_s0" "stu m06 anti_s0" "stu m24 anti_s0"
  "lam 050 antifs 0" "lam 100 antifs 0"
)
for c in "${CELLS[@]}"; do
  set -- $c
  if [ $1 = stu ]; then d=$HP/stu/$2/$3; dl=11241500; else d=$HP/lam/${3}_L$2_s$4; dl=11240800; fi
  [ -f $d/.done ] && continue
  if late $dl; then say "skip [$c]: past its latest start $dl"; continue; fi
  say "cell start [$c]"
  if [ $1 = stu ]; then bash $HP/stu_arm.sh $2 $3; else bash $HP/lam_chain.sh $2 $3 $4; fi
  if [ -f $d/.done ]; then say "cell done [$c]: $(grep 'ALL DONE' $d/logs/run.log | tail -1 | sed 's/.*ALL DONE: //' | cut -c1-200)"
  else say "cell FAILED [$c] (see $d/logs/run.log)"; fi
  bash $HP/collect.sh > /dev/null 2>&1
done
say "lane end"
