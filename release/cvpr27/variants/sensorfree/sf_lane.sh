#!/bin/bash
# [review1113 sensorfree] one of the TWO sequential lanes of the sensor-free control block (2 jobs in parallel = lane A + lane B).
#   usage: bash sf_lane.sh A|B
# Start gate: the per-head silhouette ablation H3 (heads/H3/mav_s0: .done / .failed / .skipped), polled every 5 min. H2 and H1 run before H3.
#   lane A (MAV6D):     E1 mav s0 -> E1 mav s10 -> E1 mav s20 -> [pooled E1 mav] -> E2a mav s0 -> E2b mav s0
#   lane B (AntiUAV3D): E1 antifs s0 -> E1 antifs s10 -> E1 antifs s20 -> [pooled E1 antifs] -> E2 antifs s10 -> E2 antifs s20 -> [pooled E2 antifs]
#                       -> E4 mav s0 -> E4 antifs s0
# The lane that finishes second writes logs/E.done (the E-block marker) after the pooled E2 MAV6D line (boxhull s0 + heads H7 s10/s20; waits
# for H7's markers, which are produced by the already-started heads_H7 queue line). Each run = run_sf.sh (resume / skip / markers / ALL DONE).
# Single-instance per lane (lock dir with PID). Never kills anything.
LANE=$1; case $LANE in A|B) ;; *) echo "usage: sf_lane.sh A|B"; exit 2 ;; esac
SF=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree; HD=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads
mkdir -p $SF/logs; GL=$SF/logs/sensorfree.log; LK=$SF/logs/lane$LANE.lock
say() { echo "$(date '+%m-%d %H:%M:%S') [lane $LANE] $1" | tee -a $GL; }
alive() { [ -n "$1" ] && powershell -NoProfile -c "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { 'y' }" 2>/dev/null | grep -q y; }
if ! mkdir $LK 2>/dev/null; then
  op=$(cat $LK/pid 2>/dev/null); if alive "$op"; then echo "lane $LANE already running (PID $op)"; exit 0; fi
  say "stale lock (PID ${op:-?} not alive): taking over"; fi
echo $$ > $LK/pid; WPID=$(cat /proc/$$/winpid 2>/dev/null); [ -n "$WPID" ] && echo $WPID > $LK/pid
trap 'rm -rf $LK' EXIT
fin() { local d=$HD/$1; [ -f $d/.done ] || [ -f $d/.failed ] || [ -f $d/.skipped ]; }
n=0; until fin H3/mav_s0; do [ $n = 0 ] && say "waiting for heads H3/mav_s0 (poll 5 min)"; n=1; sleep 300; done
say "H3 finished; lane $LANE starts"
run() { bash $SF/run_sf.sh "$@" > $SF/logs/$1_$2_s$3.stdout 2>&1 || say "!! $* failed (see $SF/$1/$2_s$3/logs/run.log)"; }
if [ $LANE = A ]; then
  run E1 mav 0; run E1 mav 10; run E1 mav 20; bash $SF/run_sf.sh POOL E1 mav
  run E2a mav 0; run E2b mav 0
else
  run E1 antifs 0; run E1 antifs 10; run E1 antifs 20; bash $SF/run_sf.sh POOL E1 antifs
  run E2 antifs 10; run E2 antifs 20; bash $SF/run_sf.sh POOL E2 antifs
  run E4 mav 0; run E4 antifs 0
fi
touch $SF/logs/lane$LANE.done; say "lane $LANE end"
O=A; [ $LANE = A ] && O=B
if [ -f $SF/logs/lane$O.done ]; then
  n=0; until fin H7/mav_s10 && fin H7/mav_s20; do [ $n = 0 ] && say "waiting for heads H7 mav s10/s20 (pooled E2 MAV6D)"; n=1; sleep 300; done
  bash $SF/run_sf.sh POOL E2 mav
  say "[BLOCK] sensor-free block done; pooled lines:"; grep "\[POOLED\]" $GL | tail -12 | sed 's/^/    /' | tee -a $GL > /dev/null
  touch $SF/logs/E.done
fi
