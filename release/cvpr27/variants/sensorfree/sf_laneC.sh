#!/bin/bash
# [review1113 sensorfree, 11-15] lane C: E1b = existing RGB-only arm + ONE change: the refiner's region head is supervised by the projected
# 3-D box hull (SILW = 1) and refiner composites use the hull matte; students keep the IR x LiDAR matte, constant height, one brightness class.
# Splits the RGB-only -> PST gap into "region supervision" (RGB-only -> E1b) and "sensor content" (E1b -> PST / E1 -> PST).
# Order: E1b mav s0 -> antifs s0 -> mav s10 -> antifs s10 -> mav s20 -> antifs s20 -> pooled. Starts when heads H3 has a marker. Kills nothing.
SF=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree; HD=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads
mkdir -p $SF/logs; GL=$SF/logs/sensorfree.log; LK=$SF/logs/laneC.lock
say() { echo "$(date '+%m-%d %H:%M:%S') [lane C] $1" | tee -a $GL; }
mkdir $LK 2>/dev/null || { echo "lane C lock exists ($LK)"; exit 0; }
trap 'rm -rf $LK' EXIT
fin() { local d=$HD/$1; [ -f $d/.done ] || [ -f $d/.failed ] || [ -f $d/.skipped ]; }
n=0; until fin H3/mav_s0; do [ $n = 0 ] && say "waiting for heads H3/mav_s0 (poll 5 min)"; n=1; sleep 300; done
say "H3 finished; lane C (E1b) starts"
run() { bash $SF/run_sf.sh "$@" > $SF/logs/$1_$2_s$3.stdout 2>&1 || say "!! $* failed (see $SF/$1/$2_s$3/logs/run.log)"; }
run E1b mav 0; run E1b antifs 0; run E1b mav 10; run E1b antifs 10; run E1b mav 20; run E1b antifs 20
bash $SF/run_sf.sh POOL E1b mav; bash $SF/run_sf.sh POOL E1b antifs
touch $SF/logs/laneC.done; say "lane C end"
