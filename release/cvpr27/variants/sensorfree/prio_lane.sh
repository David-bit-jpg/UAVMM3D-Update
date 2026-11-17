#!/bin/bash
# [11-15 03:xx] user priority: E1 (x3) and E1b (s0) first, 2 lanes in parallel, no gate. P1 = MAV6D, P2 = AntiUAV3D.
# After BOTH lanes are done, prio_post.sh (started by the lane that finishes second) resumes the paused jobs and the rest of the block.
SF=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree; GL=$SF/logs/sensorfree.log
L=$1; T=$([ $L = P1 ] && echo mav || echo antifs)
say() { echo "$(date '+%m-%d %H:%M:%S') [prio $L] $1" | tee -a $GL; }
mkdir $SF/logs/prio$L.lock 2>/dev/null || { echo "prio $L lock exists"; exit 0; }
trap 'rm -rf $SF/logs/prio$L.lock' EXIT
run() { bash $SF/run_sf.sh "$@" > $SF/logs/$1_$2_s$3.stdout 2>&1 || say "!! $* failed (see $SF/$1/$2_s$3/logs/run.log)"; }
say "start ($T): E1 s0 -> s10 -> s20 -> pooled -> E1b s0"
run E1 $T 0; run E1 $T 10; run E1 $T 20; bash $SF/run_sf.sh POOL E1 $T
run E1b $T 0
touch $SF/logs/prio$L.done; say "done"
O=P2; [ $L = P2 ] && O=P1
if [ -f $SF/logs/prio$O.done ] && mkdir $SF/logs/post.lock 2>/dev/null; then nohup bash $SF/prio_post.sh >> $SF/logs/prio_post.stdout 2>&1 & fi
