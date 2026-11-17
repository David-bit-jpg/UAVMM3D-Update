#!/bin/bash
# [11-15 02:5x] reordered priority lane: E1 s0 (already running, started by prio_lane.sh) -> E1b s0 -> E1 s10 -> E1 s20 -> pooled E1 -> post.
SF=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree; GL=$SF/logs/sensorfree.log
L=$1; T=$([ $L = P1 ] && echo mav || echo antifs)
say() { echo "$(date '+%m-%d %H:%M:%S') [prio2 $L] $1" | tee -a $GL; }
fin() { [ -f $SF/$1/.done ] || [ -f $SF/$1/.failed ] || [ -f $SF/$1/.skipped ]; }
run() { bash $SF/run_sf.sh "$@" > $SF/logs/$1_$2_s$3.stdout 2>&1 || say "!! $* failed (see $SF/$1/$2_s$3/logs/run.log)"; }
n=0; until fin E1/${T}_s0; do [ $n = 0 ] && say "waiting for the running E1 ${T} s0"; n=1; sleep 120; done
run E1 $T 10; run E1 $T 20; bash $SF/run_sf.sh POOL E1 $T; run E1b $T 0
touch $SF/logs/prio$L.done; say "done"
O=P2; [ $L = P2 ] && O=P1
if [ -f $SF/logs/prio$O.done ] && mkdir $SF/logs/post.lock 2>/dev/null; then nohup bash $SF/prio_post.sh >> $SF/logs/prio_post.stdout 2>&1 & fi
