#!/bin/bash
# after the E1/E1b priority block: resume paused jobs (H1 -> H3 -> H7 chain; LiDAR raw-point control) and the rest of the sensor-free block
# in 2 sub-lanes, then the pooled E2 MAV6D line and logs/E.done (which heads_seeds / heads_anti wait for).
SF=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree; R=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113; HD=$R/heads
G=/e/Open3DUAVDet/output/camnorm/geo1004/gproto; GL=$SF/logs/sensorfree.log
say() { echo "$(date '+%m-%d %H:%M:%S') [post] $1" | tee -a $GL; }
run() { bash $SF/run_sf.sh "$@" > $SF/logs/$1_$2_s$3.stdout 2>&1 || say "!! $* failed"; }
fin() { local d=$HD/$1; [ -f $d/.done ] || [ -f $d/.failed ] || [ -f $d/.skipped ]; }
say "priority block done; resuming H1 (its H3/H7 waiters follow) and the rest of the block"
( run E2a mav 0; run E2b mav 0; run E1b mav 20; bash $SF/run_sf.sh POOL E1b mav; touch $SF/logs/postA.done ) &
( run E2 antifs 10; run E2 antifs 20; bash $SF/run_sf.sh POOL E2 antifs; run E4 mav 0; run E4 antifs 0; run E1b antifs 20; bash $SF/run_sf.sh POOL E1b antifs; touch $SF/logs/postB.done ) &
wait
say "rest of block done; resuming the LiDAR raw-point control"
nohup env WORKERS=2 bash $R/lidraw/lidraw_commands.sh >> $G/local_runner_logs_lidar_rawpoint_resume.log 2>&1 &
n=0; until fin H7/mav_s10 && fin H7/mav_s20; do [ $n = 0 ] && say "waiting for heads H7 s10/s20 (pooled E2 MAV6D)"; n=1; sleep 300; done
bash $SF/run_sf.sh POOL E2 mav; touch $SF/logs/E.done; say "[BLOCK] sensor-free block done (E.done)"
