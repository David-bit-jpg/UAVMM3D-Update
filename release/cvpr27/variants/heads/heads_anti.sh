#!/bin/bash
# [review1113 heads] last line: AntiUAV3D single runs (reported seed, paired with srcx/out2antifs) of the full-chain refiner arms H2, H1, H3,
# after the MAV6D seed-shift line has finished (heads/logs/seeds.done). Each run starts only if the machine time is before LATEST_START
# (default 11-22 18:00 for full-chain arms: ~3-5 h each must end before 11-23). NOT RUN by its author.
HD=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads
mkdir -p $HD/logs; GL=$HD/logs/heads.log
say() { echo "$(date '+%m-%d %H:%M:%S') [anti] $1" | tee -a $GL; }
n=0; until [ -f $HD/logs/seeds.done ]; do [ $n = 0 ] && say "waiting for heads/logs/seeds.done"; n=1; sleep 600; done
for a in H2 H1 H3; do
  bash $HD/run_arm.sh $a antifs 0 > $HD/logs/${a}_antifs_s0.stdout 2>&1 || say "!! $a antifs s0 failed (see $HD/$a/antifs_s0/logs/run.log)"
done
say "AntiUAV3D runs end"; touch $HD/logs/anti.done
