#!/bin/bash
# 11-11 15:35 机器重启后恢复：out2anti 优先；混合精选 slotA 立即续，slotB 等 out2anti 结束再续（给 out2anti 留一个重训练位）；之后 MonoCD 等待器
X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx; O=$X/mix; H=/e/Open3DUAVDet/output/camnorm/geo1004/cmpfull/run1110
cd $X; bash run_x.sh out anti > $X/out2anti/logs/run_x.stdout 2>&1 &
sleep 60; bash $O/slotA.sh > $O/logs/slotA.stdout 2>&1 &
bash $O/finalize_mix.sh > $O/logs/finalize.stdout 2>&1 &
bash $H/monocd_resume_after_ours.sh > $H/monocd_resume.out 2>&1 &
until grep -q -E "ALL DONE|^!!" $X/out2anti/logs/run_x.log; do sleep 300; done
bash $O/slotB.sh > $O/logs/slotB.stdout 2>&1
wait
