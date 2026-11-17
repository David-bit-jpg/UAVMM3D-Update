#!/bin/bash
# 11-12：AntiUAV3D 焦距问题待定；先续跑不受影响的：混合精选第一阶段（只用仿真）两路 + MonoCD 路线 2→3（MAV6D）
X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx; O=$X/mix; H=/e/Open3DUAVDet/output/camnorm/geo1004/cmpfull/run1110
bash $O/slotA.sh > $O/logs/slotA.stdout 2>&1 &
sleep 100; bash $O/slotB.sh > $O/logs/slotB.stdout 2>&1 &
bash $O/finalize_mix.sh > $O/logs/finalize.stdout 2>&1 &
sleep 100; bash $H/monocd_chain.sh > $H/monocd_chain.out 2>&1 &
wait
