#!/bin/bash
# 11-12: AntiUAV3D v2（内参核实 + 焦距/距离均衡划分）上的原版 PCMD：室外源 + 室内源并行
X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx; cd $X
bash run_x.sh out antifs > $X/out2antifs/logs/run_x.stdout 2>&1 &
sleep 120
bash run_x.sh in antifs > $X/in2antifs/logs/run_x.stdout 2>&1 &
wait
