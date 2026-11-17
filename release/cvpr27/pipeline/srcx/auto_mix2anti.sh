#!/bin/bash
# 11-11: 混合精选第一阶段写出 MIX_STAGE1.md 后自动开 mix→anti 原版链（不能提前开：run_x 在跑会挡住非优先的第一阶段训练）
X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx
until [ -f $X/MIX_STAGE1.md ]; do sleep 300; done
grep -q "| NO" $X/MIX_STAGE1.md && { echo "$(date '+%m-%d %H:%M') mix stage1 has missing checkpoints, not starting" >> $X/mix2antifs/logs/auto.log; exit 3; }
cd $X; bash run_x.sh mix antifs > $X/mix2antifs/logs/run_x.stdout 2>&1
