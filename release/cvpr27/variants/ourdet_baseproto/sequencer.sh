#!/bin/bash
# 17:10 remaining launches (outdoor row1 + ST launched by hand 17:09; both UBs running). gpu_slot memory gate protects host RAM.
H=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto; cd $H
MR=/d/S2MUAV3D/ourdet_baseproto_models/models/uavdet_3d/camnorm/ourdet_baseproto
w() { until [ -f "$1" ]; do sleep 60; done; }
L() { echo "$(date '+%m-%d %H:%M:%S') [seq] launch $*" >> logs/run.log; nohup bash $H/$1 ${@:2} > logs/seq_$1_$2_$3.out 2>&1 & sleep 5; }
w $MR/out2antifs_ub/DET_s0/ckpt/checkpoint_epoch_8.pth;   L row1.sh in2mav; L train_eval.sh in2mav st
echo "$(date '+%m-%d %H:%M:%S') [seq] all launched" >> logs/run.log
