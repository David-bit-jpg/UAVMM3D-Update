#!/bin/bash
# [srcx] stage-1 sim-only detector for a source (recipe = indoor ramp detector, 5 epochs). usage: stage1_det.sh <cfg yaml (rel. to tools)> <seed>
# Used for the outdoor source seeds 1,2 (seed 0 = gproto/outdoor/train_det_out.sh, identical command apart from the seed).
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
CFG=$1; S=$2; X=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx; N=$(basename $CFG .yaml)
CK=/e/Open3DUAVDet/output/models/uavdet_3d/camnorm/$N/DET_s$S/ckpt/checkpoint_epoch_5.pth
[ -f $CK ] && { echo "exists $CK"; exit 0; }
cd /e/Open3DUAVDet/tools
D:/Miniconda3/envs/city/python.exe train.py --cfg_file $CFG \
  --epochs 5 --batch_size 8 --workers 0 --extra_tag DET_s$S --fix_random_seed --seed $S --val_split test --val_interval 10 --val_every 100 --skip_test_eval --max_ckpt_save_num 2 \
  > $X/logs/stage1_det_${N}_s$S.log 2>&1
echo "exit $?" > $X/logs/stage1_det_${N}_s$S.done
