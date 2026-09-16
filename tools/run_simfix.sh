#!/bin/bash
# 仿真标签修正对照（2026-09-15 夜）：S2 = 机头偏航修正，S3 = 机头 + 物理尺度修正；与 S1 同协议（24 轮、种子 0、bf16、每 2 轮仿真 test 抽帧验证选轮）。
# 两个训练并行（上限 2 个 spawn-dataloader 训练进程）；训完各自零样本评 MAV6D val/test（带 ADS）+ 仿真 test。
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=/d/Miniconda3/envs/city/python.exe
LOG=/e/Open3DUAVDet/output/camnorm/simfix
JS=/e/Open3DUAVDet/output/camnorm/simfix/json
mkdir -p $JS
run_arm() {  # $1 tag  $2 cfg stem
  local tag=$1 stem=$2
  echo "=== $tag train start $(date)" >> $LOG/driver.log
  $PY train.py --cfg_file cfgs/models/uavdet_3d/camnorm/$stem.yaml --batch_size 8 --workers 2 --fix_random_seed --seed 0 \
     --cudnn_benchmark --use_amp --epochs 24 --extra_tag $tag --max_ckpt_save_num 1 --logger_iter_interval 200 \
     --val_split test --val_interval 5 --val_every 2 --val_match greedy --skip_test_eval > $LOG/$tag.train.log 2>&1
  echo "=== $tag train exit $? $(date)" >> $LOG/driver.log
}
eval_arm() {  # $1 tag $2 stem
  local tag=$1 stem=$2 ck=/e/Open3DUAVDet/output/models/uavdet_3d/camnorm/$2/$1/ckpt/best.pth
  $PY eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d.yaml --ckpt $ck --tag Z_$tag --split val --workers 0 --json $JS/Z_${tag}_val.json > $LOG/Z_${tag}_val.log 2>&1
  $PY eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d.yaml --ckpt $ck --tag Z_$tag --split test --workers 0 --json $JS/Z_${tag}_test.json --ads > $LOG/Z_${tag}_test.log 2>&1
  $PY eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/$stem.yaml --ckpt $ck --tag ${tag}__simtest --split test --workers 0 --json $JS/${tag}__simtest.json --ads --set DATA_CONFIG.VAL_ZOOMS "[1.0]" > $LOG/${tag}__simtest.log 2>&1
  echo "=== $tag eval done $(date)" >> $LOG/driver.log
}
cat /e/mmcache/indoor8jpg/train/rgb_jpg.bin /e/mmcache/indoor8jpg/test/rgb_jpg.bin > /dev/null; echo "=== cache warmed $(date)" >> $LOG/driver.log
( run_arm S2 sim_indoor8_mz_nose ) &
P2=$!
sleep 90
( run_arm S3 sim_indoor8_mz_nosescale ) &
P3=$!
wait $P2
eval_arm S2 sim_indoor8_mz_nose
wait $P3
eval_arm S3 sim_indoor8_mz_nosescale
echo "=== ALL DONE $(date)" >> $LOG/driver.log
