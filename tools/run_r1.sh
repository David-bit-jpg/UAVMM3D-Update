#!/bin/bash
# 2026-09-16 两臂并行（只差骨干这一项）：
#   P1b = ResNet8x 从零训（归一化统计量已修正）—— 干净基准
#   R1  = ImageNet 预训练 ResNet-34 + FPN —— 本项目从未评过的最大未知数
# 数据用现有的 pp_realsize（与 UE 那边的改动无关，可并行）。
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=/d/Miniconda3/envs/city/python.exe
LOG=/e/Open3DUAVDet/output/camnorm/r1
JS=$LOG/json
mkdir -p $LOG $JS
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG/driver.log; }
train() {   # $1=cfg $2=tag $3=stem
  $PY train.py --cfg_file $1 --batch_size 8 --workers 2 --fix_random_seed --seed 0 \
     --cudnn_benchmark --use_amp --epochs 24 --extra_tag $2 --max_ckpt_save_num 1 --logger_iter_interval 200 \
     --val_split test --val_interval 5 --val_every 2 --val_match greedy --skip_test_eval \
     --set DATA_CONFIG.TRAIN_REPEAT 4 > $LOG/$2.train.log 2>&1
  say "$2 train exit $?"
  CK=/e/Open3DUAVDet/output/models/uavdet_3d/camnorm/$3/$2/ckpt/best.pth
  $PY eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d.yaml --ckpt $CK --tag Z_$2 --split test \
     --workers 0 --json $JS/Z_$2_test.json --ads > $LOG/Z_$2_test.log 2>&1
  say "Z_$2 真实 test 评测 exit $?"
}
say "启动 P1b（ResNet8x 基准）与 R1（ImageNet ResNet-34）"
train cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml P1b sim_pp_realsize &
A=$!
train cfgs/models/uavdet_3d/camnorm/sim_pp_r34.yaml      R1  sim_pp_r34 &
B=$!
wait $A $B
say "ALL DONE"
