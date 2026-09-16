#!/bin/bash
# 2026-09-16 N1：回归头加 bias + 真值自洽标准化。与 P1b 只差这两项（同数据同协议同种子）。
# 评测用 mav6d_std_pp.yaml —— 解码必须用【训练域】的 mu/sigma。
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=/d/Miniconda3/envs/city/python.exe
LOG=/e/Open3DUAVDet/output/camnorm/n1; JS=$LOG/json
mkdir -p $LOG $JS
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG/driver.log; }
say "启动 N1（加 bias + 自洽标准化）"
$PY train.py --cfg_file cfgs/models/uavdet_3d/camnorm/sim_pp_std.yaml --batch_size 8 --workers 2 \
   --fix_random_seed --seed 0 --cudnn_benchmark --use_amp --epochs 24 --extra_tag N1 \
   --max_ckpt_save_num 1 --logger_iter_interval 200 \
   --val_split test --val_interval 5 --val_every 2 --val_match greedy --skip_test_eval \
   --set DATA_CONFIG.TRAIN_REPEAT 4 > $LOG/N1.train.log 2>&1
say "N1 train exit $?"
CK=/e/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_pp_std/N1/ckpt/best.pth
$PY eval_camnorm.py --cfg cfgs/models/uavdet_3d/camnorm/mav6d_std_pp.yaml --ckpt $CK --tag Z_N1 \
   --split test --workers 0 --json $JS/Z_N1_test.json --ads > $LOG/Z_N1_test.log 2>&1
say "Z_N1 真实 test 评测 exit $?"
say "ALL DONE"
