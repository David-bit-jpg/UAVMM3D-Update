#!/bin/bash
# 2026-09-16 两臂并行（≤2 个 spawn dataloader 训练进程）：
#   P1b = 只修归一化统计量（对照，裁窗 zoom [1,4] 不变）
#   P2  = 修归一化 + 裁窗限 [2.0,3.0] 保住目标细节（检验「仿真有效分辨率不足」这个根因）
# 两臂同数据、同协议、同种子，只差裁窗区间。训练期间不要改数据集/模型代码。
cd /e/Open3DUAVDet/tools
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home
PY=/d/Miniconda3/envs/city/python.exe
LOG=/e/Open3DUAVDet/output/camnorm/pp_v2
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

say "启动 P1b（修归一化，zoom [1,4]）与 P2（修归一化 + zoom [2,3]）"
train cfgs/models/uavdet_3d/camnorm/sim_pp_realsize.yaml P1b sim_pp_realsize &
P1=$!
train cfgs/models/uavdet_3d/camnorm/sim_pp_sharp.yaml   P2  sim_pp_sharp &
P2=$!
wait $P1 $P2
say "ALL DONE"
